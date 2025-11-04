import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d, FutureLinearPts3d, AutoregressiveFuturePts3d
from .layers.improved_temporal_head import ImprovedFutureLinearPts3d
from .layers.camera_head import CameraHead, FutureCameraHead
from .layers.autoregressive_transformer import AutoregressiveTokenTransformer
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin


DINOV3_WEIGHTS = '/home/matthew_strong/Desktop/test_vfms/RADIO/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth'


# new future prediction model
# similar architecture to Pi3

# difference is the inclusion of a n+m camera pose decoder that predicits the next few camera poses
class AutonomyPi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            full_N=6,
            extra_tokens=3,
            encoder_name='dinov3',
            dinov3_checkpoint_path=None,
            use_motion_head=True, # using motion head for present and future prediction
            use_flow_head=False, # using optical flow head for present and future prediction
            use_segmentation_head=True, # using segmentation head for present and future prediction
            segmentation_num_classes=6,  # Number of segmentation classes
            use_detection_head=False,
            num_detection_classes=2,  # traffic light, road sign
            detection_architecture='dense',  # 'dense' or 'detr'
            num_object_queries=100,
            detr_hidden_dim=256,
            detr_num_heads=8,
            detr_num_layers=6,
            future_prediction_type='linear',  # 'linear' or 'autoregressive'
            autoregressive_n_heads=16,
            autoregressive_n_layers=6,
            autoregressive_dropout=0.1,
            freeze_decoders=False,  # Freeze point, conf, and camera decoders
        ):
        super().__init__()
        
        # Store future prediction configuration
        self.future_prediction_type = future_prediction_type
        self.autoregressive_n_heads = autoregressive_n_heads
        self.autoregressive_n_layers = autoregressive_n_layers
        self.autoregressive_dropout = autoregressive_dropout
        
        # ----------------------
        #        Encoder
        # ----------------------
        # Use provided checkpoint path or fallback to hardcoded path
        weights_path = dinov3_checkpoint_path or DINOV3_WEIGHTS
        
        if encoder_name == 'dinov3':
            self.encoder = torch.hub.load('dinov3', 'dinov3_vitl16', source='local', weights=weights_path)
            self.encoder.train()
            self.patch_size = 16
        elif encoder_name == 'dinov2':
            self.encoder = dinov2_vitl14_reg(pretrained=False)
            self.patch_size = 14
            del self.encoder.mask_token
        else:
            raise ValueError(f"Unsupported encoder_name: {encoder_name}. Choose 'dinov2' or 'dinov3'")

        self.full_N = full_N
        self.extra_tokens = extra_tokens

        self.use_motion_head = use_motion_head
        self.use_flow_head = use_flow_head
        self.use_segmentation_head = use_segmentation_head
        self.segmentation_num_classes = segmentation_num_classes

        self.use_detection_head = use_detection_head
        self.num_detection_classes = num_detection_classes
        self.detection_architecture = detection_architecture
        self.num_object_queries = num_object_queries
        self.detr_hidden_dim = detr_hidden_dim
        self.detr_num_heads = detr_num_heads
        self.detr_num_layers = detr_num_layers

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )

        # Choose future prediction method
        if self.future_prediction_type == 'autoregressive':
            self.point_head = AutoregressiveFuturePts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=3, 
                extra_tokens=self.extra_tokens,
                n_heads=self.autoregressive_n_heads,
                n_layers=self.autoregressive_n_layers,
                dropout=self.autoregressive_dropout
            )
        else:  # default to 'linear'
            self.point_head = ImprovedFutureLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=3, 
                extra_tokens=self.extra_tokens
            )

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        # Choose future prediction method for confidence
        if self.future_prediction_type == 'autoregressive':
            self.conf_head = AutoregressiveFuturePts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=1, 
                extra_tokens=self.extra_tokens,
                n_heads=self.autoregressive_n_heads,
                n_layers=self.autoregressive_n_layers,
                dropout=self.autoregressive_dropout
            )
        else:  # default to 'linear'
            self.conf_head = ImprovedFutureLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=1, 
                extra_tokens=self.extra_tokens
            )


        if self.use_segmentation_head:
            # composite segmentation decoder for predicting segmentation masks
            self.segmentation_decoder = deepcopy(self.point_decoder)
            
            self.segmentation_head = ImprovedFutureLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=self.segmentation_num_classes,
                extra_tokens=self.extra_tokens
            )


        if self.use_motion_head:
            # motion decoder for predicting current and future motion
            self.motion_decoder = deepcopy(self.point_decoder)

            # motion head is a copy of point head
            self.motion_head = ImprovedFutureLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=1,  # Binary motion mask (0=static, 1=moving)
                extra_tokens=self.extra_tokens
            )


        if self.use_flow_head:
            # flow decoder for predicting optical flow
            self.flow_decoder = deepcopy(self.point_decoder)

            # flow head outputs 2D optical flow (dx, dy)
            self.flow_head = ImprovedFutureLinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=2,  # 2D optical flow (dx, dy)
                extra_tokens=self.extra_tokens
            )
            

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = FutureCameraHead(dim=512, N=self.full_N - extra_tokens, M=extra_tokens)



        # ----------------------
        #   Detection Head (Optional)
        # ----------------------
        if self.use_detection_head:
            # Import here to avoid circular imports
            import sys
            import os
            sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            
            # Check detection architecture type
            detection_architecture = getattr(self, 'detection_architecture', 'dense')
            
            if detection_architecture == 'detr':
                # DETR-style detection
                from detr_components import DETRDetectionModule
                
                num_queries = getattr(self, 'num_object_queries', 100)
                detr_hidden_dim = getattr(self, 'detr_hidden_dim', 256)
                detr_num_heads = getattr(self, 'detr_num_heads', 8)
                detr_num_layers = getattr(self, 'detr_num_layers', 6)
                
                self.detr_detection = DETRDetectionModule(
                    input_dim=2*self.dec_embed_dim,
                    hidden_dim=detr_hidden_dim,
                    num_queries=num_queries,
                    num_classes=self.num_detection_classes,
                    num_heads=detr_num_heads,
                    num_layers=detr_num_layers
                )
            else:
                # Dense grid detection (original)
                self.detection_decoder = TransformerDecoder(
                    in_dim=2*self.dec_embed_dim, 
                    dec_embed_dim=1024,
                    dec_num_heads=16,
                    out_dim=512,
                    rope=self.rope,
                    use_checkpoint=False
                )
                # Detection head: num_classes + 4 bbox coordinates (x, y, w, h)
                self.detection_head = nn.Sequential(
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Linear(256, self.num_detection_classes + 4)
                )

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        # Freeze decoders if requested
        if freeze_decoders:
            self._freeze_decoders()

    def _freeze_decoders(self):
        """Freeze point, conf, and camera decoders and heads to force better token representation learning.
        NOTE: Segmentation decoder/head are NOT frozen even when freeze_decoders=True, 
        as they need to learn from scratch."""
        # Freeze decoders
        for param in self.point_decoder.parameters():
            param.requires_grad = False
        for param in self.conf_decoder.parameters():
            param.requires_grad = False
        for param in self.camera_decoder.parameters():
            param.requires_grad = False
        
        # Freeze heads
        for param in self.point_head.parameters():
            param.requires_grad = False
        for param in self.conf_head.parameters():
            param.requires_grad = False
        for param in self.camera_head.parameters():
            param.requires_grad = False
        
        # NOTE: Segmentation, motion, and flow decoders/heads are intentionally NOT frozen
        # They need to train from scratch even when other decoders are frozen
        
        print("Froze point, conf, and camera decoders and heads")
        if self.use_segmentation_head:
            print("  ℹ️  Segmentation decoder/head remain trainable (not frozen)")
        if self.use_motion_head:
            print("  ℹ️  Motion decoder/head remain trainable (not frozen)")
        if self.use_flow_head:
            print("  ℹ️  Flow decoder/head remain trainable (not frozen)")
    
    def verify_gradient_flow(self):
        """Verify which parts of the model have gradients enabled."""
        components = {
            'encoder': self.encoder,
            'point_decoder': self.point_decoder,
            'point_head': self.point_head,
            'conf_decoder': self.conf_decoder,
            'conf_head': self.conf_head,
            'camera_decoder': self.camera_decoder,
            'camera_head': self.camera_head,
        }
        
        # Add optional heads
        if self.use_motion_head:
            components['motion_decoder'] = self.motion_decoder
            components['motion_head'] = self.motion_head
            
        if self.use_flow_head:
            components['flow_decoder'] = self.flow_decoder
            components['flow_head'] = self.flow_head
            
        if self.use_segmentation_head:
            components['segmentation_decoder'] = self.segmentation_decoder
            components['segmentation_head'] = self.segmentation_head
        
        # Add autoregressive transformer if it exists
        if hasattr(self, 'autoregressive_transformer'):
            components['autoregressive_transformer'] = self.autoregressive_transformer
        
        for name, module in components.items():
            params_with_grad = sum(p.requires_grad for p in module.parameters())
            total_params = sum(1 for p in module.parameters())
            status = "✓" if params_with_grad > 0 else "✗"
            print(f"{status} {name}: {params_with_grad}/{total_params} params with gradients")

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size


        # encode with selected encoder
        imgs = imgs.reshape(B*N, _, H, W)
        if hasattr(self.encoder, 'forward_features'):
            # DINOv3 path
            hidden = self.encoder.forward_features(imgs)
        else:
            # DINOv2 path
            hidden = self.encoder(imgs, is_training=True)

        # Extract intermediate layer features
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]


        hidden, pos = self.decode(hidden, N, H, W)

        # with hidden state, we can add a detection head!

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        if self.use_segmentation_head:
            segmentation_hidden = self.segmentation_decoder(hidden, xpos=pos)


        if self.use_motion_head:
            motion_hidden = self.motion_decoder(hidden, xpos=pos)


        if self.use_flow_head:
            flow_hidden = self.flow_decoder(hidden, xpos=pos)
        
        # Optional detection head processing
        detection_hidden = None
        if self.use_detection_head:
            if self.detection_architecture == 'detr':
                # For DETR, we don't need to process through detection_decoder
                # We'll use the hidden features directly in the forward pass
                detection_hidden = hidden  # Store for later DETR processing
            else:
                # Dense grid detection (original path)
                detection_hidden = self.detection_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points - now returns [B*(N+M), H, W, output_dim]
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W), B, N)
            # local_points_flat is [B*(N+M), H, W, 3]
            total_frames = N + self.extra_tokens  # N current + M future
            local_points_raw = local_points_flat.reshape(B, total_frames, H, W, -1)
            
            xy, z = local_points_raw.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence - same temporal structure [B*(N+M), H, W, 1]
            conf_hidden = conf_hidden.float()
            conf_flat = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W), B, N)
            conf = conf_flat.reshape(B, total_frames, H, W, -1)


            ##### motion - same temporal structure [B*(N+M), H, W, 1]
            if self.use_motion_head:
                # motion prediction 
                motion_hidden = motion_hidden.float()
                motion_flat = self.motion_head([motion_hidden[:, self.patch_start_idx:]], (H, W), B, N)
                motion = motion_flat.reshape(B, total_frames, H, W, -1)
            else:
                motion = None

            ##### flow - same temporal structure [B*(N+M), H, W, 2]
            if self.use_flow_head:
                # optical flow prediction 
                flow_hidden = flow_hidden.float()
                flow_flat = self.flow_head([flow_hidden[:, self.patch_start_idx:]], (H, W), B, N)
                flow = flow_flat.reshape(B, total_frames, H, W, -1)
            else:
                flow = None
                

            # segmentation prediction [B*(N+M), H, W, 9]
            if self.use_segmentation_head:
                segmentation_hidden = segmentation_hidden.float()
                segmentation_flat = self.segmentation_head([segmentation_hidden[:, self.patch_start_idx:]], (H, W), B, N)
                segmentation = segmentation_flat.reshape(B, total_frames, H, W, -1)
            else:
                segmentation = None

            # camera poses - now returns [B*(N+M), 4, 4]
            camera_hidden = camera_hidden.float()
            camera_poses_flat = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w, B, N)
            camera_poses = camera_poses_flat.reshape(B, total_frames, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

            # Optional detection predictions
            detections = None
            if self.use_detection_head and detection_hidden is not None:
                detection_hidden = detection_hidden.float()
                
                if self.detection_architecture == 'detr':
                    # DETR-style detection
                    # Use only current N frames (not future frames) for detection
                    BN_total, hw, feat_dim = detection_hidden.shape
                    current_frames = N  # Only use current frames for detection
                    detection_features = detection_hidden[:B*current_frames]  # [B*N, hw, feat_dim]
                    
                    # Reshape to [B*N, feat_dim, H, W] for DETR input
                    detection_features = detection_features.transpose(1, 2).reshape(B*current_frames, feat_dim, patch_h, patch_w)
                    
                    # Process through DETR for each frame
                    detr_outputs = []
                    for i in range(current_frames):
                        frame_features = detection_features[i::current_frames]  # [B, feat_dim, H, W]
                        frame_output = self.detr_detection(frame_features)
                        detr_outputs.append(frame_output)
                    
                    # Combine outputs: class_logits [B, N, num_queries, num_classes+1], bbox_preds [B, N, num_queries, 4]
                    class_logits = torch.stack([out['class_logits'] for out in detr_outputs], dim=1)
                    bbox_preds = torch.stack([out['bbox_preds'] for out in detr_outputs], dim=1)
                    
                    detections = {
                        'class_logits': class_logits,  # [B, N, num_queries, num_classes+1] 
                        'bbox_preds': bbox_preds       # [B, N, num_queries, 4]
                    }
                else:
                    # Dense grid detection (original path)
                    # Apply detection head to get predictions: [B*N, patch_tokens, num_classes+4]
                    detection_logits = self.detection_head(detection_hidden[:, self.patch_start_idx:])
                    
                    # detection_hidden shape is [B*N, patch_tokens, features] - only covers current N frames, not total_frames
                    # So we reshape using N, not total_frames
                    BN, patch_tokens, _ = detection_logits.shape
                    actual_N = BN // B  # This should equal N (current frames only)
                    detections = detection_logits.reshape(B, actual_N, patch_h, patch_w, self.num_detection_classes + 4)

        result_dict = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses
        )
        
        if self.use_segmentation_head and segmentation is not None:
            result_dict["segmentation"] = segmentation
        
        if self.use_motion_head:
            result_dict["motion"] = motion
        
        if self.use_flow_head:
            result_dict["flow"] = flow
        
        if detections is not None:
            result_dict["detections"] = detections
            
        return result_dict



class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def get_decoder_features(self, imgs):
        """
        Extract decoder features for teacher-student supervision.
        
        Args:
            imgs: [B, N, C, H, W] - input images
            
        Returns:
            dict with decoder features and positional encoding
        """
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        
        # Encode images
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        
        # Get decoder features
        decoder_features, pos = self.decode(hidden, N, H, W)
        
        return {
            'decoder_features': decoder_features,  # [B*N, S, D]
            'positional_encoding': pos,            # [B*N, S, pos_dim]
            'batch_info': {'B': B, 'N': N, 'H': H, 'W': W}
        }
    
    def forward(self, imgs, return_features=False):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        dino_features = hidden

        hidden, pos = self.decode(hidden, N, H, W)
        pi3_features = hidden

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            features=hidden,  # [B*N, S, D] - decoder features for potential supervision
            pos=pos,           # [B*N, S, pos_dim] - positional encoding for
            dino_features=dino_features,  # [B*N, S, D] - DINOv2 encoder features for potential supervision
            pi3_features=pi3_features  # [B*N, S, D] - Pi3 decoder features for potential supervision
        )


class AutoregressivePi3(nn.Module, PyTorchModelHubMixin):
    """
    Pi3 with autoregressive transformer for future token generation.
    Scene tokens are generated autoregressively after encoder and decode.
    """
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
            encoder_name='dinov2',
            n_future_frames=3,
            ar_n_heads=16,
            ar_n_layers=8,
            ar_dropout=0.1,
            freeze_decoders=False,  # Freeze point, conf, and camera decoders
            use_segmentation_head=False,  # Enable segmentation head
            segmentation_num_classes=6,  # Number of segmentation classes
            use_motion_head=True,  # Enable motion head
            use_flow_head=False,  # Enable optical flow head
        ):
        super().__init__()

        self.use_segmentation_head = use_segmentation_head
        self.use_motion_head = use_motion_head
        self.use_flow_head = use_flow_head
        self.segmentation_num_classes = segmentation_num_classes

        # ----------------------
        #        Encoder
        # ----------------------
        if encoder_name == 'dinov2':
            self.encoder = dinov2_vitl14_reg(pretrained=False)
            self.patch_size = 14
            del self.encoder.mask_token
        else:
            raise NotImplementedError(f"Encoder {encoder_name} not implemented")

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope = None
        if self.pos_type.startswith('rope'):
            if RoPE2D is None:
                raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features  # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        
        # VGGT style alternating attention heads
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim

        # ----------------------
        #     Register tokens
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        # Autoregressive Transformer
        # ----------------------
        self.n_future_frames = n_future_frames
        self.autoregressive_transformer = AutoregressiveTokenTransformer(
            d_model=2 * self.dec_embed_dim,  # Concatenated features from decode
            n_heads=ar_n_heads,
            n_layers=ar_n_layers,
            d_ff=3 * self.dec_embed_dim,
            dropout=ar_dropout,
            n_future_frames=n_future_frames,
            max_seq_len=10  # Can handle up to 10 frames total
        )

        # ----------------------
        #  Task-specific Decoders
        # ----------------------
        # Point decoder
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(
            patch_size=self.patch_size, 
            dec_embed_dim=1024, 
            output_dim=3
        )

        # Confidence decoder
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(
            patch_size=self.patch_size, 
            dec_embed_dim=1024, 
            output_dim=1
        )

        # Camera pose decoder
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # Segmentation decoder and head (optional)
        if self.use_segmentation_head:
            self.segmentation_decoder = deepcopy(self.point_decoder)
            self.segmentation_head = LinearPts3d(
                patch_size=self.patch_size, 
                dec_embed_dim=1024, 
                output_dim=self.segmentation_num_classes,
            )
        
        # Motion decoder and head (optional)
        if self.use_motion_head:
            self.motion_decoder = deepcopy(self.point_decoder)
            self.motion_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=1,  # Binary motion mask (0=static, 1=moving)
            )

        # Flow decoder and head (optional)
        if self.use_flow_head:
            self.flow_decoder = deepcopy(self.point_decoder)
            self.flow_head = LinearPts3d(
                patch_size=self.patch_size,
                dec_embed_dim=1024,
                output_dim=2,  # 2D optical flow (dx, dy)
            )

        # Image normalization
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)

        # Freeze decoders if requested
        if freeze_decoders:
            self._freeze_decoders()

    def _unfreeze_decoders(self):
        """Unfreeze main decoder and task-specific decoders and heads to force better token representation learning.
        NOTE: Segmentation decoder/head are NOT frozen even when freeze_decoders=True, 
        as they need to learn from scratch."""
        # Unfreeze main decoder
        # for param in self.decoder.parameters():
        #     param.requires_grad = False
        
        # unFreeze task decoders
        for param in self.point_decoder.parameters():
            param.requires_grad = True
        for param in self.conf_decoder.parameters():
            param.requires_grad = True
        for param in self.camera_decoder.parameters():
            param.requires_grad = True
        
        # unFreeze heads
        for param in self.point_head.parameters():
            param.requires_grad = True
        for param in self.conf_head.parameters():
            param.requires_grad = True
        for param in self.camera_head.parameters():
            param.requires_grad = True

    def _freeze_decoders(self):
        """Freeze main decoder and task-specific decoders and heads to force better token representation learning.
        NOTE: Segmentation decoder/head are NOT frozen even when freeze_decoders=True, 
        as they need to learn from scratch."""
        # Freeze main decoder
        # for param in self.decoder.parameters():
        #     param.requires_grad = False
        
        # Freeze task decoders
        for param in self.point_decoder.parameters():
            param.requires_grad = False
        for param in self.conf_decoder.parameters():
            param.requires_grad = False
        for param in self.camera_decoder.parameters():
            param.requires_grad = False
        
        # Freeze heads
        for param in self.point_head.parameters():
            param.requires_grad = False
        for param in self.conf_head.parameters():
            param.requires_grad = False
        for param in self.camera_head.parameters():
            param.requires_grad = False
        
        # NOTE: Segmentation and motion decoders/heads are intentionally NOT frozen
        # They need to train from scratch even when other decoders are frozen
        
        main_decoder_params = sum(1 for p in self.decoder.parameters())
        print(f"Froze main decoder ({main_decoder_params} parameters), point, conf, and camera decoders and heads")
        if self.use_segmentation_head:
            print("  ℹ️  Segmentation decoder/head remain trainable (not frozen)")
        if self.use_motion_head:
            print("  ℹ️  Motion decoder/head remain trainable (not frozen)")
    
    def verify_gradient_flow(self):
        """Verify which parts of the model have gradients enabled."""
        components = {
            'encoder': self.encoder,
            'decoder': self.decoder,
            'autoregressive_transformer': self.autoregressive_transformer,
            'point_decoder': self.point_decoder,
            'point_head': self.point_head,
            'conf_decoder': self.conf_decoder,
            'conf_head': self.conf_head,
            'camera_decoder': self.camera_decoder,
            'camera_head': self.camera_head
        }
        
        if self.use_segmentation_head:
            components['segmentation_decoder'] = self.segmentation_decoder
            components['segmentation_head'] = self.segmentation_head
        
        if self.use_motion_head:
            components['motion_decoder'] = self.motion_decoder
            components['motion_head'] = self.motion_head
        
        if self.use_flow_head:
            components['flow_decoder'] = self.flow_decoder
            components['flow_head'] = self.flow_head
        
        for name, module in components.items():
            params_with_grad = sum(p.requires_grad for p in module.parameters())
            total_params = sum(1 for p in module.parameters())
            status = "✓" if params_with_grad > 0 else "✗"
            print(f"{status} {name}: {params_with_grad}/{total_params} params with gradients")

    def decode(self, hidden, N, H, W):
        """Same as original Pi3 decode function"""
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        
        hidden = hidden.reshape(B*N, hw, -1)

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # Encode images
        imgs = imgs.reshape(B*N, _, H, W)
        if hasattr(self.encoder, 'forward_features'):
            hidden = self.encoder.forward_features(imgs)
        else:
            hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        dino_features = hidden
        # Decode and aggregate spatial-temporal features
        hidden, pos = self.decode(hidden, N, H, W)
        pi3_features = hidden

        # Generate future tokens autoregressively
        all_hidden, all_pos = self.autoregressive_transformer(hidden, N, pos)
        autonomy_features = all_hidden  # [B*(N+M), S, D] - all features including future frames from AR transformer

        # Udoate frame count to include future frames
        total_frames = N + self.n_future_frames
        
        # Process all tokens (current + future) through task decoders
        point_hidden = self.point_decoder(all_hidden, xpos=all_pos)
        conf_hidden = self.conf_decoder(all_hidden, xpos=all_pos)
        camera_hidden = self.camera_decoder(all_hidden, xpos=all_pos)

        # lets get point, conf, cam features for all frames (current + future)
        point_features = point_hidden
        conf_features = conf_hidden
        camera_features = camera_hidden
        
        if self.use_segmentation_head:
            segmentation_hidden = self.segmentation_decoder(all_hidden, xpos=all_pos)
        
        if self.use_motion_head:
            motion_hidden = self.motion_decoder(all_hidden, xpos=all_pos)

        if self.use_flow_head:
            flow_hidden = self.flow_decoder(all_hidden, xpos=all_pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # Generate predictions for all frames
            # Points - [B*total_frames, H, W, 3]
            point_hidden = point_hidden.float()
            local_points_flat = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W))
            local_points_raw = local_points_flat.reshape(B, total_frames, H, W, -1)
            
            xy, z = local_points_raw.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # Confidence - [B*total_frames, H, W, 1]
            conf_hidden = conf_hidden.float()
            conf_flat = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W))
            conf = conf_flat.reshape(B, total_frames, H, W, -1)

            # Camera poses - [B*total_frames, 4, 4]
            camera_hidden = camera_hidden.float()
            camera_poses_flat = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w)
            camera_poses = camera_poses_flat.reshape(B, total_frames, 4, 4)

            # Segmentation - [B*total_frames, H, W, 6]
            if self.use_segmentation_head:
                segmentation_hidden = segmentation_hidden.float()
                segmentation_flat = self.segmentation_head([segmentation_hidden[:, self.patch_start_idx:]], (H, W))
                segmentation = segmentation_flat.reshape(B, total_frames, H, W, -1)
            else:
                segmentation = None
            
            # Motion - [B*total_frames, H, W, 1]
            if self.use_motion_head:
                motion_hidden = motion_hidden.float()
                motion_flat = self.motion_head([motion_hidden[:, self.patch_start_idx:]], (H, W))
                motion = motion_flat.reshape(B, total_frames, H, W, -1)  # Binary motion masks
            else:
                motion = None

            # Flow - [B*total_frames, H, W, 2]
            if self.use_flow_head:
                flow_hidden = flow_hidden.float()
                flow_flat = self.flow_head([flow_hidden[:, self.patch_start_idx:]], (H, W))
                flow = flow_flat.reshape(B, total_frames, H, W, -1)  # 2D optical flow (dx, dy)
            else:
                flow = None

            # Unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        result = dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            n_current_frames=N,
            n_future_frames=self.n_future_frames,
            dino_features=dino_features,  # [B*N, S, D] - DINOv2 encoder features for potential supervision
            pi3_features=pi3_features,    # [B*N, S, D]
            autonomy_features=autonomy_features,  # [B*(N+M), S, D] - all features including future frames from AR transformer,
            point_features=point_features,  # [B*(N+M), S, D]
            conf_features=conf_features,    # [B*(N+M), S, D
            camera_features=camera_features   # [B*(N+M), S, D]
        )
        
        if self.use_segmentation_head:
            result['segmentation'] = segmentation
        
        if self.use_motion_head:
            result['motion'] = motion
        
        if self.use_flow_head:
            result['flow'] = flow
        
        # Always include decoder features for potential supervision
        result['all_decoder_features'] = all_hidden  # Current + future decoder features [B*(N+M), S, D]
        result['all_positional_encoding'] = all_pos  # [B*(N+M), S, pos_dim]
            
        return result
