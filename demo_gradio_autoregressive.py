import os
import cv2
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi3.utils.geometry import se3_inverse, homogenize_points, depth_edge
from pi3.models.pi3 import Pi3
from pi3.utils.basic import load_images_as_tensor
from minimal_pi3_inference import load_autoregressive_pi3, run_pi3_inference, preprocess_images

import trimesh
import matplotlib
from scipy.spatial.transform import Rotation

# Import camera integration functions from original demo_gradio
from demo_gradio import integrate_camera_into_scene, get_opengl_conversion_matrix, transform_points, compute_camera_faces


"""
Gradio utils - Modified for Autoregressive Pi3
"""

def predictions_to_glb_autoregressive(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    show_cam=True,
    show_future=True,
    point_size=0.005,
) -> trimesh.Scene:
    """
    Converts Pi3 predictions to a 3D scene with support for autoregressive models.
    Shows current frames with actual depths and future frames with predicted depths but future colors.
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10

    print("Building GLB scene")
    
    # Get world points and confidence
    pred_world_points = predictions["points"]
    pred_world_points_conf = predictions.get("conf", np.ones_like(pred_world_points[..., 0]))
    
    # Handle confidence shape
    if pred_world_points_conf.ndim == 4 and pred_world_points_conf.shape[-1] == 1:
        pred_world_points_conf = pred_world_points_conf.squeeze(-1)

    # Get images from predictions
    images = predictions["images"]
    camera_poses = predictions["camera_poses"]
    
    # For autoregressive model:
    # - First 3 frames are input frames
    # - Last 3 frames are predicted frames
    num_frames = len(images)
    
    # Handle frame filtering
    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Support both single index and comma-separated indices
            if "," in filter_by_frames:
                selected_frame_idx = [int(x.strip()) - 1 for x in filter_by_frames.split(",") if x.strip().isdigit()]
            else:
                selected_frame_idx = [int(filter_by_frames.split(":")[0])]
        except (ValueError, IndexError):
            pass
    
    # If specific frames selected, filter all data
    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx]
        images = images[selected_frame_idx]
        camera_poses = camera_poses[selected_frame_idx]
    
    # Flatten points and colors for point cloud
    vertices_3d = pred_world_points.reshape(-1, 3)
    colors_rgb = images.reshape(-1, 3)
    
    # Ensure colors are in correct format
    if colors_rgb.max() <= 1.0:
        colors_rgb = (colors_rgb * 255).astype(np.uint8)
    
    # Apply confidence filtering
    conf = pred_world_points_conf.reshape(-1)
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = conf_thres / 100
    
    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]
    
    # Calculate scene scale
    if vertices_3d.size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)
        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)
    
    # Create colormap for cameras
    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
    
    # Initialize 3D scene
    scene_3d = trimesh.Scene()
    
    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    scene_3d.add_geometry(point_cloud_data)
    
    # Add camera models if requested
    if show_cam:
        num_cameras = len(camera_poses)
        
        for i in range(num_cameras):
            camera_to_world = camera_poses[i]
            
            # Color cameras based on whether they're input or predicted frames
            if selected_frame_idx is not None:
                # Use original frame indices for coloring
                original_idx = selected_frame_idx[i] if i < len(selected_frame_idx) else i
            else:
                original_idx = i
            
            # For autoregressive model with 6 frames:
            # Blue for input frames (0-2), Red for predicted frames (3-5)
            if num_frames == 6 and original_idx >= 3:
                current_color = (255, 0, 0)  # Red for predicted frames
            else:
                current_color = (0, 0, 255)  # Blue for input frames
            
            # Use the original integrate_camera_into_scene function
            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, 1.0)
    
    # Apply the same rotation as original for better visualization
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 100, degrees=True).as_matrix()
    align_rotation[:3, :3] = align_rotation[:3, :3] @ Rotation.from_euler("x", 155, degrees=True).as_matrix()
    scene_3d.apply_transform(align_rotation)
    
    print("GLB Scene built")
    return scene_3d


# -------------------------------------------------------------------------
# Core model inference - Modified for autoregressive
# -------------------------------------------------------------------------
def run_model_autoregressive(target_dir, model, use_autoregressive=True) -> dict:
    print(f"Processing images from {target_dir}")

    # Device check
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    # Move model to device
    model = model.to(device)
    model.eval()

    # Load and preprocess images
    image_names = glob.glob(os.path.join(target_dir, "images", "*"))
    image_names = sorted(image_names)
    print(f"Found {len(image_names)} images")
    if len(image_names) == 0:
        raise ValueError("No images found. Check your upload.")

    # For autoregressive model, we need exactly 6 images (3 input + 3 to predict)
    if use_autoregressive:
        if len(image_names) < 6:
            print(f"Warning: Autoregressive model expects 6 images, got {len(image_names)}. Using first 3 for input.")
            image_names = image_names[:3]
        else:
            image_names = image_names[:6]

    # Load images using the preprocess_images function
    imgs_tensor = preprocess_images(image_names).to(device)  # (N, C, H, W)
    
    # For autoregressive model, only use first 3 frames as input
    if use_autoregressive:
        input_imgs = imgs_tensor[:3]
    else:
        input_imgs = imgs_tensor

    # Run inference
    print("Running model inference...")
    dtype = torch.bfloat16
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=dtype):
            predictions = run_pi3_inference(model, input_imgs, is_autoregressive=use_autoregressive)
    
    # Store original images in predictions
    predictions['images'] = imgs_tensor.permute(0, 2, 3, 1).cpu().numpy()  # Convert to [N, H, W, C]
    
    # Handle confidence
    if 'conf' in predictions:
        predictions['conf'] = torch.sigmoid(predictions['conf'])
        
        # Edge detection on local points if available
        if 'local_points' in predictions:
            edge = depth_edge(predictions['local_points'][..., 2], rtol=0.03)
            predictions['conf'][edge] = 0.0
            # Remove local_points after edge detection
            del predictions['local_points']

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor) and key not in ['conf_features', 'point_features', 'camera_features', 'features', 'pos', 'dino_features', 'pi3_features']:
            # Handle batch dimension properly
            if predictions[key].dim() > 0 and predictions[key].shape[0] == 1:
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            else:
                predictions[key] = predictions[key].cpu().numpy()

    # Remove unnecessary keys
    for key in ['conf_features', 'point_features', 'camera_features', 'features', 'pos', 'dino_features', 'pi3_features', 'local_points']:
        if key in predictions:
            del predictions[key]

    # Clean up
    torch.cuda.empty_cache()
    gc.collect()

    return predictions


# -------------------------------------------------------------------------
# Main interface
# -------------------------------------------------------------------------
def process_upload(video_file, image_files, interval):
    """Process uploaded video or images."""
    # Create a temporary directory for processing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target_dir = f"temp_upload_{timestamp}"
    images_dir = os.path.join(target_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    uploaded_files = []

    try:
        if video_file is not None:
            # Extract frames from video
            video_path = video_file
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_count = 0
            saved_count = 0
            
            while cap.isOpened() and saved_count < 6:  # Limit to 6 frames for autoregressive
                ret, frame = cap.read()
                if not ret:
                    break
                    
                if frame_count % interval == 0:
                    frame_path = os.path.join(images_dir, f"frame_{saved_count:06d}.png")
                    cv2.imwrite(frame_path, frame)
                    uploaded_files.append(frame_path)
                    saved_count += 1
                    
                frame_count += 1
                
            cap.release()
            
        elif image_files is not None:
            # Copy uploaded images
            for i, img_file in enumerate(image_files[:6]):  # Limit to 6 images
                img_path = os.path.join(images_dir, f"image_{i:06d}.png")
                shutil.copy(img_file, img_path)
                uploaded_files.append(img_path)
    
        return target_dir, uploaded_files
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        raise e


def count_parameters(model):
    """Count the number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reconstruct_autoregressive(target_dir, model_type="autoregressive"):
    """Run reconstruction with selected model type."""
    if target_dir is None:
        return None, "Please upload images or video first."
    
    # Select model based on type
    if model_type == "autoregressive":
        print("Running autoregressive Pi3 model...")
        model, _ = load_autoregressive_pi3()
        num_params = count_parameters(model)
        print(f"Model has {num_params:,} trainable parameters ({num_params/1e6:.1f}M)")
        predictions = run_model_autoregressive(target_dir, model, use_autoregressive=True)
    else:
        print("Running standard Pi3 model...")
        model = Pi3.from_pretrained("yyfz233/Pi3")
        num_params = count_parameters(model)
        print(f"Model has {num_params:,} trainable parameters ({num_params/1e6:.1f}M)")
        predictions = run_model_autoregressive(target_dir, model, use_autoregressive=False)
    

    # Save predictions
    prediction_save_path = os.path.join(target_dir, "predictions.npz")
    np.savez(prediction_save_path, **predictions)
    
    # Generate 3D visualization
    glb_file = generate_3d(target_dir, 50.0, "all", True)
    
    return glb_file, "Reconstruction completed successfully!"
   

def generate_3d(target_dir, conf_thres, frame_filter, show_cam):
    """Generate 3D visualization from predictions."""
    predictions_path = os.path.join(target_dir, "predictions.npz")
    if not os.path.exists(predictions_path):
        return None
    
    # Load predictions
    loaded = np.load(predictions_path)
    predictions = {key: np.array(loaded[key]) for key in loaded.files}
    
    # Create 3D scene
    scene = predictions_to_glb_autoregressive(
        predictions,
        conf_thres=conf_thres,
        filter_by_frames=frame_filter,
        show_cam=show_cam,
        show_future=True
    )
    
    # Save as GLB
    glb_path = os.path.join(target_dir, "output.glb")
    scene.export(glb_path)
    
    return glb_path


# -------------------------------------------------------------------------
# Gradio Interface
# -------------------------------------------------------------------------
def create_interface():
    with gr.Blocks(title="Pi3 Autoregressive 3D Reconstruction") as demo:
        gr.Markdown(
            """
            # Pi3 Autoregressive 3D Reconstruction
            
            This demo shows the autoregressive Pi3 model that takes 3 input frames and predicts 3 future frames.
            - **Blue cameras**: Input frames (1-3)
            - **Red cameras**: Predicted future frames (4-6)
            - Point clouds use actual depths for all frames, but future frames show predicted RGB colors
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                with gr.Tab("Upload"):
                    video_input = gr.File(label="Upload Video", type="filepath", elem_id="video_upload")
                    image_input = gr.File(
                        label="Or Upload Images (6 images recommended)",
                        file_count="multiple",
                        type="filepath",
                        elem_id="image_upload"
                    )
                    interval_slider = gr.Slider(
                        minimum=1,
                        maximum=30,
                        value=10,
                        step=1,
                        label="Frame Interval (for video)",
                        info="Extract every Nth frame from video"
                    )
                    
                    upload_btn = gr.Button("Process Upload", variant="primary")
                
                # Model selection
                model_type = gr.Radio(
                    choices=["autoregressive", "standard"],
                    value="autoregressive",
                    label="Model Type",
                    info="Choose between autoregressive (3→6 frames) or standard Pi3"
                )
                
                # Reconstruct button
                reconstruct_btn = gr.Button("🚀 Reconstruct", variant="primary", size="lg")
                
                # Visualization controls
                with gr.Accordion("Visualization Settings", open=False):
                    conf_slider = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=50,
                        label="Point Confidence Threshold %",
                        info="Filter out low-confidence points"
                    )
                    frame_filter = gr.Textbox(
                        value="all",
                        label="Frame Filter",
                        info="Enter frame numbers (e.g., '1,2,3') or 'all'"
                    )
                    show_cameras = gr.Checkbox(value=True, label="Show Cameras")
                    
                    update_btn = gr.Button("Update Visualization")
                
                # Status
                status_text = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column(scale=2):
                # Preview gallery
                preview_gallery = gr.Gallery(
                    label="Input Frames Preview",
                    show_label=True,
                    elem_id="preview_gallery",
                    columns=3,
                    rows=2,
                    height="auto"
                )
                
                # 3D viewer
                model_viewer = gr.Model3D(
                    label="3D Reconstruction",
                    height=600,
                    elem_id="model_viewer"
                )
                
                # Download button
                download_btn = gr.Button("💾 Download GLB", variant="secondary")
        
        # Hidden state to store target directory
        target_dir_state = gr.State(value=None)
        
        # Event handlers
        def upload_handler(video, images, interval):
            try:
                target_dir, preview_images = process_upload(video, images, interval)
                return target_dir, preview_images, "Upload successful!"
            except Exception as e:
                return None, None, f"Upload error: {str(e)}"
        
        upload_btn.click(
            upload_handler,
            inputs=[video_input, image_input, interval_slider],
            outputs=[target_dir_state, preview_gallery, status_text]
        )
        
        def reconstruct_handler(target_dir, model_type):
            return reconstruct_autoregressive(target_dir, model_type)
        
        reconstruct_btn.click(
            reconstruct_handler,
            inputs=[target_dir_state, model_type],
            outputs=[model_viewer, status_text]
        )
        
        def update_visualization(target_dir, conf_thres, frame_filter, show_cam):
            if target_dir is None:
                return None, "No reconstruction available."
            try:
                glb_file = generate_3d(target_dir, conf_thres, frame_filter, show_cam)
                return glb_file, "Visualization updated!"
            except Exception as e:
                return None, f"Update error: {str(e)}"
        
        update_btn.click(
            update_visualization,
            inputs=[target_dir_state, conf_slider, frame_filter, show_cameras],
            outputs=[model_viewer, status_text]
        )
        
    return demo


if __name__ == "__main__":
    # Initialize model on startup
    print("Starting Gradio interface...")
    
    # Create and launch interface
    demo = create_interface()
    demo.launch(share=True)