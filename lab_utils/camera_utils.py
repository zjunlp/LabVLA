# ============================================================================
# LabUtopia camera image utilities.
# Process simulator RGB, depth, point cloud, and segmentation data.
# Supported uses:
# - Convert images to a common format for data collection.
# - Collect and display multiple image modalities.
# - Visualize instance segmentation with color coding.
# ============================================================================

import cv2  # OpenCV image processing and visualization.
import numpy as np  # NumPy array operations.


def string_to_color(category, num_colors=256):
    """Map a category string to an RGB color using Python's hash and the prime factors 37, 59, and 73. num_colors defaults to 256, yielding channel values from 0 to 255. Equal category strings produce equal colors within a process; Python hash randomization can change colors between processes."""
    hash_value = hash(category) % num_colors  # Hash the category and reduce modulo the color count.
    # Use different prime factors for the RGB channels.
    color = [(hash_value * 37) % 256,   # Red channel.
             (hash_value * 59) % 256,   # Green channel.
             (hash_value * 73) % 256]   # Blue channel.
    return tuple(color)  # Return the color tuple.


def create_color_map(id_to_labels):
    """Map instance IDs to RGB colors based on their labels. Accept a mapping such as {"0": "background", "1": "beaker"} and return ID-to-color tuples for segmentation visualization."""
    color_map = {}  # Initialize the color mapping.
    for id_str, label in id_to_labels.items():  # Iterate over ID-label pairs.
        # Equal labels receive the same color.
        color_map[id_str] = string_to_color(label)
    return color_map  # Return the color mapping.


def process_single_type(camera, image_type):
    """Retrieve and format one camera modality, returning (record_data, display_data), or (None, None) when unavailable. Supported image_type values are rgb, depth, pointcloud, segmentation, and point (RGB with object centers). RGB uses [C, H, W]; depth and segmentation records use [1, H, W] uint8; point clouds use [N, 3] float32 with RGB previews."""
    # ==================== RGB images ====================
    if image_type == "rgb":
        rgb_img = camera.get_rgb()  # Get an RGB image from the camera as [H, W, C].
        if rgb_img is None:
            return None, None
        # Ensure the image has three axes [H, W, C].
        if rgb_img.ndim == 2:
            # Expand grayscale images to three channels.
            rgb_img = np.stack([rgb_img] * 3, axis=-1)
        elif rgb_img.ndim == 3 and rgb_img.shape[2] == 4:
            # Drop the alpha channel from RGBA images.
            rgb_img = rgb_img[:, :, :3]
        elif rgb_img.ndim != 3:
            print(f"Warning: Unexpected RGB image shape: {rgb_img.shape}")
            return None, None
        # Transpose to [C, H, W] for PyTorch-style processing.
        img_for_record = np.transpose(rgb_img, (2, 0, 1))
        return img_for_record, img_for_record  # Use the same data for recording and display.
    
    # ==================== Point clouds ====================
    elif image_type == "pointcloud":
        rgb_img = camera.get_rgb()  # Get RGB data for display.
        if rgb_img is None:
            return None, None
        if rgb_img.ndim == 3 and rgb_img.shape[2] == 4:
            rgb_img = rgb_img[:, :, :3]
        img_for_display = np.transpose(rgb_img, (2, 0, 1))  # Convert the display layout.
        pointcloud = camera.get_pointcloud()  # Get the 3D point cloud.
        if pointcloud is not None:
            # Cast the point cloud to float32 to reduce storage.
            pointcloud_for_record = pointcloud.astype(np.float32)
            return pointcloud_for_record, img_for_display
        else:
            print("Warning: Point cloud data not available.")  # Print a warning.
            return None, None
    
    # ==================== Depth images ====================
    elif image_type == "depth":
        depth = camera.get_depth()  # Get the single-channel floating-point depth image.
        if depth is not None:
            # Normalize depth to the 0-255 range for recording and display.
            depth_normalized = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
            depth_for_record = depth_normalized.astype(np.uint8)  # Cast to an 8-bit integer array.
            # Add a channel axis: [H, W] -> [H, W, 1].
            depth_for_record = depth_for_record[:, :, np.newaxis]
            # Transpose to [C, H, W].
            depth_for_record = np.transpose(depth_for_record, (2, 0, 1))  # C * W * H
            # Apply the JET colormap for the depth preview.
            depth_for_display = cv2.applyColorMap(depth_for_record[0], cv2.COLORMAP_JET)
            return depth_for_record, depth_for_display
        else:
            print("Warning: Depth data not available.")  # Print a warning.
            return None, None
    
    # ==================== Instance segmentation ====================
    elif image_type == "segmentation":
        frame_dict = camera.get_current_frame()  # Get all data for the current frame.
        instance_id_seg = frame_dict.get('instance_segmentation')  # Get instance segmentation data.
        if instance_id_seg is not None:
            seg = instance_id_seg['data']  # Each segmentation pixel contains an instance ID.
            if seg is not None:
                seg_for_record = seg.astype(np.uint8)  # Cast to an 8-bit integer array.
                # Get instance labels for color generation.
                id_to_labels = instance_id_seg.get('info', {}).get('idToLabels', {})
                color_map = create_color_map(id_to_labels)  # Create the color mapping.
                
                # Allocate a color segmentation preview [H, W, 3].
                seg_for_display = np.zeros((seg_for_record.shape[0], seg_for_record.shape[1], 3), dtype=np.uint8)
                
                # Fill each instance with its assigned color.
                for id_value in np.unique(seg_for_record):  # Iterate over unique instance IDs.
                    if str(id_value) in color_map:
                        # Assign the instance color to all matching pixels.
                        seg_for_display[seg_for_record == id_value] = color_map[str(id_value)]
                
                # Add the channel axis and transpose.
                seg_for_record = seg_for_record[:, :, np.newaxis]
                seg_for_record = np.transpose(seg_for_record, (2, 0, 1))  # [1, H, W]
                seg_for_display = np.transpose(seg_for_display, (2, 0, 1))  # [3, H, W]
                
                return seg_for_record, seg_for_display
        else:
            print("Warning: Segmentation data not available.")  # Print a warning.
            return None, None
    
    # ==================== Object center overlays ====================
    elif image_type == "point":
        """Overlay object centers on an RGB image for debugging and visualization."""
        frame_dict = camera.get_current_frame()  # Get current frame data.
        rgb_img = camera.get_rgb()  # Get the RGB image.
        img_for_display = rgb_img[..., ::-1].copy()  # Convert RGB to OpenCV BGR and copy before editing.
        
        instance_id_seg = frame_dict.get('instance_segmentation')  # Get segmentation data.
        if instance_id_seg is not None:
            seg = instance_id_seg['data']
            if seg is not None:
                seg_for_process = seg.astype(np.uint8)
                id_to_labels = instance_id_seg.get('info', {}).get('idToLabels', {})
                
                # Draw a dot at each instance's center.
                for id_value in np.unique(seg_for_process):
                    # Skip ID 0 (usually background) and ID 1 (usually the ground or table).
                    if str(id_value) in id_to_labels and id_value != 0 and id_value != 1:
                        mask = (seg_for_process == id_value)  # Get the instance mask.
                        y, x = np.where(mask)  # Find the instance's pixel coordinates.
                        if len(y) > 0 and len(x) > 0:
                            # Compute the center as the mean pixel coordinate.
                            center_y, center_x = int(np.mean(y)), int(np.mean(x))
                            # Draw a filled blue circle with radius 6 at the center.
                            cv2.circle(img_for_display, (center_x, center_y), 6, (255, 100, 100), -1)
                
                # Convert the preview back to RGB.
                img_for_display_rgb = img_for_display[..., ::-1]  # BGR to RGB.
                # Transpose to [C, H, W].
                img_for_record = np.transpose(img_for_display_rgb, (2, 0, 1))
                return img_for_record, img_for_display
            else:
                print("Warning: Segmentation data not available.")
                return None, None
        else:
            print("Warning: Instance segmentation not available.")
            return None, None
    
    # ==================== Unsupported image types ====================
    else:
        return None, None  # Return None for unsupported modalities.


def process_camera_image(camera, image_type):
    """Process a single camera modality or a '+'-separated combination, such as 'rgb+pointcloud'. Return (record_data, display_data). A combined request produces a modality-to-array dictionary plus an RGB preview; a single request produces one record array and its preview. For example, process_camera_image(camera, 'rgb+pointcloud') returns RGB [C, H, W] and point cloud [N, 3] arrays."""
    # Always retrieve RGB data for a preview.
    # This provides a display image even for modalities such as point clouds.
    record, display = process_single_type(camera, "rgb")
    if record is None:
        return None, None
    
    if '+' in image_type:
        # ========== Combined modalities ==========
        types = image_type.split('+')  # Split the modality string.
        record_dict = {}  # Initialize the record dictionary.
        
        for t in types:  # Iterate over requested modalities.
            record, _ = process_single_type(camera, t)  # Retrieve this modality's data.
            if record is not None:
                record_dict[t] = record  # Add the data to the dictionary.
        
        return record_dict, display  # Return the record dictionary and RGB preview.
    else:
        # ========== Single modality ==========
        return process_single_type(camera, image_type)  # Delegate to the single-modality processor.
