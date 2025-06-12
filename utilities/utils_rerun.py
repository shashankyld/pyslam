# Functions to log different components of the SLAM system to rerun:

import rerun as rr
import numpy as np
from config_parameters import Parameters  

point_size = Parameters.kPointVisualizationRadius_Rerun

def log_coordinate_axes(entity_path, pose=None, scale=1.0):
    """
    Log 3D coordinate axes using a pose matrix.
    
    Args:
        entity_path: Rerun entity path for the axes
        pose: 4x4 transformation matrix (rotation + translation) (default: identity matrix)
        scale: Size of the axes (default: 1.0)
    """
    # Default to identity matrix if pose is None
    if pose is None:
        pose = np.eye(4)
    
    # Extract rotation matrix and translation vector from pose
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    
    # Define unit vectors for each axis and transform them
    x_axis = rotation @ np.array([scale, 0, 0])
    y_axis = rotation @ np.array([0, scale, 0])
    z_axis = rotation @ np.array([0, 0, scale])
    
    # Create line segments from origin to each axis end point
    x_line = np.stack([translation, translation + x_axis])
    y_line = np.stack([translation, translation + y_axis])
    z_line = np.stack([translation, translation + z_axis])
    
    # Log each axis with appropriate color
    rr.log(f"{entity_path}/x_axis", rr.LineStrips3D(x_line, colors=[255, 0, 0, 255]))  # Red for X
    rr.log(f"{entity_path}/y_axis", rr.LineStrips3D(y_line, colors=[0, 255, 0, 255]))  # Green for Y
    rr.log(f"{entity_path}/z_axis", rr.LineStrips3D(z_line, colors=[0, 0, 255, 255]))  # Blue for Z

def log_camera(entity_path, world_T_cam_44, K_44):
    """Logs camera intrinsics and extrinsics to rerun."""

    assert world_T_cam_44.shape == (4, 4)
    assert K_44.shape == (4, 4)

    # Convert and log camera parameters
    Rot, trans = world_T_cam_44[:3, :3], world_T_cam_44[:3, 3]
    K_33 = K_44[:3, :3]

    rr.log(
        entity_path,
        rr.Pinhole(
            image_from_camera=K_33,
            width=K_33[0, 2] * 2,  # Assuming principal point is in the center
            height=K_33[1, 2] * 2,
        ),
    )

    rr.log(entity_path, rr.Transform3D(translation=trans, mat3x3=Rot))

import cv2
def ensure_rgb(image):
    """
    Convert image to RGB format if it's a BGR image.
    
    Args:
        image: An image array
    
    Returns:
        The image in RGB format
    """
    if image is None:
        return None
        
    # Check if image has 3 channels (color image)
    if isinstance(image, np.ndarray) and len(image.shape) == 3 and image.shape[2] == 3:
        # Convert from BGR to RGB
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def log_image(entity, image, ensure_rgb_flag=False):
    """Logs an image to rerun."""
    if image is None:
        print(f"Warning: Attempted to log None image to {entity}")
        return
    # Convert image to RGB if it's in BGR format
    if ensure_rgb_flag:
        image = ensure_rgb(image)
    
    
    # Check for empty or invalid image
    if not isinstance(image, np.ndarray) or image.size == 0 or len(image.shape) < 2:
        print(f"Warning: Invalid image shape {getattr(image, 'shape', 'unknown')} for {entity}")
        # Create a small placeholder image instead
        placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Image", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rr.log(f"{entity}", rr.Image(ensure_rgb(placeholder)))
        return
    
    try:
        # Convert to RGB before logging
        rgb_image = ensure_rgb(image)
        rr.log(f"{entity}", rr.Image(rgb_image))
    except Exception as e:
        print(f"Error logging image to {entity}: {e}")
        # Create a small error image instead
        error_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        rr.log(f"{entity}", rr.Image(ensure_rgb(error_img)))

def log_image2(entity, image, ensure_rgb_flag=False):
    """Logs an image to rerun."""
    if image is None:
        print(f"Warning: Attempted to log None image to {entity}")
        return
    # Convert image to RGB if it's in BGR format
    if not ensure_rgb_flag:
        image = ensure_rgb(image)
    
    
    # Check for empty or invalid image
    if not isinstance(image, np.ndarray) or image.size == 0 or len(image.shape) < 2:
        print(f"Warning: Invalid image shape {getattr(image, 'shape', 'unknown')} for {entity}")
        # Create a small placeholder image instead
        placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(placeholder, "No Image", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        rr.log(f"{entity}", rr.Image(ensure_rgb(placeholder)))
        return
    
    try:
        # Convert to RGB before logging
        rgb_image = ensure_rgb(image)
        rr.log(f"{entity}", rr.Image(rgb_image))
    except Exception as e:
        print(f"Error logging image to {entity}: {e}")
        # Create a small error image instead
        error_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.putText(error_img, "Error", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        rr.log(f"{entity}", rr.Image(ensure_rgb(error_img)))
        
def log_local_map(entity_path, points, colors=None):  # Added 'points' parameter
    """Logs the local map to rerun."""
    # points = get_local_map(frame_id)  # Removed the call to get_local_map
    if colors is not None:
        rr.log(
            f"{entity_path}/local_map",
            rr.Points3D(points, colors=colors),
        )
    else:
        # colors = Green
        colors = np.array([0, 255, 0], dtype=np.uint8)
        rr.log(f"{entity_path}/local_map", rr.Points3D(points, colors=colors))

def log_snapshot_map(index, entity_path, points, colors=None):  # Added 'points' parameter
    """Logs the local map to rerun."""
    # points = get_local_map(frame_id)  # Removed the call to get_local_map
    if colors is not None:
        rr.log(
            f"{entity_path}/snapshot_map_{index}",
            rr.Points3D(points, colors=colors),
        )
    else:
        # colors = Green
        colors = np.array([0, 255, 0], dtype=np.uint8)
        rr.log(f"{entity_path}/snapshot_map_{index}", rr.Points3D(points, colors=colors))

def log_matching_pointclouds(entity, points1, points2):  # Added 'points' parameter
    """Logs the matching point clouds to rerun."""
    # Convert to numpy arrays if they are not already
    if not isinstance(points1, np.ndarray):
        points1 = np.array(points1)
    if not isinstance(points2, np.ndarray):
        points2 = np.array(points2)
    print("shape of points1", points1.shape)
    print("shape of points2", points2.shape)
    
    ## Add a line between points1 and points2 with idx guiding which two points to connect, p1[0], p2[0] have a line, p1[1], p2[1] have a line and so on
    if points1 is None or points2 is None:
        print(f"Warning: Attempted to log None point clouds to {entity}")
        return
    if points1.shape[0] != points2.shape[0]:
        print(f"Warning: Point clouds have different sizes {points1.shape[0]} and {points2.shape[0]} for {entity}")
        return
    if points1.shape[0]> 0:
        if points1.shape[1] != 3 or points2.shape[1] != 3:
            print(f"Warning: Point clouds have invalid shape {points1.shape} and {points2.shape} for {entity}")
            return
        # TODO- Lines between matched points
        colors_0 = np.array([0, 0, 255], dtype=np.uint8)
        colors_1 = np.array([255, 0, 0], dtype=np.uint8)
      
        # Log the points to rerun
        rr.log(
            f"{entity}/points1",
            rr.Points3D(points1, colors=colors_0),
        )
        rr.log(
            f"{entity}/points2",
            rr.Points3D(points2, colors=colors_1),
        )


def log_random_pc(entity, points, colors=None):  # Added 'points' parameter
    """Logs the local map to rerun."""
    # points = get_local_map(frame_id)  # Removed the call to get_local_map
    if colors is not None:
        rr.log(
            f"{entity}",
            rr.Points3D(points, colors=colors),
        )
    else:
        # colors = Green
        colors = np.array([0, 255, 0], dtype=np.uint8)
        rr.log(f"{entity}", rr.Points3D(points, colors=colors))

def log_random_pc2(entity, points, colors=None, radius = None):  # Added 'points' parameter
    """Logs the local map to rerun."""
    if colors == "green":
        colors = np.array([0, 255, 0], dtype=np.uint8)
    elif colors == "red":
        colors = np.array([255, 0, 0], dtype=np.uint8)
    elif colors == "blue":
        colors = np.array([0, 0, 255], dtype=np.uint8)
    elif colors == "yellow":
        colors = np.array([255, 255, 0], dtype=np.uint8)
    if radius is not None:
        radii = np.ones(points.shape[0]) * radius
    

    rr.log(
        f"{entity}",
        rr.Points3D(points, colors=colors, radii=radii),  # Added radii
    )

import os
    



def log_sam2_folder(entity="sam2", path=None):
    """Logs all the images in the folder to rerun, image with 5 columns and N rows, each element is an image concatenated to one other, N depending on the number of images"""
    if path is None:
        print(f"Warning: Attempted to log None folder path to {entity}")
        return
    
    # Get all image files in the folder
    image_files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    # Sort the image files
    image_files.sort()
    
    # Create a list to hold the images
    images = []
    
    # Read and append each image to the list
    for img_file in image_files:
        img_path = os.path.join(path, img_file)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    
    # Concatenate images into a single image with 5 columns
    if len(images) > 0:
        rows = (len(images) + 4) // 5  # Calculate number of rows needed
        concatenated_image = np.zeros((rows * images[0].shape[0], 5 * images[0].shape[1], 3), dtype=np.uint8)
        
        for i, img in enumerate(images):
            row = i // 5
            col = i % 5
            concatenated_image[row * img.shape[0]:(row + 1) * img.shape[0], col * img.shape[1]:(col + 1) * img.shape[1]] = img
        
        # Add text on each image - its image file path - only the filename and not the full path
        for i, img_file in enumerate(image_files):
            row = i // 5
            col = i % 5
            x = col * images[0].shape[1] + 5
            y = row * images[0].shape[0] + 20
            cv2.putText(concatenated_image, img_file, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        concatenated_image = ensure_rgb(concatenated_image)  # Ensure the image is in RGB format
        rr.log(f"{entity}/images", rr.Image(concatenated_image))
    else:
        print(f"Warning: No images found in folder {path} to log to {entity}")
        return


def log_local_map_snapshot(frame_id, entity_path, points, colors=None, current=None):  # Added 'points' parameter
    """Logs the local map snapshot to rerun."""
    # points = get_local_map_snapshot(frame_id)  # Removed the call to get_local_map_snapshot
    if colors is not None:
        if current:
            rr.log(
                f"{entity_path}/current_local_map_snapshot",
                rr.Points3D(points, colors=colors),
            )
        else:
            rr.log(
                f"{entity_path}/local_map_snapshot",
                rr.Points3D(points, colors=colors),
            )
    else:
        colors = np.array([0, 0, 255], dtype=np.uint8)
        if current:
            rr.log(
                f"{entity_path}/current_local_map_snapshot",
                rr.Points3D(points, colors=colors),
            )
        else:   
            rr.log(f"{entity_path}/local_map_snapshot", rr.Points3D(points, colors=colors))







def log_global_map(entity_path, points,colors=None):  # Added 'points' parameter
    """Logs the global map to rerun."""
    # points = get_global_map(frame_id)  # Removed the call to get_global_map
    if colors is not None:
        rr.log(
            f"{entity_path}/global_map",
            rr.Points3D(points, colors=colors),
        )
    else:
        rr.log(f"{entity_path}/global_map", rr.Points3D(points))

def log_current_frame_map_points(entity_path, points, colors=None):  # Added 'points' parameter
    """Logs the current frame to rerun."""
    # points = get_current_frame_3d(frame_id)  # Removed the call to get_current_frame_3d
    if colors is not None:
        rr.log(
            f"{entity_path}/map_points",
            rr.Points3D(points, colors=colors),
        )
    else:
        rr.log(f"{entity_path}/map_points", rr.Points3D(points))

def log_current_frame_pc(entity_path, points, colors=None, pose = np.eye(4)):  # Added 'points' parameter
    """Logs the current frame to rerun."""
    # Transform the points to the world frame from camera frame
    points = np.dot(pose[:3, :3], points.T).T + pose[:3, 3]
    
    # If points are less than 100, error
    if points.shape[0] < 100:
        print("Error: Points are less than 100")
        exit(1)
    radii = np.ones(points.shape[0]) * point_size  # Set a default radius for points
    # points = get_current_frame_3d(frame_id)  # Removed the call to get_current_frame_3d
    if colors is not None:
        rr.log(
            f"{entity_path}/current_frame_pc",
            rr.Points3D(points, colors=colors, radii=radii),  # Added radii
        )
    else:
        rr.log(f"{entity_path}/current_frame_pc", rr.Points3D(points, colors=(0, 255, 0), radii=radii))  # Default color green with radii

def log_frame_dense_pc(frame_id, entity_path, points, colors=None, pose = np.eye(4), fraction = 0.05):  # Added 'points' parameter
    """Logs the current frame to rerun with frame id."""
    # Transform the points to the world frame from camera frame
    points = np.dot(pose[:3, :3], points.T).T + pose[:3, 3]
    # Downsample points to 5 percent and also colors accordingly
    indexs = np.random.choice(points.shape[0], int(points.shape[0] * fraction), replace=False)
    points = points[indexs]
    if colors is not None:
        colors = colors[indexs]
        # Change colors to rgb
        colors = (colors * 255).astype(np.uint8)
    # Change from bgr to rgb 
    if colors is not None and colors.shape[1] == 3:
        colors = colors[:, [2, 1, 0]]
    elif colors is not None and colors.shape[1] == 1:
        colors = np.repeat(colors, 3, axis=1)
    
    radii = np.ones(points.shape[0]) * point_size  # Set a default radius for points
    # points = get_current_frame_3d(frame_id)  # Removed the call to get_current_frame_3d
    if colors is not None:
        rr.log(
            f"{entity_path}/frame_{frame_id}",
            rr.Points3D(points, colors=colors, radii=radii),  # Added radii
        )
    else:
        rr.log(f"{entity_path}/frame_{frame_id}", rr.Points3D(points, colors=(0, 255, 0), radii=radii))  # Default color green with radii


def log_key_frames(frame_id, entity_path, key_frames):  # Added 'key_frames' parameter
    """Logs the set of key frames to rerun."""
    # key_frames = get_key_frames(frame_id)  # Removed the call to get_key_frames
    for i, frame in enumerate(key_frames):
        # Assuming frame is a dictionary with "image" and "pose"
        image = frame["image"]
        world_T_cam_44 = frame["pose"]  # 4x4 transformation matrix
        K_44 = frame["K"]  # 4x4 intrinsic matrix
        frame_path = f"{entity_path}/key_frames/{i}"
        log_camera(frame_path, world_T_cam_44, K_44)
        log_image(frame_path, image)


def log_current_frame(frame_id, entity_path, image):  # Added 'image' parameter
    """Logs the current frame to rerun."""
    # image = get_current_frame(frame_id)  # Removed the call to get_current_frame
    rr.log(f"{entity_path}/current_frame", rr.Image(image))


def log_features(frame_id, entity_path, features):  # Added 'features' parameter
    """Logs features in the current frame."""
    # features = get_features(frame_id)  # Removed the call to get_features
    rr.log(f"{entity_path}/current_frame/features", rr.Points2D(features))


def log_dynamic_features(
    frame_id, entity_path, dynamic_features
):  # Added 'dynamic_features' parameter
    """Logs dynamic features in the current frame."""
    # dynamic_features = get_dynamic_features(frame_id)  # Removed the call to get_dynamic_features
    rr.log(
        f"{entity_path}/current_frame/dynamic_features",
        rr.Points2D(dynamic_features, color=(255, 0, 0)),
    )  # Visualize in red


def log_camera_path(frame_id, entity_path, camera_path):  # Added 'camera_path' parameter
    """Logs the camera path up to the current frame_id."""
    # camera_path = get_camera_path(frame_id)  # Removed the call to get_camera_path
    rr.log(f"{entity_path}/camera_path", rr.LineStrips3D([camera_path]))


def log_sam_masks(frame_id, entity_path, masks):  # Added 'masks' parameter
    """Logs the semantic masks generated by SAM."""
    # masks = get_sam_masks(frame_id)  # Removed the call to get_sam_masks
    for i, mask in enumerate(masks):
        rr.log(f"{entity_path}/sam_masks/{i}", rr.SegmentationImage(mask))


def log_sam_prompts(frame_id, entity_path, prompts):  # Added 'prompts' parameter
    """Logs the prompts given to SAM."""
    # prompts = get_sam_prompts(frame_id)  # Removed the call to get_sam_prompts
    rr.log(
        f"{entity_path}/sam_prompts", rr.Points2D(prompts, color=(0, 0, 255))
    )  # Visualize in blue


def log_frame_points(frame_id, entity_path, frame, colors=None, accumulate=False):
    """
    Logs the map points visible in the current frame as 3d point cloud.
    
    Parameters:
    -----------
    frame_id : int
        The ID of the current frame
    entity_path : str
        Base path for the entity in the visualization
    frame : Frame
        The frame object containing the points
    colors : np.ndarray, optional
        Custom colors for the points
    accumulate : bool, default=False
        If True, accumulate points from all frames. If False, only show current frame.
    """
    if frame is None:
        return
    
    # Get all points from the frame that are not None (matched map points)
    with frame._lock_features:
        matched_points = [p for p in frame.points if p is not None]
    
    if not matched_points:
        # No points to log
        return
    
    # Extract 3D positions and colors
    points_3d = np.array([p.pt for p in matched_points])
    
    if colors is None:
        # Use the colors stored in the map points
        point_colors = np.array([p.color for p in matched_points]) / 255.0
    else:
        point_colors = colors
    
    # Determine the entity path based on accumulation mode
    if accumulate == True:
        # Use frame-specific path to accumulate all frames
        point_path = f"{entity_path}/frame_{frame_id}/map_points"
    else:
        # Use a consistent path so new frames replace old ones
        point_path = f"{entity_path}/current_map_points"

    radii = np.ones(points_3d.shape[0]) * point_size  # Set a default radius for points
    
    # Log the points to rerun
    rr.log(point_path, rr.Points3D(points_3d, colors=point_colors, radii=radii))
    
    return points_3d, point_colors


# def log_keyframes(kfs):
#     """ 
#     Input = List of keyframedata objects
#     Outpt = log all the images and masks of the keyframes for every timestamp
#     """
#     for i, kf in enumerate(kfs):
#         # Assuming kf is a KeyFrameData object with attributes 'img' and 'mask'
#         print("type of image, mask", type(kf.img), type(kf.dynamic_mask))
#         image = kf.img
#         mask = kf.dynamic_mask  # Assuming this is the mask you want to log
#         timestamp = kf.timestamp
        
#         # Log the image and mask
#         rr.log(f"keyframes/{timestamp}/image", rr.Image(image))
#         rr.log(f"keyframes/{timestamp}/mask", rr.Image(mask))


def log_keyframes_poses(kfs):
    """ 
    Input = List of KeyFrameData objects
    Output = Logs the poses of keyframes, also add image to the view. it should be a camera view with image - typical rerun viz
    """
    if not kfs:
        return
    
    import numpy as np
    import rerun as rr
    
    # Log each keyframe as a camera with its image
    for i, kf in enumerate(kfs):
        # Extract pose information (Tcw is camera-to-world transform)
        Rcw = kf.Rcw  # Rotation matrix
        tcw = kf.tcw  # Translation vector

        Hcw = np.eye(4)
        Hcw[:3, :3] = Rcw
        Hcw[:3, 3] = tcw
        # Convert to world-to-camera transform
        Hwc = np.linalg.inv(Hcw)
        # Convert to 3x3 rotation matrix and 3D translation vector
        Rwc = Hwc[:3, :3]
        twc = Hwc[:3, 3]
        # Temporarily calling Rcw as Rwc and tcw as twc for testing viz
        Rcw = Rwc
        tcw = twc
        
        # Create entity path
        entity_path = f"keyframes/poses/{kf.id}"
        
        # Create camera intrinsic matrix (assuming a pinhole camera model)
        if hasattr(kf.camera, 'K'):
            K = kf.camera.K
            
            # Get image dimensions
            width = kf.img.shape[1] if isinstance(kf.img, np.ndarray) else kf.img.width
            height = kf.img.shape[0] if isinstance(kf.img, np.ndarray) else kf.img.height
            
            # Extract focal length and principal point from K matrix
            fx = K[0, 0]
            fy = K[1, 1]
            cx = K[0, 2]
            cy = K[1, 2]
            
            # Log camera intrinsics
            rr.log(
                f"{entity_path}/image",
                rr.Pinhole(
                    resolution=[width, height],
                    focal_length=[fx, fy],
                    principal_point=[cx, cy]
                )
            )
        else:
            # Fallback if no calibration matrix is available
            width = kf.img.shape[1] if isinstance(kf.img, np.ndarray) else kf.img.width
            height = kf.img.shape[0] if isinstance(kf.img, np.ndarray) else kf.img.height
            
            rr.log(
                f"{entity_path}/image",
                rr.Pinhole(
                    resolution=[width, height],
                )
            )
        
        # Log the image
        if kf.img is not None:
            # Convert to RGB before logging
            rgb_img = ensure_rgb(kf.img)
            rr.log(f"{entity_path}/image/rgb", rr.Image(rgb_img))
        
        # Log the camera pose
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=tcw,
                mat3x3=Rcw,
            ),
        )
        
        # # Log frame ID as text
        # rr.log(
        #     f"{entity_path}/label",
        #     rr.TextAnnotation(
        #         text=f"KF {kf.id}",
        #         size=16,
        #         background_color=(0, 0, 0, 128),  # Semi-transparent black background
        #         text_color=(255, 255, 255, 255),  # White text
        #     ),
        # )
        
        # For visualization purposes, also log a point at the camera position
        rr.log(
            f"{entity_path}/position",
            rr.Points3D(
                positions=[-tcw],  # Negative of tcw is the camera position in world frame
                colors=[(0, 255, 0)],  # Green point
                radii=[point_size],  # Small point
            ),
        )
        
        # If there's a dynamic mask, log it too
        if hasattr(kf, 'dynamic_mask') and kf.dynamic_mask is not None:
            # For masks, no color conversion is typically needed as they're usually single channel
            rr.log(f"{entity_path}/image/mask", rr.Image(kf.dynamic_mask))

def log_mask_type(type="dynamic_mask", mask=None):
    """
    Logs a binary mask as a black-and-white image to Rerun.

    Args:
        type (str): Path name for the log in rerun.
        mask (np.ndarray or torch.Tensor): Mask with shape (H, W) or (1, H, W) or (B, 1, H, W).
    """
    if mask is None:
        raise ValueError("Mask must not be None")

    # Convert torch.Tensor to np.ndarray
    if hasattr(mask, "detach"):
        mask = mask.detach().cpu().numpy()

    # Handle batch dimensions if present
    if len(mask.shape) == 4:  # (B, 1, H, W)
        mask = mask[0, 0]
    elif len(mask.shape) == 3:  # (1, H, W)
        mask = mask[0]
    elif len(mask.shape) == 2:  # (H, W)
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {mask.shape}")

    # Ensure the mask is in uint8 format (0 or 255 for visualization)
    mask_img = (mask > 0).astype(np.uint8) * 255  # Convert boolean to 0/255

    # Log the image to Rerun
    rr.log(type, rr.Image(mask_img))

    return 0

def log_keyframes(kfs):
    """ 
    Input = List of KeyFrameData objects
    Output = Logs two concatenated images: one for all images and one for all masks,
             with numbered labels showing kf id, max 5 items per row
    """
    if not kfs:
        return
    
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    
    # Get dimensions from first image (assuming all images have same size)
    sample_img = kfs[0].img
    if isinstance(sample_img, np.ndarray):
        height, width = sample_img.shape[:2]
    else:
        height, width = sample_img.size
    
    # Calculate grid dimensions
    n_items = len(kfs)
    items_per_row = min(5, n_items)  # Max 5 items per row
    n_rows = (n_items + items_per_row - 1) // items_per_row  # Ceiling division
    
    # Create blank canvases for composite images
    total_width = width * items_per_row
    total_height = height * n_rows
    composite_img = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    composite_mask = Image.new('RGB', (total_width, total_height), (255, 255, 255))
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    # Fill the composite images
    for i, kf in enumerate(kfs):
        # Calculate position in grid
        row = i // items_per_row
        col = i % items_per_row
        x = col * width
        y = row * height
        
        # Convert images if necessary and paste them
        img = kf.img
        mask = kf.dynamic_mask
        print("Shape of mask to be logged", mask.shape)
        
        # Convert img to RGB if it's a NumPy array
        if isinstance(img, np.ndarray):
            img = ensure_rgb(img)  # Convert to RGB before creating PIL Image
            img = Image.fromarray(img)
        if isinstance(mask, np.ndarray):
            mask = Image.fromarray(mask)
            
        composite_img.paste(img, (x, y))
        composite_mask.paste(mask, (x, y))
        
        # Add number label
        draw_img = ImageDraw.Draw(composite_img)
        draw_mask = ImageDraw.Draw(composite_mask)
        label = str(i)
        draw_img.text((x + 5, y + 5), label, fill=(255, 0, 0), font=font)  # Red text for image
        draw_mask.text((x + 5, y + 5), label, fill=(255, 0, 0), font=font)  # Red text for mask
    
    # Convert to numpy arrays for logging
    composite_img_array = np.array(composite_img)  # PIL Images are already in RGB format
    composite_mask_array = np.array(composite_mask)
    
    # Log the composite images
    rr.log("keyframes/composite/image", rr.Image(composite_img_array))
    rr.log("keyframes/composite/mask", rr.Image(composite_mask_array))

def log_delaunay_points_3d(curr_delaunay_pts_3d, entity_path = "world"+"/delaunay__frame_points_3d"):
    """
    Logs the Delaunay triangulation points in 3D space to rerun.
    
    Parameters:
    -----------
    curr_delaunay_pts_3d : np.ndarray
        Array of Delaunay points in 3D space.
    entity_path : str
        Base path for the entity in the visualization.
    """
    if curr_delaunay_pts_3d is None:
        return
    
    # Log the Delaunay points in 3D
    rr.log(
        f"{entity_path}/delaunay_points",
        rr.Points3D(curr_delaunay_pts_3d, colors=(255, 0, 0)),  # Red color for Delaunay points
    )

def log_delaunay_map_points_3d(delaunay_map_points_3d, entity_path = "world"+"/delaunay_map_points_3d"):
    """
    Logs the Delaunay triangulation points in 3D space to rerun.
    
    Parameters:
    -----------
    curr_delaunay_pts_3d : np.ndarray
        Array of Delaunay points in 3D space.
    entity_path : str
        Base path for the entity in the visualization.
    """
    if delaunay_map_points_3d is None:
        return
    # Log the Delaunay points in 3D
    rr.log(
        f"{entity_path}/delaunay_map_points",
        rr.Points3D(delaunay_map_points_3d, colors=(0, 255, 0)),  # Green color for Delaunay points
    )

def log_image_kps_graph(entity_path,image, kps, graph ):
    """
    Logs the image with keypoints and graph to rerun.
    
    Parameters:
    -----------
    image : np.ndarray
        The image to log.
    kps : np.ndarray
        Keypoints to log.
    graph : np.ndarray
        Graph edges to log.
    entity_path : str
        Base path for the entity in the visualization.
    """
    if image is None or kps is None or graph is None:
        return
    image = ensure_rgb(image)  # Convert to RGB if needed
    
    # Log the image
    rr.log(f"{entity_path}/image", rr.Image(image))
    
    # Log the keypoints
    rr.log(f"{entity_path}/keypoints", rr.Points2D(kps, colors=(255, 0, 0)))  # Red color for keypoints
    
    # Draw edges on the image based on the graph
    cv2_img = image.copy()
    for edge in graph:
        pt1 = tuple(kps[edge[0]].astype(int))
        pt2 = tuple(kps[edge[1]].astype(int))
        cv2.line(cv2_img, pt1, pt2, (0, 255, 0), 2)  # Green color for edges
    rr.log(f"{entity_path}/image_with_graph", rr.Image(cv2_img))  # Log the image with edges

def log_all(
    frame_id,
    entity_path="world",
    # Add parameters for all the data you want to log
    local_map_points=None,
    global_map_points=None,
    key_frames=None,
    current_frame_image=None,
    features=None,
    dynamic_features=None,
    camera_path=None,
    sam_masks=None,
    sam_prompts=None,
    current_frame=None,  # Added parameter for the current frame
    accumulate_frame_points=False,  # New parameter to control point accumulation
):
    """Logs all the components of the SLAM system to rerun."""

    rr.set_time_sequence("frame", frame_id)

    # Add toggle for each logging component as needed
    if local_map_points is not None:
        log_local_map(frame_id, entity_path, local_map_points)
    if global_map_points is not None:
        log_global_map(frame_id, entity_path, global_map_points)
    if key_frames is not None:
        log_key_frames(frame_id, entity_path, key_frames)
    if current_frame_image is not None:
        # Convert to RGB before logging
        rgb_image = ensure_rgb(current_frame_image)
        log_current_frame(frame_id, entity_path, rgb_image)
    if features is not None:
        log_features(frame_id, entity_path, features)
    if dynamic_features is not None:
        log_dynamic_features(frame_id, entity_path, dynamic_features)
    if camera_path is not None:
        log_camera_path(frame_id, entity_path, camera_path)
    if sam_masks is not None:
        log_sam_masks(frame_id, entity_path, sam_masks)
    if sam_prompts is not None:
        log_sam_prompts(frame_id, entity_path, sam_prompts)
    if current_frame is not None:
        log_frame_points(frame_id, entity_path, current_frame, accumulate=accumulate_frame_points)