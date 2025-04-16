import rerun as rr
import numpy as np

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

# Example usage
if __name__ == "__main__":
    rr.init("Coordinate Axes Demo")
    
    # Log coordinate axes with identity matrix (at origin)
    log_coordinate_axes("world/origin_axes")
    
    # Log coordinate axes with a custom pose matrix
    pose1 = np.eye(4)
    pose1[:3, 3] = np.array([1.0, 1.0, 1.0])  # Translation
    log_coordinate_axes("world/translated_axes", pose=pose1, scale=2.0)
    
    # Log coordinate axes with rotation and translation
    pose2 = np.eye(4)
    # Create a rotation matrix (45 degrees around Z axis)
    theta = np.radians(45)
    c, s = np.cos(theta), np.sin(theta)
    pose2[:3, :3] = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
    pose2[:3, 3] = np.array([2.0, 0.0, 0.0])  # Translation
    log_coordinate_axes("world/rotated_axes", pose=pose2, scale=1.5)
    
    # Log multiple axes at different positions with various orientations
    for x in range(-2, 3, 2):
        for y in range(-2, 3, 2):
            pose = np.eye(4)
            pose[:3, 3] = np.array([x, y, 0])
            # Add a small rotation based on position
            angle = 0.1 * (x + y)
            c, s = np.cos(angle), np.sin(angle)
            pose[:3, :3] = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
            log_coordinate_axes(f"world/grid/axes_{x}_{y}", pose=pose, scale=0.5)
    
    # Connect to the Rerun viewer
    rr.connect()
    # Or use this to spawn the viewer automatically:
    # rr.script_main(args=[])
