import argparse
import sys
from config import Config
import numpy as np
import cv2
import time
import platform
from slam import Slam, SlamState
from camera import PinholeCamera
from dataset import dataset_factory, SensorType
from ground_truth import groundtruth_factory, GroundTruth
from viewer3D import Viewer3D
from utils_sys import getchar, Printer
from utils_geom import inv_T
from feature_tracker_configs import FeatureTrackerConfigs
from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
import os 
import signal
import open3d as o3d
from utils_maskrcnn import MaskRCNNUtils 

def signal_handler(sig, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)
if __name__ == "__main__":
    config = Config()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=config.system_state_folder_path, help='path where we have saved the system state')
    parser.add_argument('-mask', '--mask', type=str, default="/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/pyslam/data/TUM/rgbd_bonn_crowd/masks", help='path to the precomputed masks for the frames')
    parser.add_argument('-o', '--output_path', required=False, type=str, default=config.system_state_folder_path + '_dense_reconstruction', help="Path to save the system state with the dense reconstruction")
    args = parser.parse_args()
    camera = PinholeCamera(config)
    feature_tracker_config = FeatureTrackerConfigs.TEST
    # Create SLAM object

    slam = Slam(camera, feature_tracker_config)
    slam.load_system_state(args.path) # load the system state
    keyframes =  slam.map.get_keyframes()
    print("Number of keyframes: ", len(keyframes))
    # Create a window for image visualization in OpenCV
    cv2.namedWindow('keyframe', cv2.WINDOW_NORMAL)
    # List to hold point clouds 
    point_clouds = []

    # If .ply file already exists, open it and bypass the visualization loop
    if os.path.exists(args.output_path + ".ply"):
        print("PLY file already exists. Loading it...")
        final_point_cloud = o3d.io.read_point_cloud(args.output_path + ".ply")
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name='3D Pose Visualization')
        # Add a coordinate frame to visualize the scene
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
        vis.add_geometry(coordinate_frame)
        vis.add_geometry(final_point_cloud)
        # Set an initial viewpoint (adjust parameters as needed) 
        view_control = vis.get_view_control()
        # Run the visualizer
        vis.run() 
        vis.destroy_window()
        sys.exit(0)

    # Iterate over the keyframes 
    for i in range(len(keyframes)):
        keyframe = keyframes[i]
        img = keyframe.img
        # Get frame id 
        frame_id = keyframe.id 
        
        
        # # Get the mask for the keyframe - name is same as rgb image with the similar frame id - time stamp 
        # mask_names_path = "/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/pyslam/data/TUM/rgbd_bonn_crowd/masks"
        # mask_ordered_names = sorted([f for f in os.listdir(mask_names_path) if f.endswith(".png")])
        # mask_name = mask_names_path + "/" + mask_ordered_names[frame_id]
        # mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)

        maskrcnn = MaskRCNNUtils()
        mask = maskrcnn.human_mask(img)
        # Dilate the mask with a kernel of 5x5
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4) # This is to make sure that the mask covers the entire human body

        
        
        pose = keyframe.pose  
        points, rgb_values = keyframe.get_dense_depth_map(transform_in_world=True, mask=mask)
        print("Number of points: ", len(points))
        print("Pose for keyframe: ", pose)
        # Create a point cloud from the points and RGB values
        if points is not None and rgb_values is not None:
            if len(points) == len(rgb_values):
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(points) 
                pc.colors = o3d.utility.Vector3dVector(rgb_values / 255.0) 
                point_clouds.append(pc) 
        # Update the window with the new image
        cv2.imshow('keyframe', img)
        # Add mask to the image to same window
        # mask = cv2.imread(mask_name)
        cv2.imshow('mask', mask)

        

        # Wait for a key press 
        key = cv2.waitKey(5)  
        if key == 27:  # ESC key
            break
    cv2.destroyAllWindows()
    # --- Visualization after the loop ---
    # Create a 3D visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Pose Visualization')
    # Add a coordinate frame to visualize the scene
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)
    vis.add_geometry(coordinate_frame)
    # Accumulate all point clouds into a single point cloud
    final_point_cloud = o3d.geometry.PointCloud()
    for pc in point_clouds:
        final_point_cloud += pc 
    # Save the point cloud to a file 
    o3d.io.write_point_cloud(args.output_path + ".ply", final_point_cloud)
    # Add the final accumulated point cloud to the visualizer
    vis.add_geometry(final_point_cloud)
    
    # Set an initial viewpoint (adjust parameters as needed) 
    view_control = vis.get_view_control()
    # Run the visualizer
    vis.run() 
    vis.destroy_window()