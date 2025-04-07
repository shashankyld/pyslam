#!/usr/bin/env python3


# Platform: Linux (Ubuntu 20.04)

from config import Config
from dataset import dataset_factory, SensorType
from ground_truth import groundtruth_factory
import logging 
from camera import PinholeCamera
from feature_tracker_configs import FeatureTrackerConfigs
from loop_detector_configs import LoopDetectorConfigs
from slam import Slam, SlamState
import rerun as rr
import numpy as np
# from utils_rerun import log_image, log_common_map_points
from utils_depth import depth2pointcloud
from utils_maskrcnn import MaskRCNNUtils 
from utils_delaunay import filter_delaunay_edges_by_3d_distance
from utils_delaunay import filter_delaunay_edges_by_3d_distance_last_frame , get_connected_components, delaunay_dynamic_visualization
from utils_delaunay import draw_simplicies_on_image, convert_frame_to_kdtree, convert_frame_to_reference_dict, convert_frame_to_kdtree_masked
from utils_geom import hamming_distance, hamming_distances, l2_distance, l2_distances
from utils_draw import visualize_matched_kps , visualize_matched_edges, visualize_common_simplicies
from utils_draw import *
from utils_misc import remove_duplicates_from_index_arrays, convert_frame_to_delaunay_dict, delaunay_with_kps, delaunay_visualization, get_common_edges, draw_common_edges, draw_dynamic_edges, get_dynamic_edges, draw_static_edges, get_static_edges, get_connected_components_from_edges, draw_connected_components
from rerun_interface import Rerun
from utilities.utils_rerun import log_all, log_keyframes, log_keyframes_poses
from utilities.utils_rerun import *
import time
import math
import cv2
from config_parameters import Parameters  
from search_points import search_frame_by_projection
import random
from keyframe_data import KeyFrameData
from keyframe import KeyFrame
import sys
import torch
from sam2_kf_processor import SAM2KeyframeProcessor
from search_points import *
import open3d as o3d
import numpy as np

# Initialize the SAM2 processor
sam2_processor = SAM2KeyframeProcessor()



# --- Add thirdparty/sam2 to sys.path ---
# Get the absolute path to the SLAM project root
SLAM_ROOT = os.path.dirname(os.path.abspath(__file__))
# Add the thirdparty/sam2 directory to the Python path
sys.path.append(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))


if __name__ == "__main__":
    
    config = Config()
    dataset = dataset_factory(config) 

    groundtruth = groundtruth_factory(config.dataset_settings)
    
    
    # Using logging module: Available levels are DEBUG, INFO, WARNING, ERROR, CRITICAL 
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("dataset: %s", dataset)
    logging.debug("groundtruth: %s", groundtruth)

    camera = PinholeCamera(config)
    logging.debug("camera: %s", camera)

    num_features = 5000
    if config.num_features_to_extract > 0:
        num_features = config.num_features_to_extract   
    logging.debug("num_features overriden to: %d", num_features)

    # Setting feature detection and matching using feature tracker configs 
    feature_tracker_config = FeatureTrackerConfigs.ORB2  # Using ORB2 feature tracker
    logging.debug("feature_tracker_config: %s", feature_tracker_config)
    feature_tracker_config["num_features"] = num_features
    logging.debug("Overriding num_features in feature_tracker_config to: %d, this is defined by the slam script", num_features)

    # Setting loop closing 
    loop_detection_config = LoopDetectorConfigs.DBOW3 # Using DBOW3 loop detector

    # Setting SLAM object 
    slam = Slam(camera, feature_tracker_config, loop_detection_config, dataset.sensorType(), groundtruth=None, environment_type=dataset.environmentType())
    slam.set_viewer_scale(dataset.scale_viewer_3d) #TODO: Check if this is necessary, looks like it is used, but check if it is necessary by check different values of scale_viewer_3d
    logging.debug("slam: %s", slam)


    # Initialize SAM2 segmentation model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # Hot restart the slam system state  # TODO: Understand this part better 
    # load system state if requested         
    if config.system_state_load: 
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
        print(f'viewer_scale: {viewer_scale}')
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)
   
   # Loading Groundtruth trajectory
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
    print("gt_traj3d: ", gt_traj3d.shape)


    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

 
    poses = []
    point_clouds = []

    # Processing the dataset 
    starting_img_id = 215  # 215 is close to human entrance
    img_id = starting_img_id
    camera_path = []  # To collect camera positions for trajectory visualization
    while True: 


        img, depth_img = None, None 

        # Check if dataset is ok
        if dataset.isOk(): 
            logging.debug("dataset is ok") 
            img = dataset.getImage(img_id)
            img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
            depth_img = dataset.getDepth(img_id)
            logging.debug("img_id: %d", img_id)

        if img is not None:
            timestamp = dataset.getTimestamp() 
            next_timestamp = dataset.getNextTimestamp() 
            frame_duration = next_timestamp - timestamp if (timestamp is not None and next_timestamp is not None) else -1.0 
            logging.debug("image with id %d has timestamp %f and next_timestamp %f, frame_duration: %f", img_id, timestamp, next_timestamp, frame_duration)
            logging.debug("logging data associated to id %d to rerun", img_id)

            point_cloud = depth2pointcloud(depth_img, img, 
                                       config.cam_settings["Camera.fx"], config.cam_settings["Camera.fy"], 
                                       config.cam_settings["Camera.cx"], config.cam_settings["Camera.cy"], 
                                       max_depth=100000.0, min_depth=0.0)



            

            # Entry point to dynamic object segmentation
            #  TODO: Firstly make use of GPU, then try to see if this can be parallelized, I can see that loop detection code is much faster and is waiting for this code to finish 
            # maskrcnn = MaskRCNNUtils()
            # logging.debug("Estimating dynamic mask")
            # dynamic_mask = maskrcnn.human_mask(img)
            # # Visualize the mask
            # cv2.imshow("Dynamic Mask_prediction", dynamic_mask)
            # cv2.waitKey(1)
            # # Dialte the mask to make it more robust - dialate a lot
            # kernel = np.ones((5, 5), np.uint8)
            # dynamic_mask = cv2.dilate(dynamic_mask, kernel, iterations=5)

            # # Visualize the mask
            # cv2.imshow("Dynamic Mask", dynamic_mask)
            # cv2.waitKey(1)
            # # Set full black mask by force with one channel
            dynamic_mask = np.zeros_like(img)[:, :, 0]
            print("Dynamic mask shape: ", dynamic_mask.shape) # (480, 640)


            # SLAM processing
            time_start = time.time() 
            slam.track(img, img_right, depth_img, img_id, timestamp, mask = dynamic_mask)
            logging.debug("SLAM tracking took %f seconds", time.time() - time_start)


                


            # Getting access to the current frame properties after being populated by the SLAM system
            cur_frame = slam.tracking.f_cur  # Class Frame
            cur_frame_points, cur_frame_colors = cur_frame.get_points_as_np()
            curr_img = cur_frame.img

            visualize_frame_kps(cur_frame, "Current Frame", scale_factor=1)

            # Add current camera position to camera path for trajectory visualization
            if cur_frame is not None and cur_frame.pose is not None:
                camera_position = cur_frame.Ow
                camera_path.append(camera_position)

            # Collect data for rerun visualization
            global_map_points, global_map_colors = slam.map.get_points_as_np()
            local_map_points, local_map_colors = slam.map.local_map.get_points_as_np()

            
            # Check if cu_frame_points is a subset of global_map_points and also local_map_points
            if cur_frame_points is not None and global_map_points is not None:
                if len(cur_frame_points) > 0 and len(global_map_points) > 0:
                    # Check if cur_frame_points is a subset of global_map_points
                    is_subset = np.all(np.isin(cur_frame_points, global_map_points))
                    print("Is cur_frame_points a subset of global_map_points: ", is_subset)

                if len(cur_frame_points) > 0 and len(local_map_points) > 0:
                    # Check if cur_frame_points is a subset of local_map_points
                    is_subset = np.all(np.isin(cur_frame_points, local_map_points))
                    print("Is cur_frame_points a subset of local_map_points: ", is_subset)

                # Check inverse
                if len(global_map_points) > 0 and len(cur_frame_points) > 0:
                    # Check if global_map_points is a subset of cur_frame_points
                    is_subset = np.all(np.isin(global_map_points, cur_frame_points))
                    print("Is global_map_points a subset of cur_frame_points: ", is_subset)

                if len(local_map_points) > 0 and len(cur_frame_points) > 0:
                    # Check if local_map_points is a subset of cur_frame_points
                    is_subset = np.all(np.isin(local_map_points, cur_frame_points))
                    print("Is local_map_points a subset of cur_frame_points: ", is_subset)
            
            pc_pts = point_cloud.points
            pc_colors = point_cloud.colors

            # Frame Twc 
            Twc = cur_frame.Twc
            # Transform the point cloud to the world frame
            pc_pts = np.dot(Twc[:3, :3], pc_pts.T).T + Twc[:3, 3]
        


            # Check if the point cloud contains any points
            if len(pc_pts) > 0:
                   # Initialize an empty point cloud
                pc_pcd = o3d.geometry.PointCloud()
                # *** FIX: Update the points and colors of the EXISTING pc_pcd object ***
                pc_pcd.points = o3d.utility.Vector3dVector(pc_pts)
                pc_pcd.colors = o3d.utility.Vector3dVector(pc_colors)
                point_clouds.append(pc_pcd)
                # Visualize the point clouds and origin
                o3d.visualization.draw_geometries([origin] + point_clouds,
                                                  window_name="Point Cloud Visualization",
                                                  width=800, height=600,
                                                  left=50, top=50,
                                                  mesh_show_back_face=True)
            else:
                # Optional: Handle empty point clouds, e.g., clear the visualizer's points
                pc_pcd.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0]])) # Keep dummy or clear
                pc_pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 0]]))

                logging.warning("Empty point cloud detected for frame %d", img_id)


            # # # Log to rerun
            # log_all(
            #     frame_id=img_id,
            #     entity_path="world",
            #     local_map_points=local_map_points,
            #     global_map_points=global_map_points,
            #     current_frame_image=img,
            #     current_frame=cur_frame,
            #     camera_path=np.array(camera_path),
            #     accumulate_frame_points=True,  # New parameter to control point accumulation
            # )

            print("#################ABOUT MAP SNAPSHOTS#################")
            print("Map snapshots: " ,slam.map_snapshots.snapshots.keys())
            

            # Comparing [0-N, 1-N+1, 2-2+N, 3-3+N, .....]
            if img_id > starting_img_id + Parameters.kNumFramesAway - 1: 

                

                print("Processing frame id:", img_id)

                prev_frame = slam.map.get_frame(-(Parameters.kNumFramesAway + 1))
                print("Previous frame id: ", prev_frame.timestamp)
                if prev_frame is None:
                    print("Warning: Could not retrieve previous frame")
                    continue

                # Get mapsnapshot at the current timestamp - kNumFramesAway
                map_snapshot = slam.map_snapshots.get_snapshot(timestamp=prev_frame.timestamp)
                # print("Map snapshot: ", map_snapshot)
                print("Map snapshot keys: ", map_snapshot.keys())
                snapshot_points, snapshot_colors = map_snapshot["map_points"]["points"], map_snapshot["map_points"]["colors"]
           

                matched_indices_snap, matched_indices_frame, matched_points_snap_3d_arr, matched_kps_frame_2d_arr, snap_frame_img = search_common_points_snapshot_frame(map_snapshot,  cur_frame,max_reproj_distance=25, max_descriptor_distance=50, ratio_test = 0.8,visualize=True, frame_img = curr_img)
                print("Number of matched points between two snapshots: ", len(matched_indices_snap))
                print("Number of matched points in the previous snapshot: ", len(matched_points_snap_3d_arr))
                print("Number of matched points in the current frame: ", len(matched_kps_frame_2d_arr))

                
                # Update the current frame's Delaunay attributes with matched indices
                if len(matched_indices_frame) > 0:
                    cur_frame.update_delaunay_attributes(matched_indices_frame)
                    

                # Initialize detected keypoints if missing
                if prev_frame.kps is not None and prev_frame.kps_detected is None:
                    prev_frame.kps_detected = prev_frame.kps.copy()
                    prev_frame.kpsu_detected = prev_frame.kpsu.copy() if prev_frame.kpsu is not None else None
                    print("Initialized detected keypoints from stored keypoints")

                # Check if prev_frame has required attributes
                if not hasattr(prev_frame, 'kpsu') or prev_frame.kpsu is None:
                    print("Warning: Previous frame does not have keypoints properly initialized")
                    continue

                # Matching across two frames is needed.
                print("Number of detected keypoints in the current frame: ", len(cur_frame.kpsu))
                print("Number of detected keypoints in the previous frame: ", len(prev_frame.kpsu))

                print("Number of keypoints in the current frame: ", len(cur_frame.kps_detected))
                print("Number of keypoints in the previous frame: ", len(prev_frame.kps_detected))

                # Fix: Check if kpsu_delaunay exists and is not None before checking its length
                if hasattr(cur_frame, 'kpsu_delaunay') and cur_frame.kpsu_delaunay is not None and len(cur_frame.kpsu_delaunay) > 0:
                    cur_image_with_kps = visualize_frame_kps(cur_frame, "Current Frame", scale_factor=1)
                    prev_image_with_kps = visualize_frame_kps(prev_frame, "Previous Frame", scale_factor=1)
                  
                    tri_indices,tri_vertices,curr_delaunay_img = delaunay_with_kps(cur_frame, matched_indices_frame)
                 
                    curr_delaunay_pts_3d,_ = cur_frame.unproject_points_3d(matched_indices_frame)
                    # Using matched_indices_snap extract mapsnapshot points
                    delaunay_points_snap_3d_arr = np.array(matched_points_snap_3d_arr)

                    print("Length of delaunay_points_snap_3d_arr: ", len(delaunay_points_snap_3d_arr))
                    print("Length of curr_delaunay_pts_3d: ", len(curr_delaunay_pts_3d))



                ## FRAME POINTS
                print("##############ALL ABOUT FRAME POINTS#################")
                print("Number of points in the current frame: ", len(cur_frame.points)) # SAME AS KPS except some are NONE and others are MapPoints
                print("Type of points in the current frame: ", type(cur_frame.points[0]))

                # Matching across two frames 
                # idxs_ref, idxs_cur = slam.tracking.idxs_ref, slam.tracking.idxs_cur # This works only for the last frame.
                # PARAMS SHOULD BE A FUNC OF PARAMS.KNUMFRAMESAWAY
                idxs_ref, idxs_cur, found_points_count = search_frame_by_projection(prev_frame, cur_frame, max_reproj_distance=2*Parameters.kMaxReprojectionDistanceFrame,
                                                                                 max_descriptor_distance=0.5*slam.tracking.descriptor_distance_sigma,
                                                                                 is_monocular=(slam.tracking.sensor_type == SensorType.STEREO))
                print("Idxs ref: ", len(idxs_ref))
                print("Idxs cur: ", len(idxs_cur))
                print("Found points count: ", found_points_count)

                # show current frame image
                cv2.imshow("Current frame", curr_img) # Wait of 1ms
                cv2.waitKey(1)

                # Duplicate idxs check - duplicates imply that the same keypoint is matched to multiple keypoints - matching error. 
                # For now, naively remove duplicates. 

                if len(idxs_cur) != len(np.unique(idxs_cur)):
                    print("Warning: Duplicates found in idxs_cur")
                    values, counts = np.unique(idxs_cur, return_counts=True)
                    print("Duplicates:", len(values[counts > 1]))
                
                if len(idxs_ref) != len(np.unique(idxs_ref)):
                    print("Warning: Duplicates found in idxs_ref")
                    values, counts = np.unique(idxs_ref, return_counts=True)
                    print("Duplicates:", len(values[counts > 1]))
                    
                
                idxs_ref, idxs_cur = remove_duplicates_from_index_arrays(idxs_ref, idxs_cur)
                print("Idxs ref after removing duplicates: ", len(idxs_ref))
                print("Idxs cur after removing duplicates: ", len(idxs_cur))


                print("Number of kps for the current frame: ", len(cur_frame.kpsu))
                print("Number of kps for the current frame unchanged : ", len(cur_frame.kpsu_detected))
                # Number of kps in the map (cur_frame.points), where it is not None
                mathes_cur_kps_map = 0
                for i in range(len(cur_frame.points)):
                    if cur_frame.points[i] is not None:
                        mathes_cur_kps_map += 1
                print("Number of kps in the map: ", mathes_cur_kps_map) 
            

                if len(idxs_cur) > 3 and len(idxs_ref) > 3:
                    # Now applying Delaunay triangulation on the matched keypoints 
                    _, _, prev_delaunay_img = delaunay_with_kps(prev_frame, idxs_ref)
                    _, _, curr_delaunay_img = delaunay_with_kps(cur_frame, idxs_cur)
                    if Parameters.kShowDebugImages:
                        delaunay_visualization(prev_delaunay_img, curr_delaunay_img)


     
          
            if img_id == starting_img_id:
                print("Frame id: ", img_id)
                # # Initialize the prev_dict 
                # prev_dict = convert_frame_to_kdtree_masked(slam)
                # print("Prev dict - also the first frame - created: ", len(prev_dict))
                

            # Logging global and local map points to rerun
            logging.debug("logging global and local map points to rerun")
            map_points_xyz, map_points_colors = slam.map.get_points_as_np()
            local_map_points = slam.map.local_map.get_points_as_np()
        

            img_id += 1
            time.sleep(0.0001)
          
            if img_id ==300:
                # Save the map 
                slam.save_system_state("/home/shashank/Documents/UniBonn/thesis/pyslam/results/maskrcnn_dynamic_slam/maskrcnn_500_slam_state_2/")
                print("Map saved")
                break
        # When dataset is not ok or image is None
        else:
            logging.debug("Either Dataset is not ok or image is None")
            break


