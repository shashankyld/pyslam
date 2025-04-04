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
from utils_rerun import log_image
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

    # Initialize rerun for visualization
    rerun_record_name = f"pyslam_{dataset.name}_{int(time.time())}"  # Add timestamp for uniqueness
    rr.init(rerun_record_name, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP)

    # Processing the dataset 
    starting_img_id = 215# 215 is close to human entrance
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

            # Set rerun time to current timestamp if available
            if timestamp is not None:
                rr.set_time_seconds("frame_timestamp", timestamp)

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

            log_local_map(frame_id=img_id, entity_path="world", points=local_map_points, colors=local_map_colors)
            log_global_map(frame_id=img_id, entity_path="world", points=global_map_points, colors=global_map_colors)
            log_current_frame_map_points(frame_id=img_id, entity_path="world", points=cur_frame_points, colors=cur_frame_colors)
            log_current_frame_pc(frame_id=img_id, entity_path="world", points=(point_cloud.points/5000), colors=point_cloud.colors)

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
            



            # # Log to rerun
            # log_all(
            #     frame_id=img_id,
            #     entity_path="world",
            #     local_map_points=local_map_points,
            #     global_map_points=global_map_points,
            #     current_frame_image=img,
            #     current_frame=cur_frame,
            #     camera_path=np.array(camera_path),
            #     accumulate_frame_points=False,  # New parameter to control point accumulation
            # )

            


            # Comparing [0-N, 1-N+1, 2-2+N, 3-3+N, .....]
            if img_id > starting_img_id + Parameters.kNumFramesAway - 1: 

                
                print("Processing frame id:", img_id)

                prev_frame = slam.map.get_frame(-(Parameters.kNumFramesAway + 1))
                if prev_frame is None:
                    print("Warning: Could not retrieve previous frame")
                    continue

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

                visualize_frame_kps(cur_frame, "Current Frame", scale_factor=1)
                visualize_frame_kps(prev_frame, "Previous Frame", scale_factor=1)

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


                ## TODO: PART OF THE CODE LOGS Keyframe images and masks (COMPLETE  )
                # # Print keyframes 
                # print("Keyframes: ", slam.map.get_keyframes()) # Ordered Set
                # kf_data = []
                # for kf in slam.map.get_keyframes():
                #     kf_data_i = KeyFrameData(kf)
                #     kf_data.append(kf_data_i)
                # log_keyframes(kf_data)
                # # log_keyframes_poses(kf_data)


                
                ## TODO: PART OF THE CODE THAT IMPLIMENTS SAM2 BASED SEGMENTATION (COMPLETE)
                # # Now add the code to process keyframes when new ones are created
                # if slam.map.num_keyframes() > 0:
                #     # Get all keyframes for processing
                #     keyframes = slam.map.get_keyframes()
                    
                #     # Process keyframes with SAM2
                #     print(f"Processing {len(keyframes)} keyframes with SAM2...")
                #     updated_keyframes = sam2_processor.process_keyframes(keyframes)
                

                ## TODO: VISUALIZATION OF KEYFRAME ONLY POINT CLOUDS (COMPLETE)
                # # If depth data is available, visualize it as a point cloud
                # if depth_img is not None and cur_frame is not None:
                #     # If the cur_frame is a keyframe, we can visualize the depth point clou
                #     if cur_frame.is_keyframe_candidate:
                        
                #         # Get point cloud from depth image
                #         point_cloud_3d, point_cloud_colors = cur_frame.get_dense_depth_map(
                #             transform_in_world=True, 
                #             mask=dynamic_mask if dynamic_mask is not None else None
                #         )
                        
                #         if point_cloud_3d is not None and len(point_cloud_3d) > 0:
                #             # Downsample the point cloud to avoid overwhelming visualization
                #             downsample_factor = 10  # Adjust as needed
                #             downsampled_points = point_cloud_3d[::downsample_factor]
                #             downsampled_colors = point_cloud_colors[::downsample_factor] / 255.0
                            
                #             # Log depth point cloud
                #             rr.log(
                #                 f"world/frame_{img_id}/depth_cloud",
                #                 rr.Points3D(
                #                     downsampled_points,
                #                     colors=downsampled_colors,
                #                     radii=0.01
                #                 )
                #             )

                        # Log prompt points on the image 
                        
                

                # prev_frame_dict = convert_frame_to_delaunay_dict(prev_frame, idxs_ref)
                # cur_frame_dict = convert_frame_to_delaunay_dict(cur_frame, idxs_cur)

                # common_edges = get_common_edges(prev_frame_dict, cur_frame_dict)
                
                # if Parameters.kShowDebugImages:
                #     draw_common_edges(prev_frame, cur_frame, common_edges)
            
                # dynamic_edges = get_dynamic_edges(cur_frame, prev_frame, common_edges, threshold=0.2)

                # if Parameters.kShowDebugImages:
                #     draw_dynamic_edges(prev_frame, cur_frame, dynamic_edges)
                
                # static_edges = get_static_edges(common_edges, dynamic_edges)

                # print("Static edges: ", static_edges)
                    
                
                # if Parameters.kShowDebugImages:
                #     draw_static_edges(prev_frame, cur_frame, static_edges)

                # if Parameters.kShowDebugImages:
                #     visualize_matched_kps(prev_frame, cur_frame, idxs_ref, idxs_cur)
             

     
          
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
                break
        # When dataset is not ok or image is None
        else:
            logging.debug("Either Dataset is not ok or image is None")
            break



