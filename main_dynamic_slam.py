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
from utils_misc import remove_duplicates_from_index_arrays, convert_frame_to_delaunay_dict, delaunay_with_kps, delaunay_visualization, get_common_edges, draw_common_edges, draw_dynamic_edges, get_dynamic_edges, draw_static_edges, get_static_edges, get_connected_components_from_edges, draw_connected_components
from rerun_interface import Rerun
import time
import math
import cv2
from config_parameters import Parameters  
from search_points import search_frame_by_projection
import random





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


    # Processing the dataset 
    starting_img_id = 0 # 215 is close to human entrance
    img_id = starting_img_id
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


            # SLAM processing
            time_start = time.time() 
            slam.track(img, img_right, depth_img, img_id, timestamp, mask = None)
            logging.debug("SLAM tracking took %f seconds", time.time() - time_start)


            # Getting access to the current frame properties after being populated by the SLAM system
            cur_frame = slam.tracking.f_cur  # Class Frame
            cur_frame_points, cur_frame_colors = cur_frame.get_points_as_np()
            curr_img = cur_frame.img




            if img_id > starting_img_id:
                print("Iffff")
                print("Frame id: ", img_id)

                # # We will have access to prev_dict. 
                prev_frame = slam.map.get_frame(-2)
                prev_img = prev_frame.img.copy()

                # Matching across two frames is needed.
                print("Number of detected keypoints in the current frame: ", len(cur_frame.kpsu))
                print("Number of detected keypoints in the previous frame: ", len(prev_frame.kpsu))

                # Matching across two frames 
                idxs_ref, idxs_cur = slam.tracking.idxs_ref, slam.tracking.idxs_cur

                print("Idxs ref: ", len(idxs_ref))
                print("Idxs cur: ", len(idxs_cur))


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


                if len(idxs_cur) > 3 and len(idxs_ref) > 3:
                    # Now applying Delaunay triangulation on the matched keypoints 
                    _, _, prev_delaunay_img = delaunay_with_kps(prev_frame, idxs_ref)
                    _, _, curr_delaunay_img = delaunay_with_kps(cur_frame, idxs_cur)
                    if Parameters.kShowDebugImages:
                        delaunay_visualization(prev_delaunay_img, curr_delaunay_img)

                prev_frame_dict = convert_frame_to_delaunay_dict(prev_frame, idxs_ref)
                cur_frame_dict = convert_frame_to_delaunay_dict(cur_frame, idxs_cur)

                common_edges = get_common_edges(prev_frame_dict, cur_frame_dict)
                
                if Parameters.kShowDebugImages:
                    draw_common_edges(prev_frame, cur_frame, common_edges)
            
                dynamic_edges = get_dynamic_edges(cur_frame, prev_frame, common_edges, threshold=0.2)

                if Parameters.kShowDebugImages:
                    draw_dynamic_edges(prev_frame, cur_frame, dynamic_edges)
                
                static_edges = get_static_edges(common_edges, dynamic_edges)

                print("Static edges: ", static_edges)
                    
                
                if Parameters.kShowDebugImages:
                    draw_static_edges(prev_frame, cur_frame, static_edges)

                if Parameters.kShowDebugImages:
                    visualize_matched_kps(prev_frame, cur_frame, idxs_ref, idxs_cur)
             

     
          
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
          
            
            if img_id ==290:
                # Save the map 
                slam.save_system_state("/home/shashank/Documents/UniBonn/thesis/pyslam/results/maskrcnn_dynamic_slam/slam_state/")
                break
        # When dataset is not ok or image is None
        else:
            logging.debug("Either Dataset is not ok or image is None")
            break



