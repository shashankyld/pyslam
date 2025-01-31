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
from utils_delaunay import delaunay_visualization, filter_delaunay_edges_by_3d_distance, filter_delaunay_edges_by_3d_distance_last_frame , get_connected_components, delaunay_dynamic_visualization, draw_simplicies_on_image, convert_frame_to_kdtree, get_common_edges, convert_frame_to_reference_dict
from utils_geom import hamming_distance, hamming_distances, l2_distance, l2_distances
from utils_draw import visualize_matched_kps , visualize_matched_edges
from rerun_interface import Rerun
import time
import math
import cv2
from config_parameters import Parameters  
from search_points import search_frame_by_projection
import random
# from sam2.sam2_image_predictor import SAM2ImagePredictor





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

    num_features = 2000
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

    # Setting up Rerun
    # rr.init("dynamic slam", spawn=True)
    
    
    # TODO: Figure out a way to save the rerun logs, should be simple
    # rr.save("/home/shashank/Documents/UniBonn/thesis/pyslam/logs/save_test.rrd")

    # TODO : Load GT trajectory to rerun at the start 
    # TODO : CLue - check about the data type of gt_traj3d - it is np.float32 but is it valid?  
    # rr.log("GT/trajectory", rr.LineStrips3D([gt_traj3d], colors=[0, 255, 0], radii=0.008, labels=["GT trajectory"]))
    # # like point cloud 
    # rr.log("GT/trajectory", rr.Points3D(gt_traj3d, colors=[0, 255, 0], radii=0.008, labels=["GT trajectory"]))
    # Rerun.log_gt_trajectory(points=gt_traj3d)


    # Processing the dataset 
    starting_img_id = 0 # 215 is close to human entrance
    img_id = starting_img_id
    while True: 
        # if img_id == 2:
        #     break   

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
            # rr.set_time_sequence("frame", img_id) 
            logging.debug("logging data associated to id %d to rerun", img_id)

            # rr.log("frame/camera", rr.Image(img))
            # rr.log("frame/depth", rr.DepthImage(depth_img))
            point_cloud = depth2pointcloud(depth_img, img, 
                                       config.cam_settings["Camera.fx"], config.cam_settings["Camera.fy"], 
                                       config.cam_settings["Camera.cx"], config.cam_settings["Camera.cy"], 
                                       max_depth=100000.0, min_depth=0.0)
            # rr.log("frame/point_cloud", rr.Points3D(point_cloud.points, colors=point_cloud.colors))

            # Entry point to dynamic object segmentation
            #  TODO: Firstly make use of GPU, then try to see if this can be parallelized, I can see that loop detection code is much faster and is waiting for this code to finish 
            # maskrcnn = MaskRCNNUtils()
            # logging.debug("Estimating dynamic mask")
            # dynamic_mask = maskrcnn.human_mask(img)

            # Set full black mask by force with one channel
            # dynamic_mask = np.zeros_like(img)[:, :, 0]
            # rr.log("dynamic_mask", rr.Image(dynamic_mask))


            ## TODO: 1. After the mask is obtained from SAM2 from previous time stamp, Apply tracking as usual as done below

            # SLAM processing
            time_start = time.time() 
            slam.track(img, img_right, depth_img, img_id, timestamp, mask = None)
            logging.debug("SLAM tracking took %f seconds", time.time() - time_start)

            ## TODO: 
            # 1. After the tracking is done, get the current frame and apply delaunay triangulation - This gives 2D mesh connecting features close to each other in the image
            # 2. Get the 3D points for these features that already excludes the masked dynamic objects in the current frame
            # 3. Get the local map from previous frame (Or remove points in the latest local frame by removing points added after tracking is done, I guess this is valid only if the local map is updated after tracking is done)
            # 4. From 2D delaunay triangulation, create an efficient datastruture for pairing feautres with in the delaunay triangles
            # 5. For each edge in the delaunay triangle, track its length in 3D
            # 6. For rigid objects, the length of the edges should be constant, if not - then the edge is connecting two different rigid objects
            # 7. All static objects and world together can be considered as a single rigid object (highly likely that they will have the majority of the edges)
            # 8. Create an algotithm to seperate the features in 2D into connected graphs of rigid objects can be static or dynamic
            # 9. For all the connected components that have lower edges than the highest group, they are dynamic objects
            # 10. Now for the next frame, send these features detected as dynamic objects to the segmentation prompt for a better mask
            # 11. Actually, for every frame, and its delaunay triangulation, we can just use the edges in the current frame and query distance directly from the local map. 

                

            # # Task1 - Run Delaunay triangulation on the current frame
            # delaunay_image = delaunay_visualization(slam)
            # rr.log("delaunay_triangulation", rr.Image(delaunay_image))

            # Getting access to the current frame properties after being populated by the SLAM system
            cur_frame = slam.tracking.f_cur  # Class Frame
            cur_frame_points, cur_frame_colors = cur_frame.get_points_as_np()
            curr_img = cur_frame.img
            # Visualize the current frame image
            cv2.imshow("Current Frame", curr_img)

            # Add mask to the current frame 
            # cur_frame.add_mask(dynamic_mask)

            logging.debug("logging current frame points to rerun")
            # rr.log("frame/curr_frame_points", rr.Points3D(cur_frame_points, colors=cur_frame_colors, radii=0.01))
            
            # Task2 - delaunay triangulation to networkx graph
            # Filter delaunay edges by 3D distance

            # Time 
            time_start = time.time()
            curr_dict = convert_frame_to_kdtree(slam)
            print("Keys in the current dict: ", curr_dict.keys())
            curr_dict = convert_frame_to_reference_dict(slam)
            print("Keys in the current dict: ", curr_dict.keys())
            logging.debug("Converting frame to kdtree took %f seconds", time.time() - time_start)
  
            if img_id > starting_img_id:
                print("Iffff")
                print("Frame id: ", img_id)

                delaunay_image = draw_simplicies_on_image(curr_img, curr_dict)
                # Show this image 
                # cv2.imshow("Delaunay Triangulation", delaunay_image)
                # cv2.waitKey(2)
                # # We will have access to prev_dict. 
                prev_frame = slam.map.get_frame(-2)
                prev_img = prev_frame.img.copy()
                prev_delaunay_image = draw_simplicies_on_image(prev_img, prev_dict)
                # cv2.imshow("Prev Delaunay Triangulation", prev_delaunay_image)
                # cv2.waitKey(2)

                # Visualize the current frame image
                cv2.imshow("Current Frame", curr_img)
                cv2.waitKey(2)
                cv2.imshow("Previous Frame", prev_img)
                cv2.waitKey(2)

                
                ''' 
                    # dict = {"(x1,y1)": {"desc" : "desc1", "id" : 1, "neighbors" : ["(kp2, desc2)", "(kp3, desc3)"]}}
                    # prev_dict = {"(x1,y1)": {"desc" : "desc1", "id" : 1, "neighbors" : ["(kp4, desc4)", "(kp3, desc3)"]}}
                    # common_nodes = ["(x1,y1)"]
                    # common_neighbors = ["(kp3, desc3)"]
                    # common_edges = ["(x1,y1)_(kp3, desc3)"] - dist = kp3 - x1,y1
                '''

                # Matching across two frames is needed.
                print("Number of detected keypoints in the current frame: ", len(cur_frame.kpsu))
                print("Number of detected keypoints in the previous frame: ", len(prev_frame.kpsu))
                idxs_ref, idxs_cur = slam.tracking.idxs_ref, slam.tracking.idxs_cur


                print("Idxs ref: ", len(idxs_ref))
                print("Idxs cur: ", len(idxs_cur))

                
                if Parameters.kShowDebugImages:
                    visualize_matched_kps(prev_frame, cur_frame, idxs_ref, idxs_cur)

                common_edges_overall = get_common_edges(idxs_ref, idxs_cur, prev_dict, curr_dict, prev_frame, cur_frame)
                # print("Common edges overall in the previous frame: ", common_edges_overall_prev)

                if Parameters.kShowDebugImages:
                    visualize_matched_edges(prev_frame, cur_frame, idxs_ref, idxs_cur, common_edges_overall, fraction=1)
     
            prev_dict = curr_dict

            # Logging global and local map points to rerun
            logging.debug("logging global and local map points to rerun")
            map_points_xyz, map_points_colors = slam.map.get_points_as_np()
            # rr.log("map/points", rr.Points3D(map_points_xyz, colors=map_points_colors, radii=0.01))
            local_map_points = slam.map.local_map.get_points_as_np()
            # rr.log("local_map/points", rr.Points3D(local_map_points[0], colors=local_map_points[1], radii=0.02))
            
            # Drop the image from the frame object
            # slam.tracking.f_cur.drop_img() # TODO: Try to implement drop after two frames

            img_id += 1
            time.sleep(0.0001)
            print(img_id/351 * 100, "% progress is done")
            
            if img_id ==150:
                # Save the map 
                slam.save_system_state("/home/shashank/Documents/UniBonn/thesis/pyslam/results/maskrcnn_dynamic_slam/slam_state/")
                break
        # When dataset is not ok or image is None
        else:
            logging.debug("Either Dataset is not ok or image is None")
            break




