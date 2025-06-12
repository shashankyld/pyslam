#!/usr/bin/env -S python3 -O
"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""
import logging 
import rerun as rr
from utilities.utils_rerun import *
import cv2
import time 
import os
import sys
import numpy as np
import json
from test_delaunay import get_connected_components, visualize_edge_distance_histogram, unproject_kps, filter_features_by_depth, filter_features_by_depth2, display_depth, display_frame, visualize_matches
from matplotlib import pyplot as plt
import platform 
from utilities.utils_delaunay import *
from config import Config
from utils_sam2 import *
from slam import Slam, SlamState
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import DatasetType, SensorType
from trajectory_writer import TrajectoryWriter
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from thirdparty.LightGlue.lightglue import viz2d
from thirdparty.LightGlue.lightglue.utils import rbd

# from viewer3D import Viewer3D
from utils_sys import getchar, Printer, force_kill_all_and_exit
from utils_img import ImgWriter
from utils_eval import eval_ate
from utils_geom_trajectory import find_poses_associations
from utils_geom import xyzq2Tmat
from utils_colors import GlColors
from utils_serialization import SerializableEnumEncoder
from utils_maskrcnn import MaskRCNNUtils 
from feature_tracker_configs import FeatureTrackerConfigs
from utils_draw import *
from loop_detector_configs import LoopDetectorConfigs

from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
from utils_depth import img_from_depth, filter_shadow_points, depth2pointcloud, depth2pointcloud_with_mask
from search_points import *
from config_parameters import Parameters  
from utils_depth import *
from rerun_interface import Rerun

from datetime import datetime
import traceback

import argparse


datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
k_frames_away = 25
k_num_resample_prompts = 20
effective_distance_threshold = 0.5
k_segment_upto = 5 


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Optional path for custom configuration file')
    parser.add_argument('--no_output_date', action='store_true', help='Do not append date to output directory')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')    
    args = parser.parse_args()
    
    if args.config_path:
        config = Config(args.config_path) # use the custom configuration path file
    else:
        config = Config()
        
    if args.no_output_date:
        print('Not appending date to output directory')
        datetime_string = None

    dataset = dataset_factory(config)
    
    dataset_images_path_dir = config.dataset_path + "/"+ config.dataset_settings['name'] + '/rgb_jpg/'
    images_paths_ordered = sorted(os.listdir(dataset_images_path_dir))
    print(f"images_paths_ordered: {images_paths_ordered}")

    is_monocular=(dataset.sensor_type==SensorType.MONOCULAR)    
    num_total_frames = dataset.num_frames

    online_trajectory_writer = None
    final_trajectory_writer = None
    if config.trajectory_saving_settings['save_trajectory']:
        trajectory_online_file_path, trajectory_final_file_path, trajectory_saving_base_path = config.get_trajectory_saving_paths(datetime_string)
        online_trajectory_writer = TrajectoryWriter(format_type=config.trajectory_saving_settings['format_type'], filename=trajectory_online_file_path)
        final_trajectory_writer = TrajectoryWriter(format_type=config.trajectory_saving_settings['format_type'], filename=trajectory_final_file_path)
    metrics_save_dir = trajectory_saving_base_path
        
    groundtruth = groundtruth_factory(config.dataset_settings)

    camera = PinholeCamera(config)
    depth_factor = 1/camera.depth_factor #eg final value = 5000
    
    # Select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_tracker_config = FeatureTrackerConfigs.ORB2 # ORB2
        
    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
    # LoopDetectorConfigs: DBOW2, DBOW2_INDEPENDENT, DBOW3, DBOW3_INDEPENDENT, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES  etc.
    # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
    loop_detection_config = LoopDetectorConfigs.DBOW3 # DBOW3

    # Override the feature tracker and loop detector configuration from the `settings` file
    if config.feature_tracker_config_name is not None:  # Check if we set `FeatureTrackerConfig.name` in the `settings` file 
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(config.feature_tracker_config_name) # Override the feature tracker configuration from the `settings` file
    if config.num_features_to_extract > 0:             # Check if we set `FeatureTrackerConfig.nFeatures` in the `settings` file 
        Printer.yellow('Setting feature_tracker_config num_features from settings: ',config.num_features_to_extract)
        feature_tracker_config['num_features'] = config.num_features_to_extract  # Override the number of features from the `settings` file
    if config.loop_detection_config_name is not None:  # Check if we set `LoopDetectorConfig.name` in the `settings` file 
        loop_detection_config = LoopDetectorConfigs.get_config_from_name(config.loop_detection_config_name) # Override the loop detector configuration from the `settings` file
        
    Printer.green('feature_tracker_config: ',json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder))          
    Printer.green('loop_detection_config: ',json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder))

    # Using logging module: Available levels are DEBUG, INFO, WARNING, ERROR, CRITICAL 
    logging.basicConfig(level=logging.DEBUG)
    logging.debug("dataset: %s", dataset)
    logging.debug("groundtruth: %s", groundtruth)
    logging.debug("camera: %s", camera)
    logging.debug("feature_tracker_config: %s", feature_tracker_config)
    logging.debug("loop_detection_config: %s", loop_detection_config)
    logging.debug("dataset.sensor_type: %s", dataset.sensor_type)
    logging.debug("dataset.environment_type: %s", dataset.environmentType())
    logging.debug("dataset.scale_viewer_3d: %s", dataset.scale_viewer_3d)
    logging.debug("dataset.num_frames: %s", dataset.num_frames)    

    # Select your depth estimator in the front-end (EXPERIMENTAL, WIP)
    depth_estimator = None
    if Parameters.kUseDepthEstimatorInFrontEnd:
        Parameters.kVolumetricIntegrationUseDepthEstimator = False  # Just use this depth estimator in the front-end (This is not a choice, we are imposing it for avoiding computing the depth twice)
        # Select your depth estimator (see the file depth_estimator_factory.py)
        # DEPTH_ANYTHING_V2, DEPTH_PRO, DEPTH_RAFT_STEREO, DEPTH_SGBM, etc.
        depth_estimator_type = DepthEstimatorType.DEPTH_PRO
        max_depth = 20
        depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, max_depth=max_depth,
                                                  dataset_env_type=dataset.environmentType(), camera=camera) 
        Printer.green(f'Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}')       
    

    # Import SAM2 modules
    import sys
    import os
    SLAM_ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))
    
    from sam2.build_sam import build_sam2_video_predictor
    

    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for SAM2: {device}")
    
    # Change to SAM2 directory for loading checkpoints
    original_dir = os.getcwd()
    sam2_dir = os.path.join(SLAM_ROOT, "thirdparty", "sam2")
    os.chdir(sam2_dir)
    
    # Load SAM2 model
    sam2_checkpoint = os.path.join("checkpoints", "sam2.1_hiera_tiny.pt")
    model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
    
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    
    # Change back to original directory
    os.chdir(original_dir)

    # create SLAM object
    slam = Slam(camera, feature_tracker_config, 
                loop_detection_config, dataset.sensorType(), 
                environment_type=dataset.environmentType(), 
                config=config,
                headless=args.headless)
    slam.set_viewer_scale(dataset.scale_viewer_3d)
    time.sleep(1) # to show initial messages 
    
    # load system state if requested         
    if config.system_state_load: 
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
        print(f'viewer_scale: {viewer_scale}')
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

    if args.headless:
        # Do something in rerun   
        print("Running in headless mode")

    
    if groundtruth:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        # IF they are None, error in the dataset
        if gt_traj3d is None or gt_poses is None or gt_timestamps is None:
            raise ValueError("Groundtruth data is None. Please check the dataset.")
        
    # Initialize rerun for visualization
    # if not args.headless:
    if True:
        rerun_record_name = f"pyslam_{dataset.name}_{int(time.time())}"  # Add timestamp for uniqueness
        rr.init(rerun_record_name, spawn=True)
        
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        
            
    do_step = False          # proceed step by step on GUI 
    do_reset = False         # reset on GUI 
    is_paused = False        # pause/resume on GUI 
    is_map_save = False      # save map on GUI
    is_bundle_adjust = False # bundle adjust on GUI
    is_viewer_closed = False # viewer GUI was closed
    dynamic_object_detected = False # dynamic object detected
    key = None
    key_cv = None
    
    num_tracking_lost = 0
    num_frames = 0

    rr.set_time_seconds("frame_timestamp", 0)
    starting_img_id =150
     #210, 340, 400, 770   # you can start from a desired frame id if needed 
    img_id = starting_img_id
    log_coordinate_axes(entity_path="world/Origin", pose=np.eye(4), scale=1)
    end_img_id =200

    sam2_pivot_end = starting_img_id
    
    try:
        fake_img = dataset.getImageColor(img_id)
        # Set full black mask by force with one channel
        dynamic_mask = np.zeros_like(fake_img)[:, :, 0]
        print("Dynamic mask shape: ", dynamic_mask.shape) # (480, 640)

        print("Entering main loop...")
        while not is_viewer_closed:
            
            img, img_right, depth = None, None, None    
            
            if do_step:
                Printer.orange('do step: ', do_step)
                
            if do_reset: 
                Printer.yellow('do reset: ', do_reset)
                slam.reset()
                   
            if not is_paused or do_step:
            
                if dataset.isOk():
                    print('..................................')               
                    img = dataset.getImageColor(img_id)
                    depth = dataset.getDepth(img_id) 
                    img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None

                    if img_id == end_img_id:
                        print("End image id reached, exiting...")
                        is_viewer_closed = True
                        break
                else:
                    # Dataset has ended, break the loop
                    print("Dataset has ended at frame:", img_id)
                    is_viewer_closed = True
                    break

                
                
                if img is not None:
                    timestamp = dataset.getTimestamp()          # get current timestamp 
                    next_timestamp = dataset.getNextTimestamp() # get next timestamp 
                    frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1
                    
                    # Set rerun time to current timestamp if available
                    if timestamp is not None:
                        rr.set_time_seconds("frame_timestamp", timestamp)

                    print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}') 
                    
                    time_start = None 
                    if img is not None:
                        time_start = time.time()    
                        
                        if depth is None and depth_estimator:
                            depth_prediction, pts3d_prediction = depth_estimator.infer(img, img_right)
                            if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                                depth = filter_shadow_points(depth_prediction) 
                            else: 
                                depth = depth_prediction 
                            
                            if not args.headless:
                                depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                                log_image("depth_prediction", depth_img)
                        
                        
                        # Entry point to dynamic object segmentation
                        #  TODO: Firstly make use of GPU, then try to see if this can be parallelized, I can see that loop detection code is much faster and is waiting for this code to finish 
                        
                        
                        maskrcnn = MaskRCNNUtils()
                        logging.debug("Estimating dynamic mask")
                        dynamic_mask = maskrcnn.human_mask(img)
                        # Visualize the mask
                        if not args.headless:
                            print("logging mask")
                            # log_mask_type("dynamic_mask", dynamic_mask)
                        # Dialte the mask to make it more robust - dialate a lot
                        kernel = np.ones((5, 5), np.uint8)
                        dynamic_mask = cv2.dilate(dynamic_mask, kernel, iterations=5)

                        # Visualize the mask
                        if not args.headless:
                            print("logging mask")

                            # log_mask_type("dynamic_mask_dilated", dynamic_mask)
                        
                        # Set full black mask by force with one channel
                        # dynamic_mask = np.zeros_like(img)[:, :, 0]
                        # print("Dynamic mask shape: ", dynamic_mask.shape) # (480, 640)


                        # curr_dense_pc = depth2pointcloud(depth, img, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50, scale=depth_factor)
                        curr_dense_pc = depth2pointcloud_with_mask(depth, img, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=dynamic_mask, scale=depth_factor)

                        curr_gt_timestamp, x,y,z, qx,qy,qz,qw, abs_scale  = groundtruth.getTimestampPoseAndAbsoluteScale(img_id)
                        cur_gt_Twc = xyzq2Tmat(x,y,z,qx,qy,qz,qw)
                        

                        cur_gt_Tcw = np.linalg.inv(cur_gt_Twc)
                        if not args.headless:
                            log_coordinate_axes(entity_path = "world/GT/Curr Frame Pose", pose = cur_gt_Twc, scale=1)
                            # log_current_frame_pc(entity_path="world/GT/curr_scan/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_gt_Twc)
                            # log_frame_dense_pc(frame_id=img_id, entity_path="world/GT/scans/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_gt_Twc)

                        ###################TRACKING#####################################     
                        slam.track(img, img_right, depth, img_id, timestamp, dynamic_mask=dynamic_mask)  # main SLAM function 
                        
                        
                        ###################CURRENT FRAME 
                        # Getting access to the current frame properties after being populated by the SLAM system
                        cur_frame = slam.tracking.f_cur  # Class Frame
                        curr_frame_map_points, curr_frame_map_colors = cur_frame.get_points_as_np()
                        curr_img = cur_frame.img
                        cur_Tcw = slam.tracking.f_cur.pose
                        print("cur_Tcw: ", cur_Tcw)
                        print("cur translation: ", slam.tracking.cur_t)
                        # Invert this 
                        cur_Twc = np.linalg.inv(cur_Tcw)
                        
                        ################# LOGGING Current Frame pose, point cloud, global map, local map, and mathced map points with current frame and current map
                        if not args.headless:
                            log_coordinate_axes("world/slam/Curr Frame Pose", pose=cur_Twc, scale=0.5)
                            log_frame_dense_pc(frame_id=img_id, entity_path="world/slam/scans/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_Twc)                                                  
                            # log_current_frame_pc(entity_path="world/slam/curr_scan/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_Twc)

                        # Collect data for rerun visualization
                        global_map_points, global_map_colors = slam.map.get_points_as_np()
                        local_map_points, local_map_colors = slam.map.local_map.get_points_as_np()

                        if not args.headless:
                            print("Logging current frame map points")
                            # log_local_map(entity_path="world/slam", points=local_map_points)
                            # log_global_map(entity_path="world/slam", points=global_map_points, colors=global_map_colors)
                            # log_current_frame_map_points(entity_path="world/slam", points=curr_frame_map_points, colors=curr_frame_map_colors)
                                

                        if not args.headless:
                            # Draw feature trails if map is available
                            if slam.map is not None:
                                try:
                                    img_draw = slam.map.draw_feature_trails(img)
                                    if img_draw is not None:
                                        print("feature trial")
                                        # log_image("feature_trails - Green(Tracked in many frames; Blue(Tracked in less than 2 frames))", img_draw)
                                except Exception as e:
                                    print(f"Error drawing feature trails: {e}")                        
                        
                        
                                
                    if online_trajectory_writer is not None and slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                        online_trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp)
                        
                    if time_start is not None: 
                        duration = time.time()-time_start
                        if(frame_duration > duration):
                            time.sleep(frame_duration-duration) 
                        
                    img_id += 1 
                    num_frames += 1
                else: 
                    time.sleep(0.1)     # img is None
                    if args.headless:
                        if not dataset.isOk():  # Only exit if we've reached the end of dataset
                            print("Dataset has ended at frame:", img_id)
                            break # exit from the loop if headless and dataset is finished
                    

                                  
            else:
                time.sleep(0.1)     # pause or do step on GUI                           
            
            if slam.tracking.state==SlamState.LOST:
                num_tracking_lost += 1                              
                    
            # manage interface infos  
            if is_map_save:
                slam.save_system_state(config.system_state_folder_path)
                dataset.save_info(config.system_state_folder_path)
                groundtruth.save(config.system_state_folder_path)
                Printer.blue('\nuncheck pause checkbox on GUI to continue...\n')    
                
            if is_bundle_adjust:
                slam.bundle_adjust()    
                Printer.blue('\nuncheck pause checkbox on GUI to continue...\n')
                
            # Break loop if we've processed all frames in headless mode
            if args.headless and img_id >= num_total_frames:
                print("Processed all frames in headless mode. Exiting...")
                break
                
        print("\nProcessing final metrics and saving trajectories...")
        
        # Compute metrics and save trajectories
        try: 
            est_poses, timestamps, ids = slam.get_final_trajectory()
            is_final = True
            
            if groundtruth:
                assoc_timestamps, assoc_est_poses, assoc_gt_poses = find_poses_associations(
                    timestamps, est_poses, gt_timestamps, gt_poses)        
                ape_stats, T_gt_est = eval_ate(
                    poses_est=assoc_est_poses, 
                    poses_gt=assoc_gt_poses, 
                    frame_ids=ids, 
                    curr_frame_id=img_id, 
                    is_final=is_final, 
                    is_monocular=is_monocular, 
                    save_dir=metrics_save_dir
                )
                Printer.green(f"EVO stats: {json.dumps(ape_stats, indent=4)}")
            
            if final_trajectory_writer:
                final_trajectory_writer.write_full_trajectory(est_poses, timestamps)
                final_trajectory_writer.close_file()
                
            # Save other metrics
            other_metrics_file_path = os.path.join(metrics_save_dir, 'other_metrics_info.txt')
            with open(other_metrics_file_path, 'w') as f:
                f.write(f'num_total_frames: {num_total_frames}\n')
                f.write(f'num_processed_frames: {num_frames}\n')
                f.write(f'num_lost_frames: {num_tracking_lost}\n')
                f.write(f'percent_lost: {num_tracking_lost/num_total_frames*100:.2f}\n')
            
            print(f"\nProcessed {num_frames} frames")
            print(f"Lost tracking in {num_tracking_lost} frames ({num_tracking_lost/num_total_frames*100:.2f}%)")
            
        except Exception as e:
            print('Exception while computing metrics: ', e)
            print(f'traceback: {traceback.format_exc()}')

    except Exception as e:
        print('Exception in main loop: ', e)
        print(f'traceback: {traceback.format_exc()}')
        is_viewer_closed = True
        if args.headless:
            force_kill_all_and_exit(verbose=True)

    finally:
        # Clean shutdown
        print("\nShutting down SLAM system...")
        try:
            # First stop SLAM components to avoid broken pipe errors
            if slam is not None:
                # Stop loop closing thread first
                if hasattr(slam, 'loop_closing') and slam.loop_closing is not None:
                    try:
                        slam.loop_closing.quit()
                        print("Loop closing thread stopped")
                    except Exception as e:
                        print(f"Error stopping loop closing: {e}")

                # Stop other SLAM components
                try:
                    slam.quit()
                    print("SLAM system stopped")
                except Exception as e:
                    print(f"Error stopping SLAM: {e}")

            # Close trajectory writers
            if online_trajectory_writer is not None:
                try:
                    online_trajectory_writer.close_file()
                    print("Online trajectory writer closed")
                except Exception as e:
                    print(f"Error closing online trajectory writer: {e}")
            
            if final_trajectory_writer is not None:
                try:
                    final_trajectory_writer.close_file()
                    print("Final trajectory writer closed")
                except Exception as e:
                    print(f"Error closing final trajectory writer: {e}")

            # Finally force kill remaining processes
            force_kill_all_and_exit(verbose=True)
            
        except Exception as e:
            print('Exception during shutdown:', e)
            print(f'traceback: {traceback.format_exc()}')
            force_kill_all_and_exit(verbose=True)

        if args.headless:
            force_kill_all_and_exit(verbose=True)