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
from utils_noise import *
from slam import Slam, SlamState
from delaunay_dynamic import DelaunayDynamic
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import DatasetType, SensorType
from trajectory_writer import TrajectoryWriter
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from thirdparty.LightGlue.lightglue import viz2d
from thirdparty.LightGlue.lightglue.utils import rbd
from dynamic_objects import DynamicObjects, DynamicObject
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
k_num_resample_prompts = 20
effective_distance_threshold = 0.2
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
    starting_img_id = 100 #210, 340, 400, 770   # you can start from a desired frame id if needed 
    img_id = starting_img_id

    # Set delaunay reference frame 
    slam.delaunay_ref_f = starting_img_id

    log_coordinate_axes(entity_path="world/Origin", pose=np.eye(4), scale=1)
    end_img_id =310

    sam2_pivot_end = slam.sam2_num_frames_to_propagate_backwards
    
    try:
        fake_img = dataset.getImageColor(img_id)
        fake_depth_image = dataset.getDepth(img_id)

        # Set full black mask by force with one channel

        dynamic_objects = DynamicObjects(mask_size=fake_img.shape[:2])
        dynamic_mask = dynamic_objects.get_combined_mask()
        print("Dynamic mask shape: ", dynamic_mask.shape) # (720, 1280)
        
        

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
                        """
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
                        """
                        


                        # curr_dense_pc = depth2pointcloud(depth, img, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50, scale=depth_factor)
                        curr_dense_pc = depth2pointcloud_with_mask(depth, img, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=dynamic_mask, scale=depth_factor)

                        curr_gt_timestamp, x,y,z, qx,qy,qz,qw, abs_scale  = groundtruth.getTimestampPoseAndAbsoluteScale(img_id)
                        cur_gt_Twc = xyzq2Tmat(x,y,z,qx,qy,qz,qw)
                        

                        cur_gt_Tcw = np.linalg.inv(cur_gt_Twc)
                        if not args.headless:
                            log_coordinate_axes(entity_path = "world/GT/Curr Frame Pose", pose = cur_gt_Twc, scale=1)
                            log_current_frame_pc(entity_path="world/GT/curr_scan/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_gt_Twc)
                            log_frame_dense_pc(frame_id=img_id, entity_path="world/GT/scans/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_gt_Twc)

                     
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
                            log_current_frame_pc(entity_path="world/slam/curr_scan/", points=curr_dense_pc.points, colors=curr_dense_pc.colors, pose = cur_Twc)

                        # Collect data for rerun visualization
                        global_map_points, global_map_colors = slam.map.get_points_as_np()
                        local_map_points, local_map_colors = slam.map.local_map.get_points_as_np()

                        if not args.headless:
                            log_local_map(entity_path="world/slam", points=local_map_points)
                            log_global_map(entity_path="world/slam", points=global_map_points, colors=global_map_colors)
                            log_current_frame_map_points(entity_path="world/slam", points=curr_frame_map_points, colors=curr_frame_map_colors)
                                

                        if not args.headless:
                            # Draw feature trails if map is available
                            if slam.map is not None:
                                try:
                                    img_draw = slam.map.draw_feature_trails(img)
                                    if img_draw is not None:
                                        print("feature trial")
                                        log_image("feature_trails - Green(Tracked in many frames; Blue(Tracked in less than 2 frames))", img_draw)
                                except Exception as e:
                                    print(f"Error drawing feature trails: {e}")                        
                        
                        ### Extract frame from k_frames_away
                        delaunay_ref_f = slam.delaunay_ref_f
                        k_frames_away = img_id - delaunay_ref_f 
                        print("img_id: ", img_id)   
                        print("cur_frame id: ", cur_frame.id)
                        print("delaunay_ref_f: ", delaunay_ref_f)
                        print("k_frames_away: ", k_frames_away)
                        print("img_id - delaunay_ref_f: ", img_id - delaunay_ref_f)
                        print("-k_frames_away -1: ", -k_frames_away-1)
                        # print number of frames in the map
                        print("Number of frames in the map: ", len(slam.map.frames))
                        
                        # Wait for the map to add the current frame to the frames list
                        if delaunay_ref_f == starting_img_id:
                            print("Waiting for the map to add the current frame to the frames list...")
                        
                        if img_id > starting_img_id+25:
                            # Get the frame from k_frames_away
                            k_frames_away_frame = slam.map.get_frame(-k_frames_away)
                            print("k_frames_away_frame: ", k_frames_away_frame)
                            print("current frame id: ", cur_frame.id)
                            print("k_frames_away_frame - id: ", k_frames_away_frame.id)

                            

                            k_frames_away_points, k_frames_away_colors = k_frames_away_frame.get_points_as_np()
                            k_frames_away_Tcw = k_frames_away_frame.pose
                            k_frames_away_Twc = np.linalg.inv(k_frames_away_Tcw)
                            
                            # Log the frame
                            log_coordinate_axes(entity_path="world/slam/k_frames_away", pose=k_frames_away_Twc, scale=0.5)
                            log_current_frame_map_points(entity_path="world/slam/k_frames_away", points=k_frames_away_points, colors=k_frames_away_colors)
                            k_frames_away_dense_pc = depth2pointcloud_with_mask((k_frames_away_frame.depth_img / k_frames_away_frame.camera.depth_factor) , k_frames_away_frame.img, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=k_frames_away_frame.dynamic_mask, scale=depth_factor)
                            print("k_frames_away_dense_pc: ", k_frames_away_dense_pc)
                            # log_frame_dense_pc(frame_id=img_id-k_frames_away, entity_path="world/slam/k_frames_away", points=k_frames_away_dense_pc.points, colors=k_frames_away_dense_pc.colors, pose=k_frames_away_Twc)
                            log_current_frame_pc(entity_path="world/slam/k_frames_away/curr_scan", points=k_frames_away_dense_pc.points, colors=k_frames_away_dense_pc.colors, pose=k_frames_away_Twc)
                            log_image("world/k_frames_away_img", k_frames_away_frame.img)
                            k_frames_away_frame.print_frame_stats(entity="k_frames_away")
                            
                            ## Checking the new class
                            delaunay_dynamic = DelaunayDynamic(camera = camera)
                            ref_feat, cur_feat, m_kpts0, m_kpts1, matches = delaunay_dynamic._extract_and_match_features(k_frames_away_frame, cur_frame, dynamic_mask)
                            # delaunay_dynamic._apply_delaunay_triangulation_and_get_graph(k_frames_away_frame, cur_frame, dynamic_mask)
                            delaunay_dynamic._update_graph_properties(k_frames_away_frame, cur_frame, dynamic_mask)
                            print("Exiting for diagnostics")
                            sys.exit(0)

                            # TODO
                            # 1. Get new kps,des for the current frame img and the k_frames_away frame img using a new feature tracker and matcher objects 
                            # 2. Get the matches between the two frames
                            # 3. Get prompts for SAM2, import and create SAM2 object. For SAM2 video object, create a tmp folder with imgs loaded and saved from dataloader - apply on all the images in the tmp folder, tmp folder contains all the images until the current frame and the next frame. when propagation is complete, 
                            # 4. Get the masks from the SAM2 object for teh next frame and use it in the next iteration, currently is using MASKRCNN if you can see. have a new variable for the predicted mask, if exists use it, else MASKRCNN or black mask as already existing in the code.
                            # 5. For all the above steps, follow the same process used in test_delaunay.py 

                            # 1. Get new kps,des for the current frame img and the k_frames_away frame img using a new feature tracker and matcher objects
                            import torch
                            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            num_features = 1000
                            extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
                            matcher = LightGlue(features="superpoint").eval().to(device)

                            # Convert images to torch tensors
                            img1_torch_HWC = torch.from_numpy(k_frames_away_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                            img2_torch_HWC = torch.from_numpy(cur_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0

                            img1_torch_HWC = img1_torch_HWC.to(device)
                            img2_torch_HWC = img2_torch_HWC.to(device)

                            ref_feat = extractor.extract(img1_torch_HWC.to(device))
                            curr_feat = extractor.extract(img2_torch_HWC.to(device))

                            # Print # of keypoints before filtering
                            print("ref_feat keypoints before filtering depth: ", ref_feat["keypoints"].shape)
                            print("curr_feat keypoints before filtering depth: ", curr_feat["keypoints"].shape)

                            ref_depth = k_frames_away_frame.depth_img 
                            curr_depth = cur_frame.depth_img

                            # Print max and min of ref_depth and curr_depth
                            print("ref_depth max: ", ref_depth.max())   
                            print("ref_depth min: ", ref_depth.min())
                            print("curr_depth max: ", curr_depth.max())
                            print("curr_depth min: ", curr_depth.min())

                            ref_feat = filter_features_by_depth2(ref_feat, ref_depth)
                            curr_feat = filter_features_by_depth2(curr_feat, curr_depth)

                            # Print # of keypoints after filtering
                            print("ref_feat keypoints after filtering depth: ", ref_feat["keypoints"].shape)
                            print("curr_feat keypoints after filtering depth: ", curr_feat["keypoints"].shape)

                            # 2. Get the matches between the two frames
                            matches01 = matcher({"image0": ref_feat, "image1": curr_feat})
                            feats0, feats1, matches01 = [
                                rbd(x) for x in [ref_feat, curr_feat, matches01]
                            ]  # remove batch dimension

                            kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
                            m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

                            print("Number of keypoints in ref image: ", len(kpts0))
                            print("Number of keypoints in curr image: ", len(kpts1))
                            print("matches shape: ", matches.shape)

                            # Visualize matches
                            output_img = visualize_matches(k_frames_away_frame.img,curr_img,  kpts0, kpts1, matches, add_text=True)
                            log_image(entity=f"Matches between Frame curr and Frame k_frames_away", image=output_img)

                            ## TODO: Complete Delaunay triangulation on the current frames matched points and then check if the edges are dynamic and create prompts for the current frame. 
                            
                            # 1. Prepare keypoints for Delaunay triangulation
                            m_kpts0_np = m_kpts0.int().cpu().numpy()
                            m_kpts1_np = m_kpts1.int().cpu().numpy()

                            print("m_kpts0_np shape: ", m_kpts0_np.shape)
                            print("m_kpts1_np shape: ", m_kpts1_np.shape)
                            
                            # 2. Apply Delaunay triangulation to the matched keypoints
                            img_delaunay, tri = delaunay_image_kps(cur_frame.img, m_kpts1_np)
                            if not args.headless:
                                log_image("world/matched_kps/cur_frame/delaunay_triangulation", img_delaunay)
                            
                            # 3. Create a graph from the Delaunay triangulation
                            delaunay_graph = convert_delauany_to_networkx(tri)
                            
                            # 

                            
                            # 4. Unproject keypoints to 3D points
                            points0, z_1 = unproject_kps(ref_depth, 
                                                    m_kpts0_np, camera, k_frames_away_Twc, transform_to_world=True)
                            points1, z_2 = unproject_kps(curr_depth, 
                                                    m_kpts1_np, camera, cur_Twc, transform_to_world=True)
                            
                            print("points0 shape: ", points0.shape)
                            print("points1 shape: ", points1.shape)
                            print("points0 depth zero:", z_1 )
                            print("points1 depth zero:", z_2 )

                            # Visualize the 3D points
                            log_random_pc2(entity= "world/slam/kps matched in k_frames_away", points=points0, colors="green", radius=0.04)
                            log_random_pc2(entity= "world/slam/kps matched in cur_frame", points=points1, colors="blue", radius=0.04)


                            # 5. Calculate and store edge properties
                            # Store 3D distances in both frames
                            nx.set_edge_attributes(delaunay_graph, 
                                {edge: {'distance_3d': np.linalg.norm(points1[edge[0]] - points1[edge[1]])}
                                for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
                                
                            
                            nx.set_edge_attributes(delaunay_graph, 
                                {edge: {'distance_3d_other': np.linalg.norm(points0[edge[0]] - points0[edge[1]])} 
                                for edge in delaunay_graph.edges if edge[0] < len(points0) and edge[1] < len(points0)})
                            
                            # Store distance differences between frames
                            nx.set_edge_attributes(delaunay_graph, 
                                {edge: {'distance_diff': np.abs(delaunay_graph.edges[edge]['distance_3d'] - 
                                                                delaunay_graph.edges[edge]['distance_3d_other'])} 
                                for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
                            
                            # Store motion of nodes
                            nx.set_edge_attributes(delaunay_graph,
                                {edge: {'node1motion': np.linalg.norm(points0[edge[0]] - points1[edge[0]]), 
                                        'node2motion': np.linalg.norm(points0[edge[1]] - points1[edge[1]])} 
                                for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
                            
                            # Calculate angle changes and R*theta metric for rotation detection
                            for edge in delaunay_graph.edges:
                                if edge[0] < len(points0) and edge[1] < len(points0):
                                    # Get the points for the edge
                                    pt1 = points0[edge[0]]
                                    pt2 = points0[edge[1]]
                                    pt3 = points1[edge[0]]
                                    pt4 = points1[edge[1]]

                                    # Calculate the edge vectors
                                    vec1 = pt2 - pt1
                                    vec2 = pt4 - pt3

                                    # Calculate the angle between the two vectors
                                    angle = np.arccos(np.clip(np.dot(vec1, vec2) / 
                                                            (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

                                    # Store the angle, average length, and R*theta in the graph
                                    delaunay_graph.edges[edge]['angle_change'] = angle
                                    avg_length = (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / 2
                                    delaunay_graph.edges[edge]['avg_length'] = avg_length
                                    delaunay_graph.edges[edge]['R_theta'] = avg_length * angle

                            # Effective distance metric = sqrt (( l1cos(theta) - l2)**2 + l1sin(theta)**2)
                            nx.set_edge_attributes(delaunay_graph, 
                                {edge: {'effective_distance': np.sqrt((delaunay_graph.edges[edge]['distance_3d'] * 
                                                                np.cos(delaunay_graph.edges[edge]['angle_change']) - 
                                                                delaunay_graph.edges[edge]['distance_3d_other'])**2 + 
                                                                (delaunay_graph.edges[edge]['distance_3d'] * 
                                                                np.sin(delaunay_graph.edges[edge]['angle_change']))**2)} 
                                for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
                            
                            # Calulate change in the length of every node and store it in the graph
                            nx.set_node_attributes(delaunay_graph, 
                                {node: {'length_change_vector': points0[node] - points1[node]} 
                                for node in range(len(points1)) if node < len(points1)})

                            nx.set_node_attributes(delaunay_graph, 
                                {node: {'length_change': np.linalg.norm(points0[node] - points1[node])} 
                                for node in range(len(points1)) if node < len(points1)})    
                            
                            



                            # 6. Detect dynamic edges and remove them from the graph
                            modified_delaunay_graph = delaunay_graph.copy()
                            dynamic_edge_image = img_delaunay.copy()

                            for edge in list(modified_delaunay_graph.edges):
                                is_dynamic = False
                                # Check if edge properties exist
                                if 'distance_diff' in delaunay_graph.edges[edge] and delaunay_graph.edges[edge]['distance_diff'] > effective_distance_threshold:
                                    is_dynamic = True
                                elif 'R_theta' in delaunay_graph.edges[edge] and delaunay_graph.edges[edge]['R_theta'] > effective_distance_threshold:
                                    is_dynamic = True

                                # Check effective distance 
                                if 'effective_distance' in delaunay_graph.edges[edge] and delaunay_graph.edges[edge]['effective_distance'] > effective_distance_threshold:
                                    is_dynamic = True
                                if is_dynamic:
                                    # Draw dynamic edges in blue
                                    pt1 = tuple(m_kpts1_np[edge[0]])
                                    pt2 = tuple(m_kpts1_np[edge[1]])
                                    cv2.line(dynamic_edge_image, pt1, pt2, (255, 0, 0), 1)
                                    # Remove edge from graph
                                    modified_delaunay_graph.remove_edge(edge[0], edge[1])
                            
                            # if not args.headless:
                            if True:
                                log_image("delaunay_dynamic_edges", dynamic_edge_image)
                            
                            # 7. Extract connected components (potential dynamic objects)
                            connected_components = get_connected_components(modified_delaunay_graph)
                            print(f"Number of connected components: {len(connected_components)}")


                            ## For each connected component, get the avg motion of the nodes
                            for i, component in enumerate(connected_components):
                                avg_motion = np.mean([delaunay_graph.nodes[node]['length_change_vector'] for node in component.nodes], axis=0)
                                avg_length_change = np.mean([delaunay_graph.nodes[node]['length_change'] for node in component.nodes])
                                # print avg motion and number of nodes
                                print(f"Component {i}: Avg motion: {avg_motion}, Number of nodes: {len(component.nodes)}, Avg length change: {avg_length_change}")

                            ## For components with avg length change < effective_distance_threshold/factor, remove them from connected components list and create a new static_component by combining them 
                            static_component = nx.Graph()
                            for i, component in enumerate(connected_components):
                                avg_length_change = np.mean([delaunay_graph.nodes[node]['length_change'] for node in component.nodes])
                                if avg_length_change < effective_distance_threshold / 2:
                                    # Add nodes and edges to static component
                                    static_component.add_nodes_from(component.nodes)
                                    static_component.add_edges_from(component.edges)
                                    # Remove component from connected components list
                                    # connected_components.remove(component)
                                    print(f"Component {i} is static, avg length change: {avg_length_change}")
                                else:
                                    print(f"Component {i} is dynamic, avg length change: {avg_length_change}")

                            # For each component, creat a prompt for SAM2 and save all of them in a dictionay set it to the current frame. 




                            # 8. Visualize connected components
                            # if not args.headless:
                            if True:
                                connected_components_image = cur_frame.img.copy()
                                colors_for_components = [(255, 255, 255),(255,0,0), (0, 255, 0), (0, 0, 255), 
                                                        (255, 255, 0), (255, 0, 255), (0, 255, 255)]
                                
                                for i, component in enumerate(connected_components):
                                    color = colors_for_components[i % len(colors_for_components)]

                                    if component.number_of_nodes() == 1:
                                        # Draw single node in Large
                                        for node in component.nodes:
                                            pt = tuple(m_kpts1_np[node])
                                            cv2.circle(connected_components_image, pt, 5, color, -1)

                                    for edge in component.edges:
                                        pt1 = tuple(m_kpts1_np[edge[0]])
                                        pt2 = tuple(m_kpts1_np[edge[1]])
                                        if component.number_of_nodes() < 4:
                                            # Draw edge with large line since easier to see
                                            cv2.line(connected_components_image, pt1, pt2, color, 5)
                                        cv2.line(connected_components_image, pt1, pt2, color, 1)
                                
                                log_image("connected_components", connected_components_image)
                            
                            # 9. Prepare for SAM2 processing
                            # Create temporary directory for frames if needed
                            sam2_folder_path = create_temp_actual_folder(dataset_images_path_dir, img_id-k_segment_upto , img_id+2, temp_folder_name="sam2_temp_symlinks")
                            if dynamic_object_detected == True:
                                sam2_pivot_end = img_id-k_segment_upto

                            log_sam2_folder(entity="sam2", path=sam2_folder_path)
                            # 10. For SAM2 integration, prepare points from top components as prompts
                            sorted_components = sorted(connected_components, key=lambda x: x.number_of_nodes(), reverse=True)
                            
                            
                            
                            # TODO: Use SAM2 for segmentation with the second largest component
                            if len(sorted_components) > 0 and len(sorted_components[1].nodes) > 10:
                                # Get the  largest component
                                dynamic_component = sorted_components[1]
                                print(f"Second largest component: {dynamic_component.nodes}")
                                
                                # Get the points for the second largest component
                                points_for_sam2 = []
                                for node in dynamic_component.nodes:
                                    pt = tuple(m_kpts1_np[node])
                                    points_for_sam2.append(pt)
                                
                                # Convert to numpy array
                                points_for_sam2 = np.array(points_for_sam2, dtype=np.float32)
                                

                                dynamic_object_detected = True

                            else:
                                # No prompts
                                points_for_sam2 = np.array([], dtype=np.float32)

                            # Set negative prompts from the static component
                            negative_points_for_sam2 = []
                            for i, node in enumerate(static_component.nodes):
                                # Randomly sample points from the static component - 10 points
                                total = len(static_component.nodes)
                                if total > 10:
                                    if i % (total // 10) == 0:
                                        pt = tuple(m_kpts1_np[node])
                                        negative_points_for_sam2.append(pt)
                                else:
                                    pt = tuple(m_kpts1_np[node])
                                    negative_points_for_sam2.append(pt)

                            
                            # Apply SAM2 for the sam2_folder_path and visualize the results of the mask 
                            if sam2_folder_path and len(points_for_sam2) > 0:
                                try:
                                    print(f"Running SAM2 on folder: {sam2_folder_path}")
                                    # Print # files in folder
                                    print(f"Number of files in folder: {len(os.listdir(sam2_folder_path))}")
                                    
                                    # Initialize inference state with the video path
                                    inference_state = predictor.init_state(video_path=sam2_folder_path)
                                    
                                    # Select a subset of points for SAM2 prompt (no more than k_num_resample_prompts)
                                    if points_for_sam2.shape[0] > k_num_resample_prompts:
                                        # Randomly sample points
                                        indices = np.random.choice(points_for_sam2.shape[0], k_num_resample_prompts, replace=False)
                                        prompt_points = points_for_sam2[indices]
                                    else:
                                        prompt_points = points_for_sam2
                                        
                                    # Create labels array (all points are positive)
                                    labels = np.ones(prompt_points.shape[0], dtype=np.int32)

                                    len_of_folder = len(os.listdir(sam2_folder_path))
                                    
                                    
                                    # For each frame in the video, update prompts inference
                                    for i in range(sam2_pivot_end, img_id):
                                        prompts = {}
                                        # Access from the map 
                                        frame = slam.map.get_frame(i)
                                        if frame is not None:
                                            # Get the mask for the frame
                                            mask = frame.dynamic_mask
                                            frame_idx = i - sam2_pivot_end
                                            # Apply the mask as a prompt 
                                            frame_idx, obj_ids, video_res_masks = predictor.add_new_mask(
                                                                                                    inference_state,
                                                                                                    frame_idx,
                                                                                                    obj_id=1,
                                                                                                    mask=mask)
                                                                                                


                                            

                                    # Get the last frame in the video
                                    # Add points as prompts (use first frame in sequence)
                                    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                                        inference_state=inference_state,
                                        frame_idx=len_of_folder-2, # one before the last frame
                                        obj_id=1,     # First object ID
                                        points=prompt_points,
                                        labels=labels
                                    )



                                    dynamic_masks = sam2_logits_to_masks(out_mask_logits, curr_img, threshold=0)
                                    # Visualize the masks using rerun
                                    for i, mask in enumerate(dynamic_masks):
                                        mask = mask.astype(np.uint8) * 255
                                        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                                        mask = cv2.addWeighted(curr_img, 0.5, mask, 0.5, 0)
                                        log_image(f"sam2/mask_{i}", mask)

                                    

                                    # Propagate the masks trough the video sequence
                                    # Forward propagation
                                    forward_segments = {}
                                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                                        inference_state,
                                        start_frame_idx=len_of_folder-2,  # Your prompt frame
                                        reverse=False
                                    ):
                                        forward_segments[out_frame_idx] = {
                                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                            for i, out_obj_id in enumerate(out_obj_ids)
                                        }

                                    # Backward propagation
                                    backward_segments = {}
                                    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
                                        inference_state,
                                        start_frame_idx=len_of_folder-2,  # Your prompt frame
                                        reverse=True
                                    ):
                                        backward_segments[out_frame_idx] = {
                                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                                            for i, out_obj_id in enumerate(out_obj_ids)
                                        }

                                    # Reverse the backward segments to match the forward order
                                    backward_segments = {k: v for k, v in reversed(backward_segments.items())}

                                    # Combine the segments
                                    video_segments = {**backward_segments, **forward_segments}

                                    # Extract the masks for the next frame
                                    print("###################################################")
                                    print("video_segments: ", video_segments)
                                    print("###################################################")
                                    # Log all the masks in the video segments
                                    # Log all the masks in the video segments
                                    for i, (frame_idx, obj_masks) in enumerate(video_segments.items()):
                                        print(f"Frame {frame_idx} masks: {obj_masks}")
                                        for obj_id, mask in obj_masks.items():  # Iterate through dict items (obj_id, mask)
                                            if isinstance(mask, np.ndarray):  # Ensure mask is a NumPy array
                                                try:
                                                    # Extract first channel if mask has multiple dimensions
                                                    if len(mask.shape) == 3 and mask.shape[0] == 1:
                                                        mask = mask[0]  # Get the first mask if it's shaped [1, H, W]
                                                    
                                                    # Convert boolean to uint8
                                                    mask = mask.astype(np.uint8) * 255
                                                    
                                                    # Convert to BGR for blending (same as the working code above)
                                                    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                                                    
                                                    # Blend with the original image
                                                    mask = cv2.addWeighted(curr_img, 0.5, mask, 0.5, 0)

                                                    
                                                    # Log the result
                                                    log_image(f"mask_{i}_{obj_id}", mask)
                                                except Exception as e:
                                                    print(f"Error processing mask for object {obj_id}: {e}")
                                                    print(f"Mask shape: {mask.shape}, Image shape: {curr_img.shape}")
                                            else:
                                                print(f"Mask for object {obj_id} is not a valid NumPy array: {type(mask)}")        

                                    predictor.reset_state(inference_state)

                                    # Get the mask for the next frame   
                                    dynamic_mask = video_segments[len_of_folder-1][1]  # Assuming you want the first object
                                    # Convert to uint8, numpy and with shape np.zeros_like(img)[:, :, 0]
                                    dynamic_mask = dynamic_mask.astype(np.uint8) * 255 # Black and white mask means black - 0 and white - 255
                                    # Reshape to H x W
                                    dynamic_mask = dynamic_mask.reshape(curr_img.shape[0], curr_img.shape[1])
                                    kernel = np.ones((5, 5), np.uint8)
                                    dynamic_mask = cv2.dilate(dynamic_mask, kernel, iterations=5)
                                    
                                    # For each frame, update in the map.frame object, its mask and for the current frame, update its prompts as a dict for each object. 
                                    # Update the current frame with the mask
                                    cur_frame.dynamic_mask = dynamic_mask
                                    cur_frame.dynamic_prompts = {1: dynamic_mask}  # Assuming you want the first object

                                    # For each frame in the video, update the corresponding map.frame object with the mask 
                                    for frame_idx, obj_masks in video_segments.items(): # Frame_idx 0 for the first frame in the video, which is img_id-k_segment_upto frame
                                        for obj_id, mask in obj_masks.items():
                                            if isinstance(mask, np.ndarray):
                                                frame = slam.map.get_frame(img_id-k_segment_upto+frame_idx)
                                                if frame is not None:
                                                    # Convert to uint8, numpy and with shape np.zeros_like(img)[:, :, 0]
                                                    mask = mask.astype(np.uint8) * 255
                                                    # Reshape to H x W
                                                    mask = mask.reshape(frame.img.shape[0], frame.img.shape[1])
                                                    kernel = np.ones((8, 8), np.uint8)
                                                    mask = cv2.dilate(mask, kernel, iterations=5)
                                                    frame.dynamic_mask = mask
                                                    # Sample 20 points from the mask and add them to the frame as prompts
                                                    mask_points = np.argwhere(mask > 0)
                                                    if len(mask_points) > 20:
                                                        indices = np.random.choice(mask_points.shape[0], 20, replace=False)
                                                        mask_points = mask_points[indices]

                                                    # Convert to float32
                                                    mask_points = mask_points.astype(np.float32)
                                                    # Add the points to the frame as prompts
                                                    frame.dynamic_prompts[obj_id] = mask_points
                                                    
                                            else:   
                                                print(f"Mask for object {obj_id} is not a valid NumPy array: {type(mask)}")
                                    
                                    

                                    


                                except Exception as e:
                                    print(f"Error in SAM2 processing: {e}")
                                    print(traceback.format_exc())
                                    print("Continuing without SAM2 mask...")
                            else:
                                print("No SAM2 processing - either no folder path or no points available")

                                
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