#!/usr/bin/env -S python3 -O
"""
* Simplified version of main_demo_maker.py - Uses GT poses and logs dense point clouds
* Maintains keypoint extraction, matching and visualization using rerun
"""
import logging 
import rerun as rr
from utilities.utils_rerun import *
import cv2
import time 
import os
import numpy as np
import json
from datetime import datetime
import argparse

from config import Config
from camera import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import SensorType

from utils_sys import Printer, force_kill_all_and_exit
from utils_geom import xyzq2Tmat
from utils_serialization import SerializableEnumEncoder
from utils_depth import img_from_depth, depth2pointcloud_with_mask

# For keypoint extraction and matching
import torch
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint
from thirdparty.LightGlue.lightglue.utils import rbd
from test_delaunay import visualize_matches, filter_features_by_depth2

datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
k_frames_away = 25  # Reset reference every k_frames_away frames

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Optional path for custom configuration file')
    parser.add_argument('--no_output_date', action='store_true', help='Do not append date to output directory')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')    
    args = parser.parse_args()
    
    if args.config_path:
        config = Config(args.config_path)
    else:
        config = Config()
        
    if args.no_output_date:
        print('Not appending date to output directory')
        datetime_string = None

    # Initialize dataset 
    dataset = dataset_factory(config)
    print(f"Dataset loaded: {dataset.name}")
    
    # Get camera parameters
    camera = PinholeCamera(config)
    depth_factor = 1/camera.depth_factor
    
    # Set up groundtruth
    groundtruth = groundtruth_factory(config.dataset_settings)
    if groundtruth:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        if gt_traj3d is None or gt_poses is None or gt_timestamps is None:
            raise ValueError("Groundtruth data is None. Please check the dataset.")
    else:
        raise ValueError("Groundtruth is required for this demo")
    
    # Initialize rerun for visualization
    rerun_record_name = f"pyslam_gt_{dataset.name}_{int(time.time())}"
    rr.init(rerun_record_name, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_coordinate_axes(entity_path="world/Origin", pose=np.eye(4), scale=1)
    
    # Setup constants
    starting_img_id = 0
    end_img_id = 200
    img_id = starting_img_id
    ref_img_id = starting_img_id
    ref_img = None
    ref_depth = None
    ref_gt_Twc = None

    # Initialize device and feature extractor/matcher
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    num_features = 1000
    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    try:
        # Create blank mask for depth filtering
        fake_img = dataset.getImageColor(img_id)
        dynamic_mask = np.zeros_like(fake_img)[:, :, 0]
        
        print("Entering main loop...")
        is_viewer_closed = False
        
        while not is_viewer_closed and img_id <= end_img_id:
            print('..................................')
            
            # Get data from dataset
            if dataset.isOk():
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                
                if img is None or depth is None:
                    print(f"Missing image or depth at frame {img_id}")
                    img_id += 1
                    continue
                
                timestamp = dataset.getTimestamp()
                if timestamp is not None:
                    rr.set_time_seconds("frame_timestamp", timestamp)
                
                print(f'Processing image: {img_id}, timestamp: {timestamp}') 
                
                # Get GT pose for current frame
                curr_gt_timestamp, x, y, z, qx, qy, qz, qw, abs_scale = groundtruth.getTimestampPoseAndAbsoluteScale(img_id)
                curr_gt_Twc = xyzq2Tmat(x, y, z, qx, qy, qz, qw)
                # # Invert this 
                # curr_gt_Twc = np.linalg.inv(curr_gt_Twc)
                
                # Generate dense point cloud
                curr_dense_pc = depth2pointcloud_with_mask(
                    depth, img, camera.fx, camera.fy, camera.cx, camera.cy, 
                    max_depth=50, mask=dynamic_mask, scale=depth_factor
                )
                
                # Log current frame information
                log_coordinate_axes(entity_path="world/GT/Curr_Frame_Pose", pose=curr_gt_Twc, scale=1)
                log_frame_dense_pc(
                    frame_id=img_id, 
                    entity_path="world/GT/scans", 
                    points=curr_dense_pc.points, 
                    colors=curr_dense_pc.colors, 
                    pose=curr_gt_Twc
                )
                
                # Log the image
                log_image("current_image", img)
                
                # Set reference frame either at start or every k_frames_away frames
                if img_id == starting_img_id or (img_id - ref_img_id) >= k_frames_away:
                    print(f"Setting reference frame to {img_id}")
                    ref_img_id = img_id
                    ref_img = img.copy()
                    ref_depth = depth.copy()
                    ref_gt_Twc = curr_gt_Twc.copy()
                    
                    # Log reference frame information
                    log_coordinate_axes(entity_path="world/GT/Reference_Frame", pose=ref_gt_Twc, scale=0.8)
                    
                    # Create point cloud for reference frame
                    ref_dense_pc = depth2pointcloud_with_mask(
                        ref_depth, ref_img, camera.fx, camera.fy, camera.cx, camera.cy, 
                        max_depth=50, mask=dynamic_mask, scale=depth_factor
                    )
                    
                    # Log reference frame's point cloud
                    log_current_frame_pc(
                        entity_path="world/GT/reference", 
                        points=ref_dense_pc.points, 
                        colors=ref_dense_pc.colors, 
                        pose=ref_gt_Twc
                    )
                    
                    log_image("reference_image", ref_img)
                
                # # If we have a reference frame and current frame is different, extract and match features
                # if ref_img is not None and ref_img_id != img_id:
                #     # Calculate relative pose between reference and current
                #     ref_gt_Tcw = np.linalg.inv(ref_gt_Twc)
                #     relative_pose = ref_gt_Tcw @ curr_gt_Twc
                #     distance = np.linalg.norm(relative_pose[:3, 3])
                #     print(f"Distance from reference: {distance:.3f}m")
                    
                #     # Convert images to torch tensors
                #     ref_img_torch = torch.from_numpy(ref_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                #     cur_img_torch = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                    
                #     ref_img_torch = ref_img_torch.to(device)
                #     cur_img_torch = cur_img_torch.to(device)
                    
                #     # Extract features
                #     ref_feat = extractor.extract(ref_img_torch)
                #     curr_feat = extractor.extract(cur_img_torch)
                    
                #     print("Reference keypoints before filtering depth: ", ref_feat["keypoints"].shape)
                #     print("Current keypoints before filtering depth: ", curr_feat["keypoints"].shape)
                    
                #     print("Reference depth", ref_depth)
                #     # Max value for ref depth
                #     print("Max depth in reference frame: ", np.max(ref_depth))
                #     # Filter features by depth
                #     ref_feat = filter_features_by_depth2(ref_feat, ref_depth/depth_factor)
                #     curr_feat = filter_features_by_depth2(curr_feat, depth/depth_factor)
                    
                #     print("Reference keypoints after filtering depth: ", ref_feat["keypoints"].shape)
                #     print("Current keypoints after filtering depth: ", curr_feat["keypoints"].shape)
                    
                #     # Match features
                #     matches01 = matcher({"image0": ref_feat, "image1": curr_feat})
                #     feats0, feats1, matches01 = [rbd(x) for x in [ref_feat, curr_feat, matches01]]
                    
                #     kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
                #     m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
                    
                #     print(f"Matches found: {matches.shape[0]}")
                    
                #     # Visualize matches
                #     output_img = visualize_matches(ref_img, img, kpts0, kpts1, matches, add_text=True)
                #     log_image("matches", output_img)
                    
                #     # Log keypoints in 3D space (reference frame)
                #     if matches.shape[0] > 0:
                #         m_kpts0_np = m_kpts0.cpu().numpy().astype(int)
                #         m_kpts1_np = m_kpts1.cpu().numpy().astype(int)
                        
                #         # Create colored points for visualization
                #         ref_kp_colors = np.zeros((m_kpts0_np.shape[0], 3), dtype=np.uint8)
                #         ref_kp_colors[:, 1] = 255  # Green for reference keypoints
                        
                #         curr_kp_colors = np.zeros((m_kpts1_np.shape[0], 3), dtype=np.uint8)
                #         curr_kp_colors[:, 2] = 255  # Blue for current keypoints
                        
                #         # Log matched keypoints in 2D
                #         ref_kps_img = ref_img.copy()
                #         for kp in m_kpts0_np:
                #             cv2.circle(ref_kps_img, (kp[0], kp[1]), 5, (0, 255, 0), -1)
                #         log_image("reference_keypoints", ref_kps_img)
                        
                #         curr_kps_img = img.copy()
                #         for kp in m_kpts1_np:
                #             cv2.circle(curr_kps_img, (kp[0], kp[1]), 5, (0, 0, 255), -1)
                #         log_image("current_keypoints", curr_kps_img)
            
            img_id += 1
            if img_id > end_img_id:
                print("End image id reached, exiting...")
                is_viewer_closed = True
            
            # Small sleep to prevent UI lag
            time.sleep(0.05)
            
    except Exception as e:
        print('Exception in main loop: ', e)
        import traceback
        print(f'traceback: {traceback.format_exc()}')
    
    finally:
        print("\nFinished processing frames")
        force_kill_all_and_exit(verbose=True)