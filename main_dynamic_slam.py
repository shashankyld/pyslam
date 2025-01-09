#!/usr/bin/env python3


# Platform: Linux (Ubuntu 20.04)

from config import Config
from dataset import dataset_factory
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
import time
import math



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
    rr.init("dynamic slam", spawn=True)
    # TODO : Load GT trajectory to rerun at the start 
    # TODO : CLue - check about the data type of gt_traj3d - it is np.float32 but is it valid?  
    # rr.log("GT/trajectory", rr.LineStrips3D([gt_traj3d], colors=[0, 255, 0], radii=0.008, labels=["GT trajectory"]))
    # # like point cloud 
    # rr.log("GT/trajectory", rr.Points3D(gt_traj3d, colors=[0, 255, 0], radii=0.008, labels=["GT trajectory"]))


    # Processing the dataset 
    img_id = 0
    while True: 
        if img_id == 25:
            break   

        img, depth_img = None, None 

        if dataset.isOk(): 
            logging.debug("dataset is ok") 
            img = dataset.getImage(img_id)
            depth_img = dataset.getDepth(img_id)
            logging.debug("img_id: %d", img_id)

        if img is not None:
            timestamp = dataset.getTimestamp() 
            next_timestamp = dataset.getNextTimestamp() 
            frame_duration = next_timestamp - timestamp if (timestamp is not None and next_timestamp is not None) else -1.0 
            logging.debug("image with id %d has timestamp %f and next_timestamp %f, frame_duration: %f", img_id, timestamp, next_timestamp, frame_duration)
            rr.set_time_sequence("frame", img_id) 
            logging.debug("logging data associated to id %d to rerun", img_id)

            rr.log("frame/camera", rr.Image(img))
            rr.log("frame/depth", rr.DepthImage(depth_img))
            point_cloud = depth2pointcloud(depth_img, img, 
                                       config.cam_settings["Camera.fx"], config.cam_settings["Camera.fy"], 
                                       config.cam_settings["Camera.cx"], config.cam_settings["Camera.cy"], 
                                       max_depth=100000.0, min_depth=0.0)
            rr.log("frame/point_cloud", rr.Points3D(point_cloud.points, colors=point_cloud.colors))
            img_id += 1
            time.sleep(0.01)
        else:
            logging.debug("Either Dataset is not ok or image is None")
            break




