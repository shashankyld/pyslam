import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import networkx as nx
import time
import rerun as rr
from config import Config
from dataset_factory import dataset_factory
from camera import PinholeCamera
from ground_truth import groundtruth_factory
from utils_geom import xyzq2Tmat
from utilities.utils_delaunay import *
from feature_tracker_configs import FeatureTrackerConfigs
from utils_rerun import *
from utils_depth import depth2pointcloud_with_mask
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from thirdparty.LightGlue.lightglue import viz2d
from thirdparty.LightGlue.lightglue.utils import rbd


def get_connected_components(G):
    """
    Returns a list of NetworkX Graph objects, each representing a connected component
    of the input graph G, sorted by number of nodes in decreasing order, including all edges
    within each component along with their data.
    
    Args:
        G (nx.Graph): Input NetworkX graph (undirected)
    
    Returns:
        list: List of nx.Graph objects, each a connected component, sorted by node count
    """
    def dfs(v, visited, component_nodes):
        """DFS to collect nodes of a connected component."""
        visited.add(v)
        component_nodes.add(v)
        for u in G.neighbors(v):
            if u not in visited:
                dfs(u, visited, component_nodes)
    
    visited = set()
    components = []
    
    for node in G.nodes():
        if node not in visited:
            component_nodes = set()
            dfs(node, visited, component_nodes)
            component_graph = nx.Graph()
            component_graph.add_nodes_from((n, G.nodes[n]) for n in component_nodes)
            for u in component_nodes:
                for v in G.neighbors(u):
                    if v in component_nodes and (u, v) not in component_graph.edges():
                        component_graph.add_edge(u, v, **G.edges[u, v])
            components.append(component_graph)
    
    components.sort(key=lambda x: x.number_of_nodes(), reverse=True)
    return components


def visualize_edge_distance_histogram(delaunay_graph):
    """
    Creates a histogram visualization of edge distance differences using OpenCV.
    
    Args:
        delaunay_graph: A NetworkX graph containing edges with 'distance_diff' attributes
        
    Returns:
        hist_img: A CV2 image showing the histogram
    """
    edge_lengths = [delaunay_graph.edges[edge]['distance_diff'] for edge in delaunay_graph.edges if 'distance_diff' in delaunay_graph.edges[edge]]
    hist_height = 400
    hist_width = 600
    bin_width = 15
    max_val = max(edge_lengths) if edge_lengths else 1.0
    min_val = 0
    num_bins = min(30, hist_width // bin_width)
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)
    
    hist, _ = np.histogram(edge_lengths, bins=bin_edges)
    hist_normalized = hist * hist_height / (max(hist) if max(hist) > 0 else 1)
    
    hist_img = np.ones((hist_height + 50, hist_width, 3), dtype=np.uint8) * 255
    
    for i in range(len(hist)):
        x1 = int(i * hist_width / len(hist))
        x2 = int((i + 1) * hist_width / len(hist)) - 1
        y1 = hist_height - int(hist_normalized[i])
        y2 = hist_height
        cv2.rectangle(hist_img, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.rectangle(hist_img, (x1, y1), (x2, y2), (0, 0, 0), 1)
    
    step = max(1, len(bin_edges) // 10)
    for i in range(0, len(bin_edges), step):
        x = int(i * hist_width / len(hist))
        label = f"{bin_edges[i]:.2f}"
        cv2.putText(hist_img, label, (x - 10, hist_height + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.putText(hist_img, "Histogram of Edge Length Differences", 
                (hist_width // 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(hist_img, "Distance Difference (meters)", 
                (hist_width // 3, hist_height + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(hist_img, "Frequency", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    mean_val = np.mean(edge_lengths) if edge_lengths else 0
    median_val = np.median(edge_lengths) if edge_lengths else 0
    stats_text = f"Mean: {mean_val:.3f}m  Median: {median_val:.3f}m  Max: {max_val:.3f}m"
    cv2.putText(hist_img, stats_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return hist_img


def unproject_kps(depth_img, kps, camera, pose, transform_to_world=False):
    """
    Unproject keypoints to 3D points.
    
    Args:
        depth_img: Depth image with values in meters
        kps: Keypoint coordinates to be unprojected
        camera: Camera object with fx, fy, cx, cy
        pose: 4x4 pose matrix of the camera w.r.t. the world
        transform_to_world: If True, transform points to world coordinates using the pose
    
    Returns:
        points: 3D points in camera or world coordinates
    """
    points = []
    depth_zero_count = 0
    for kp in kps:
        x, y = int(kp[0]), int(kp[1])
        depth = depth_img[y, x]
        if depth == 0:
            depth_zero_count += 1
            continue
        z = depth
        x3d = (x - camera.cx) * z / camera.fx
        y3d = (y - camera.cy) * z / camera.fy
        points.append([x3d, y3d, z])
    
    points = np.array(points)
    if transform_to_world:
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world = pose @ points_homogeneous.T
        return points_world[:3].T
    
    print(f"Number of keypoints with zero depth: {depth_zero_count}")
    print(f"Number of valid keypoints: {len(points)}")
    print("Total keypoints:", len(kps))
    
    return points


def filter_features_by_depth(ref_feat, depth_scaled):
    """
    Filters keypoints from reference features where depth is zero.
    
    Args:
        ref_feat: Dictionary containing feature data (keypoints, keypoint_scores, descriptors)
        depth_scaled: Scaled depth map (numpy array)
    
    Returns:
        ref_feat: Filtered reference features
    """
    ref_feat_kps = ref_feat["keypoints"][0].int().cpu().numpy()
    valid_indices = [
        i for i, kp in enumerate(ref_feat_kps)
        if depth_scaled[int(kp[1]), int(kp[0])] != 0
    ]
    
    if len(valid_indices) < len(ref_feat_kps):
        print(f"Removing {len(ref_feat_kps) - len(valid_indices)} keypoints with zero depth")
        ref_feat["keypoints"] = ref_feat["keypoints"][:, valid_indices]
        ref_feat["keypoint_scores"] = ref_feat["keypoint_scores"][:, valid_indices]
        ref_feat["descriptors"] = ref_feat["descriptors"][:, valid_indices]
    
    return ref_feat


def display_frame(img, title="Frame"):
    """Display a frame with cv2"""
    log_image(entity=title, image=img)


def display_depth(depth_img, title="Depth"):
    """Display a depth image with cv2"""
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    log_image(entity=title, image=depth_vis)


def visualize_matches(img0, img1, kpts0, kpts1, matches, color=(0, 255, 0), thickness=2, radius=6, add_text=False):
    """
    Visualizes keypoint matches between two images.
    
    Args:
        img0: First image (torch 1,3,H,W) or (numpy H,W,3)
        img1: Second image (torch 1,3,H,W) or (numpy H,W,3)
        kpts0: Keypoints in the first image (torch.Tensor, Nx2)
        kpts1: Keypoints in the second image (torch.Tensor, Nx2)
        matches: Tensor of shape (M, 2) with indices (i, j) indicating matches
        color: Color of lines and circles (B, G, R)
        thickness: Thickness of connecting lines
        radius: Radius of keypoint circles
        add_text: Whether to add text
    
    Returns:
        output_img: Combined image with matches visualized
    """
    if isinstance(img0, torch.Tensor):
        img0 = img0.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()
    
    img0 = (img0).astype(np.uint8)
    img1 = (img1).astype(np.uint8)
    
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    output_img = np.hstack((img0, img1))
    
    for i in range(matches.shape[0]):
        pt0 = tuple(kpts0[matches[i, 0]].int().cpu().numpy())
        pt1 = tuple(kpts1[matches[i, 1]].int().cpu().numpy() + np.array([img0.shape[1], 0]))
        cv2.circle(output_img, pt0, radius, color, -1)
        cv2.circle(output_img, pt1, radius, color, -1)
    
    return output_img


if __name__ == "__main__":
    online_poses_path = "results/metrics_20250428_144655_frame_to_frame/trajectory_online.txt"
    
    poses = []
    with open(online_poses_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            numbers = list(map(float, line.strip().split()))
            assert len(numbers) == 12
            pose = np.array(numbers).reshape(3, 4)
            T = np.eye(4)
            T[:3, :4] = pose
            poses.append(T)
    
    poses = np.array(poses)
    print(poses.shape)
    print(poses[0])
    
    starting_frame = 30
    ending_frame = 55
    
    config = Config()
    dataset = dataset_factory(config)
    print(f"Dataset: {dataset.name}")
    print(f"Total frames: {dataset.num_frames}")
    
    camera = PinholeCamera(config)
    depth_factor = 1 / camera.depth_factor
    
    groundtruth = groundtruth_factory(config.dataset_settings)
    if groundtruth:
        print("Ground truth available")
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
    
    if dataset.isOk():
        frame_id = starting_frame
        img1 = dataset.getImageColor(frame_id)
        depth1 = dataset.getDepth(frame_id)
        
        if img1 is not None:
            print(f"Frame {frame_id} loaded, shape: {img1.shape}")
            display_frame(img1, f"Frame {frame_id}")
            if depth1 is not None:
                display_depth(depth1, f"Depth for Frame {frame_id}")
        else:
            print(f"Failed to load frame {frame_id}")
    
    if dataset.isOk():
        frame_id = ending_frame
        img2 = dataset.getImageColor(frame_id)
        depth2 = dataset.getDepth(frame_id)
        
        if img2 is not None:
            print(f"Frame {frame_id} loaded, shape: {img2.shape}")
            display_frame(img2, f"Frame {frame_id}")
            if depth2 is not None:
                display_depth(depth2, f"Depth for Frame {frame_id}")
        else:
            print(f"Failed to load frame {frame_id}")
    
    camera = PinholeCamera(config)
    depth_factor = 1 / camera.depth_factor
    feature_tracker_config = FeatureTrackerConfigs.LIGHTGLUE
    
    pc1 = depth2pointcloud_with_mask(depth1, img1, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=None, scale=depth_factor)
    pc1_points, pc1_colors = pc1.points, pc1.colors
    
    pc2 = depth2pointcloud_with_mask(depth2, img2, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=None, scale=depth_factor)
    pc2_points, pc2_colors = pc2.points, pc2.colors
    
    rerun_record_name = f"pyslam_{dataset.name}_{int(time.time())}"
    rr.init(rerun_record_name, spawn=True)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    log_coordinate_axes(entity_path="world/Origin", pose=np.eye(4), scale=1)
    
    pose1 = poses[0]
    pose2 = poses[ending_frame - starting_frame]
    print("pose1", pose1)
    print("pose2", pose2)
    
    log_frame_pc(frame_id=starting_frame, entity_path="world/GT/scans/", points=pc1_points, colors=pc1_colors, pose=pose1, fraction=1.0)
    log_frame_pc(frame_id=ending_frame, entity_path="world/GT/scans/", points=pc2_points, colors=pc2_colors, pose=pose2, fraction=1.0)
    
    print("Depth 1 max and min", np.max(depth1), np.min(depth1))
    print("Depth 2 max and min", np.max(depth2), np.min(depth2))
    
    depth1_scaled = depth1 / 1000
    depth2_scaled = depth2 / 1000
    print("Depth 1 scaled max and min", np.max(depth1_scaled), np.min(depth1_scaled))
    print("Depth 2 scaled max and min", np.max(depth2_scaled), np.min(depth2_scaled))
    
    num_features = 1000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)
    
    print("img1 shape", img1.shape)
    print("img2 shape", img2.shape)
    print("img1 dtype", img1.dtype)
    print("img2 dtype", img2.dtype)
    img1_torch_HWC = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_torch_HWC = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    
    img1_torch_HWC = img1_torch_HWC.to(device)
    img2_torch_HWC = img2_torch_HWC.to(device)
    
    ref_feat = extractor.extract(img1_torch_HWC.to(device))
    curr_feat = extractor.extract(img2_torch_HWC.to(device))
    
    ref_feat = filter_features_by_depth(ref_feat, depth1_scaled)
    curr_feat = filter_features_by_depth(curr_feat, depth2_scaled)
    
    matches01 = matcher({"image0": ref_feat, "image1": curr_feat})
    feats0, feats1, matches01 = [rbd(x) for x in [ref_feat, curr_feat, matches01]]
    
    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]
    
    print("Number of keypoints in ref image: ", len(kpts0))
    print("Number of keypoints in curr image: ", len(kpts1))
    print("shape of the image1: ", img1_torch_HWC.shape)
    print("shape of the image2: ", img2_torch_HWC.shape)
    print("matches shape: ", matches.shape)
    
    output_img = visualize_matches(img1, img2, kpts0, kpts1, matches, add_text=True)
    log_image(entity=f"Matches between Frame {starting_frame} and Frame {ending_frame}", image=output_img)
    
    kps0 = kpts0[matches[..., 0]].int().cpu().numpy()
    kps1 = kpts1[matches[..., 1]].int().cpu().numpy()
    print("kps0 shape", kps0.shape)
    print("kps1 shape", kps1.shape)
    
    points0 = unproject_kps(depth1_scaled, kps0, camera, pose1, transform_to_world=True)
    points1 = unproject_kps(depth2_scaled, kps1, camera, pose2, transform_to_world=True)
    print("points0 shape", points0.shape)
    print("points1 shape", points1.shape)
    
    log_random_pc2(entity="kps matched in frame 1", points=points0, colors="green", radius=0.04)
    log_random_pc2(entity="kps matched in frame 2", points=points1, colors="blue", radius=0.04)
    
    img1_delaunay, tri = delaunay_image_kps(img1, kps0)
    log_image(entity=f"Delaunay Triangulation on Frame {starting_frame}", image=img1_delaunay)
    
    delaunay_graph = convert_delauany_to_networkx(tri)
    annotated_image = img1_delaunay.copy()
    
    counter = 0
    for edge in delaunay_graph.edges:
        if counter % 20 != 0:
            counter += 1
            continue
        counter += 1
        idx1, idx2 = edge
        point1_idx = matches[idx1, 0].item()
        point2_idx = matches[idx2, 0].item()
        
        if idx1 >= len(kps0) or idx2 >= len(kps0):
            continue
        
        edge_length_3d = np.linalg.norm(points0[idx1] - points0[idx2])
        midpoint_2d = ((kps0[idx1][0] + kps0[idx2][0]) // 2, 
                      (kps0[idx1][1] + kps0[idx2][1]) // 2)
        cv2.putText(annotated_image, f"{edge_length_3d:.2f}m", 
                    midpoint_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        pt1 = tuple(kps0[idx1])
        pt2 = tuple(kps0[idx2])
        cv2.line(annotated_image, pt1, pt2, (0, 0, 255), 1)
    
    log_image(entity="Delaunay Triangulation with 3D Distances", image=annotated_image)
    
    nx.set_edge_attributes(delaunay_graph, 
                           {edge: {'distance_3d': np.linalg.norm(points0[edge[0]] - points0[edge[1]])}
                            for edge in delaunay_graph.edges if edge[0] < len(points0) and edge[1] < len(points0)})
    nx.set_edge_attributes(delaunay_graph, 
                           {edge: {'distance_3d_other': np.linalg.norm(points1[edge[0]] - points1[edge[1]])} 
                            for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
    nx.set_edge_attributes(delaunay_graph, 
                           {edge: {'distance_diff': np.abs(delaunay_graph.edges[edge]['distance_3d'] - delaunay_graph.edges[edge]['distance_3d_other'])} 
                            for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
    nx.set_edge_attributes(delaunay_graph,
                           {edge: {'node1motion': np.linalg.norm(points0[edge[0]] - points1[edge[0]]), 
                                   'node2motion': np.linalg.norm(points0[edge[1]] - points1[edge[1]])} 
                            for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
    
    for edge in delaunay_graph.edges:
        if edge[0] < len(points0) and edge[1] < len(points0):
            pt1 = points0[edge[0]]
            pt2 = points0[edge[1]]
            pt3 = points1[edge[0]]
            pt4 = points1[edge[1]]
            vec1 = pt2 - pt1
            vec2 = pt4 - pt3
            angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))
            delaunay_graph.edges[edge]['angle_change'] = angle
            avg_length = (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / 2
            delaunay_graph.edges[edge]['avg_length'] = avg_length
            delaunay_graph.edges[edge]['R_theta'] = avg_length * angle
    
    hist_img = visualize_edge_distance_histogram(delaunay_graph)
    log_image(entity="Edge Length Differences Histogram", image=hist_img)
    
    print("Delaunay Graph Info:")
    print("Number of edges in the graph:", delaunay_graph.number_of_edges())
    print("Number of nodes in the graph:", delaunay_graph.number_of_nodes())
    print("Number of connected components in the graph:", nx.number_connected_components(delaunay_graph))
    largest_cc = max(nx.connected_components(delaunay_graph), key=len)
    print("Largest connected component size:", len(largest_cc))
    
    modified_delaunay_graph = delaunay_graph.copy()
    dynamic_edge_image = img1_delaunay.copy()
    
    for edge in list(modified_delaunay_graph.edges):
        if delaunay_graph.edges[edge]['distance_diff'] > 0.6:
            pt1 = tuple(kps0[edge[0]])
            pt2 = tuple(kps0[edge[1]])
            cv2.line(dynamic_edge_image, pt1, pt2, (255, 0, 0), 1)
            modified_delaunay_graph.remove_edge(edge[0], edge[1])
        elif delaunay_graph.edges[edge]['R_theta'] > 0.25:
            pt1 = tuple(kps0[edge[0]])
            pt2 = tuple(kps0[edge[1]])
            cv2.line(dynamic_edge_image, pt1, pt2, (255, 0, 0), 1)
            modified_delaunay_graph.remove_edge(edge[0], edge[1])
    
    log_image(entity="Dynamic Edges in Delaunay Triangulation", image=dynamic_edge_image)
    
    connected_components = get_connected_components(modified_delaunay_graph)
    print("Number of connected components in the modified graph:", len(connected_components))
    print("First connected component:", connected_components[0])
    
    connected_components_image = img1.copy()
    colors_for_components = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, component in enumerate(connected_components):
        if component.number_of_nodes() < 3:
            continue
        color = colors_for_components[i % len(colors_for_components)]
        for edge in component.edges:
            pt1 = tuple(kps0[edge[0]])
            pt2 = tuple(kps0[edge[1]])
            cv2.line(connected_components_image, pt1, pt2, color, 1)
    
    log_image(entity="Connected Components in Delaunay Triangulation", image=connected_components_image)
    
    tmp_dir = "tmp"
    import shutil
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    print(f"Created temporary directory - full path: {os.path.abspath(tmp_dir)}")
    cv2.imwrite(os.path.join(tmp_dir, f"{starting_frame}.jpg"), img1)
    cv2.imwrite(os.path.join(tmp_dir, f"{ending_frame}.jpg"), img2)
    print(f"Saved images to {tmp_dir}")
    
    SLAM_ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))
    
    try:
        from sam2.build_sam import build_sam2_video_predictor
        sam2_available = True
        print("SAM2 module imported successfully!")
    except ImportError:
        sam2_available = False
        print("SAM2 module not found. Skipping SAM2 processing.")
    
    if sam2_available:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device for SAM2: {device}")
        original_dir = os.getcwd()
        sam2_dir = os.path.join(SLAM_ROOT, "thirdparty", "sam2")
        os.chdir(sam2_dir)
        
        try:
            sam2_checkpoint = os.path.join("checkpoints", "sam2.1_hiera_tiny.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"
            
            if not os.path.exists(sam2_checkpoint):
                print(f"SAM2 checkpoint not found at: {os.path.join(os.getcwd(), sam2_checkpoint)}")
                print("Downloading checkpoint would be required - skipping SAM2 processing.")
                sam2_available = False
            else:
                predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
                print("SAM2 predictor loaded successfully!")
        except Exception as e:
            print(f"Error loading SAM2: {str(e)}")
            sam2_available = False
        finally:
            os.chdir(original_dir)
        
        if sam2_available:
            inference_state = predictor.init_state(video_path=tmp_dir)
            sorted_components = sorted(connected_components, key=lambda x: x.number_of_nodes(), reverse=True)
            
            print(f"Processing {len(sorted_components) - 1} dynamic components (skipping largest static component)")
            
            masks_by_frame = {starting_frame: {}, ending_frame: {}}
            
            for comp_idx, component in enumerate(sorted_components[1:], start=1):
                if component.number_of_nodes() < 3:
                    print(f"Skipping component {comp_idx} with only {component.number_of_nodes()} nodes")
                    continue
                    
                comp_nodes = list(component.nodes())
                print(f"Processing component {comp_idx} with {len(comp_nodes)} nodes")
                
                points = kps0[comp_nodes]
                labels = np.ones(len(points), dtype=np.int32)
                
                _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=comp_idx,
                    points=points,
                    labels=labels
                )
                
                masks_by_frame[starting_frame][comp_idx] = (out_mask_logits[0] > 0.0).cpu().numpy()
                
                mask_overlay = img1.copy()
                mask = (out_mask_logits[0] > 0.0).cpu().numpy()
                mask_colored = np.zeros_like(mask_overlay, dtype=np.uint8)
                
                color = np.random.randint(0, 255, size=3).tolist()
                mask_colored[mask[0]] = color
                cv2.addWeighted(mask_overlay, 0.7, mask_colored, 0.3, 0, mask_overlay)
                
                for pt in points:
                    cv2.circle(mask_overlay, tuple(pt), 5, (0, 255, 0), -1)
                
                log_image(entity=f"SAM2 Object {comp_idx} Segmentation", image=mask_overlay)
            
            for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
                if out_frame_idx == 1:
                    for i, obj_id in enumerate(out_obj_ids):
                        masks_by_frame[ending_frame][obj_id] = (out_mask_logits[i] > 0.0).cpu().numpy()
            
            mask_overlay_frame2 = img2.copy()
            for obj_id, mask in masks_by_frame[ending_frame].items():
                np.random.seed(obj_id)
                color = np.random.randint(0, 255, size=3).tolist()
                mask_colored = np.zeros_like(mask_overlay_frame2, dtype=np.uint8)
                mask_colored[mask[0]] = color
                cv2.addWeighted(mask_overlay_frame2, 0.7, mask_colored, 0.3, 0, mask_overlay_frame2)
            
            log_image(entity=f"SAM2_Propagated_Segmentation_Frame_{ending_frame}", image=mask_overlay_frame2)
            
            dynamic_objects_vis = img1.copy()
            static_region_mask = np.ones_like(dynamic_objects_vis[:,:,0], dtype=bool)
            
            for obj_id, mask in masks_by_frame[starting_frame].items():
                static_region_mask = static_region_mask & (~mask[0])
                np.random.seed(obj_id)
                color = np.random.randint(0, 255, size=3)
                mask_colored = np.zeros_like(dynamic_objects_vis, dtype=np.uint8)
                mask_colored[mask[0]] = color
                dynamic_objects_vis = cv2.addWeighted(dynamic_objects_vis, 0.7, mask_colored, 0.3, 0)
            
            dynamic_objects_vis[static_region_mask] = (dynamic_objects_vis[static_region_mask] * 0.7).astype(np.uint8)
            log_image(entity="Static_vs_Dynamic_Regions", image=dynamic_objects_vis)