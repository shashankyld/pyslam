import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
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
import time
import rerun as rr 
from utils_rerun import *

import networkx as nx


def get_connected_components(G):
    """
    Returns a list of.ConcurrentHashMap<Region, Set<Region>> NetworkX Graph objects, each representing a connected component
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
        
        # Explore neighbors
        for u in G.neighbors(v):
            if u not in visited:
                dfs(u, visited, component_nodes)
    
    visited = set()
    components = []
    
    # Iterate through all nodes to find unvisited ones
    for node in G.nodes():
        if node not in visited:
            component_nodes = set()
            
            # Run DFS to collect nodes of current component
            dfs(node, visited, component_nodes)
            
            # Create new subgraph for the component
            component_graph = nx.Graph()
            # Add nodes with their data
            component_graph.add_nodes_from((n, G.nodes[n]) for n in component_nodes)
            
            # Add all edges between nodes in the component with their data
            for u in component_nodes:
                for v in G.neighbors(u):
                    if v in component_nodes and (u, v) not in component_graph.edges():
                        component_graph.add_edge(u, v, **G.edges[u, v])
            
            components.append(component_graph)
    
    # Sort components by number of nodes in decreasing order
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
    # Get the edge lengths
    edge_lengths = [delaunay_graph.edges[edge]['distance_diff'] for edge in delaunay_graph.edges if 'distance_diff' in delaunay_graph.edges[edge]]

    # Create histogram using OpenCV
    # Define histogram parameters
    hist_height = 400
    hist_width = 600
    bin_width = 15
    max_val = max(edge_lengths) if edge_lengths else 1.0
    min_val = 0
    num_bins = min(30, hist_width // bin_width)  # Limit number of bins
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Calculate histogram
    hist, _ = np.histogram(edge_lengths, bins=bin_edges)
    hist_normalized = hist * hist_height / (max(hist) if max(hist) > 0 else 1)

    # Create a white image for the histogram
    hist_img = np.ones((hist_height + 50, hist_width, 3), dtype=np.uint8) * 255

    # Draw the histogram bars
    for i in range(len(hist)):
        x1 = int(i * hist_width / len(hist))
        x2 = int((i + 1) * hist_width / len(hist)) - 1
        y1 = hist_height - int(hist_normalized[i])
        y2 = hist_height
        cv2.rectangle(hist_img, (x1, y1), (x2, y2), (0, 0, 255), -1)
        cv2.rectangle(hist_img, (x1, y1), (x2, y2), (0, 0, 0), 1)  # Draw outline

    # Add x-axis labels (every few bins for readability)
    step = max(1, len(bin_edges) // 10)
    for i in range(0, len(bin_edges), step):
        x = int(i * hist_width / len(hist))
        label = f"{bin_edges[i]:.2f}"
        cv2.putText(hist_img, label, (x - 10, hist_height + 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Add title and labels
    cv2.putText(hist_img, "Histogram of Edge Length Differences", 
                (hist_width // 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(hist_img, "Distance Difference (meters)", 
                (hist_width // 3, hist_height + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(hist_img, "Frequency", 
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    # Add statistics to the image
    mean_val = np.mean(edge_lengths) if edge_lengths else 0
    median_val = np.median(edge_lengths) if edge_lengths else 0
    stats_text = f"Mean: {mean_val:.3f}m  Median: {median_val:.3f}m  Max: {max_val:.3f}m"
    cv2.putText(hist_img, stats_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return hist_img

def unproject_kps(depth_img, kps, camera, pose, transform_to_world=False):
    """
    input: 
    depth_img: depth image with values in meters
    kps: keypoint coordinates to be unprojected
    camera: camera object - with fx, fy, cx, cy that can be accessed by camera.fx, camera.fy, etc
    pose: 4x4 pose matrix of the camera wrt the world
    transform_to_world: if True, transform the points to world coordinates using the pose
    
    output:
    points: 3D points in the camera or world coordinates depending on the transform_to_world flag
    """
    # Unproject keypoints to 3D points
    points = []
    depth_zero_count = 0
    for kp in kps:
        x, y = int(kp[0]), int(kp[1])
        depth = depth_img[y, x]  # Get the depth value at the keypoint location
        if depth == 0:
            depth_zero_count += 1
            continue  # Skip if depth is zero
        z = depth
        x3d = (x - camera.cx) * z / camera.fx
        y3d = (y - camera.cy) * z / camera.fy
        points.append([x3d, y3d, z])

    points = np.array(points)
    if transform_to_world:
        # Transform points to world coordinates using the pose
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        points_world = pose @ points_homogeneous.T
        return points_world[:3].T
    
    print(f"Number of keypoints with zero depth: {depth_zero_count}")
    print(f"Number of valid keypoints: {len(points)}")
    print("total keypoints", len(kps))
    
    

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
    # Normalize depth for visualization
    depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
    
    log_image(entity=title, image=depth_vis)

def visualize_matches(img0, img1, kpts0, kpts1, matches, color=(0, 255, 0), thickness=2, radius=6,  add_text = False):
    """
    Visualizes keypoint matches between two images.

    Args:
        img0: The first image (torch 1,3,H,W) or (numpy H,W,3).
        img1: The second image (torch 1,3,H,W) or (numpy H,W,3).
        kpts0: Keypoints in the first image (torch.Tensor, Nx2).
        kpts1: Keypoints in the second image (torch.Tensor, Nx2).
        matches: A tensor of shape (M, 2) where each row contains indices (i, j) 
                indicating that kpts0[i] matches kpts1[j].
        color:  Color of the lines and circles (B, G, R). Default: Green.
        thickness: Thickness of the connecting lines.
        radius: Radius of the keypoint circles.
        add_text : to add the text or not.

    Returns:
        output_img:  A combined image with matches visualized.
    """
    if isinstance(img0, torch.Tensor):
        img0 = img0.squeeze(0).permute(1, 2, 0).cpu().numpy()
    if isinstance(img1, torch.Tensor):
        img1 = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Convert to uint8
    img0 = (img0 ).astype(np.uint8)
    img1 = (img1 ).astype(np.uint8)

    # Convert to RGB
    img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    # Create a blank canvas for the output image
    output_img = np.hstack((img0, img1))

    # Draw keypoints and matches
    for i in range(matches.shape[0]):
        # Every 40th match is drawn
        # if i % 3 != 0:
        #     continue
        pt0 = tuple(kpts0[matches[i, 0]].int().cpu().numpy())
        pt1 = tuple(kpts1[matches[i, 1]].int().cpu().numpy() + np.array([img0.shape[1], 0]))

        cv2.circle(output_img, pt0, radius, color, -1)
        cv2.circle(output_img, pt1, radius, color, -1)
        # cv2.line(output_img, pt0, pt1, color, thickness)

        
    return output_img

if __name__ == "__main__":
    online_poses_path = "results/metrics_20250428_144655_frame_to_frame/trajectory_online.txt"
    # Sample line = 0.99976625736828173 -0.0096848061004579621 -0.019329644554804667 0.053494451619213915 0.0097149148417431559 0.99995173688927541 0.0014643502787692246 -0.0071527894115715387 0.019314529697516222 -0.0016517938484523994 0.99981209260522852 0.028181244859041363
    #Read and convert to poses

    poses = []
    with open(online_poses_path, "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            numbers = list(map(float, line.strip().split()))
            assert len(numbers) == 12
            pose = np.array(numbers).reshape(3,4)
            T = np.eye(4)
            T[:3, :4]  = pose
            poses.append(T)

    poses = np.array(poses)
    print(poses.shape)
    print(poses[0])

    starting_frame = 30
    ending_frame = 55

    # Initialize configuration
    config = Config()

    # Create dataset
    dataset = dataset_factory(config)
    print(f"Dataset: {dataset.name}")
    print(f"Total frames: {dataset.num_frames}")

    # Initialize camera
    camera = PinholeCamera(config)
    depth_factor = 1/camera.depth_factor

    # Initialize ground truth if available
    groundtruth = groundtruth_factory(config.dataset_settings)
    if groundtruth:
        print("Ground truth available")
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()



    # Get a sample frame
    if dataset.isOk():
        frame_id = starting_frame
        img1 = dataset.getImageColor(frame_id)
        depth1 = dataset.getDepth(frame_id)
        
        # Display the frame
        if img1 is not None:
            print(f"Frame {frame_id} loaded, shape: {img1.shape}")
            display_frame(img1, f"Frame {frame_id}")
            
            # If depth is available, visualize it
            if depth1 is not None:
                display_depth(depth1, f"Depth for Frame {frame_id}")
        else:
            print(f"Failed to load frame {frame_id}")
    else:
        print("Dataset is not properly initialized")

    # Get a sample frame
    if dataset.isOk():
        frame_id = ending_frame
        img2 = dataset.getImageColor(frame_id)
        depth2 = dataset.getDepth(frame_id)
        
        # Display the frame
        if img2 is not None:
            print(f"Frame {frame_id} loaded, shape: {img2.shape}")
            display_frame(img2, f"Frame {frame_id}")
            
            # If depth is available, visualize it
            if depth2 is not None:
                display_depth(depth2, f"Depth for Frame {frame_id}")
        else:
            print(f"Failed to load frame {frame_id}")
    else:
        print("Dataset is not properly initialized")


    # Extract superpoint features and match frames using lightglue.
    # Create point clouds from the depth images
    # Apply online trajecory poses to the point clouds - consider only frame 30 and frame 50
    # Create point clouds for the matched keypoints
    # Visualize the point clouds

    camera = PinholeCamera(config)
    depth_factor = 1/camera.depth_factor #eg final value = 5000
    feature_tracker_config = FeatureTrackerConfigs.LIGHTGLUE

    pc1= depth2pointcloud_with_mask(depth1, img1, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=None, scale=depth_factor)
    pc1_points, pc1_colors = pc1.points, pc1.colors

    pc2= depth2pointcloud_with_mask(depth2, img2, camera.fx, camera.fy, camera.cx, camera.cy, max_depth=50000000, mask=None, scale=depth_factor)
    pc2_points, pc2_colors = pc2.points, pc2.colors


    # Initialize rerun for visualization
    if True:
        rerun_record_name = f"pyslam_{dataset.name}_{int(time.time())}"  # Add timestamp for uniqueness
        rr.init(rerun_record_name, spawn=True)
        
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        log_coordinate_axes(entity_path="world/Origin", pose=np.eye(4), scale=1)


    pose1 = poses[0]
    pose2 = poses[ending_frame-starting_frame]
    print("pose1", pose1)
    print("pose2", pose2)

    log_frame_pc(frame_id=starting_frame, entity_path="world/GT/scans/", points=pc1_points, colors=pc1_colors, pose = pose1, fraction = 1.0)
    log_frame_pc(frame_id=ending_frame, entity_path="world/GT/scans/", points=pc2_points, colors=pc2_colors, pose = pose2, fraction = 1.0)


    print("Depth 1 max and min", np.max(depth1), np.min(depth1))
    print("Depth 2 max and min", np.max(depth2), np.min(depth2))
    """ 
    Depth 1 max and min 15623 0
    Depth 2 max and min 9019 0

    Divide by 5000 to cover depth values 
    """
    depth1_scaled = depth1 / 1000
    depth2_scaled = depth2 / 1000
    # Print max and min values of pc1 and pc2
    print("Depth 1 scaled max and min", np.max(depth1_scaled), np.min(depth1_scaled))
    print("Depth 2 scaled max and min", np.max(depth2_scaled), np.min(depth2_scaled))

    # TODO: Create feature extractor and matcher classes - superpoint and lightglue


    num_features = 1000
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = SuperPoint(max_num_keypoints=num_features).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    import torch

    print("img1 shape", img1.shape) 
    print("img2 shape", img2.shape)
    print("img1 dtype", img1.dtype)
    print("img2 dtype", img2.dtype)
    img1_torch_HWC = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    img2_torch_HWC = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

    # Convert images to torch tensors

    img1_torch_HWC = img1_torch_HWC.to(device)
    img2_torch_HWC = img2_torch_HWC.to(device)

    ref_feat = extractor.extract(img1_torch_HWC.to(device))
    curr_feat = extractor.extract(img2_torch_HWC.to(device))
    ## Remove feature at whose location the depth is zero
    print("ref_feat keys: ", ref_feat.keys()) # ref_feat keys:  dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
    print("ref_feat_ descriptors: ", ref_feat["descriptors"].shape) 
    print("ref_feat keypoint_scores: ", ref_feat["keypoint_scores"].shape)
    print("ref_feat image_size: ", ref_feat["image_size"].shape) 
    print("ref_feat keys: ", ref_feat["keypoints"].shape)
    print("ref_feat image_size: ", ref_feat["image_size"]) 
    print("depth image shape: ", depth1_scaled.shape) 
    
    """
    ref_feat keys:  dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
    ref_feat_ descriptors:  torch.Size([1, 1000, 256])
    ref_feat keypoint_scores:  torch.Size([1, 1000])
    ref_feat image_size:  torch.Size([1, 2])
    ref_feat keys:  torch.Size([1, 1000, 2])
    ref_feat image_size:  tensor([[1280.,  720.]], device='cuda:0')
    ref_feat_kps shape:  (1000, 2)
    depth image shape:  (720, 1280)

    """
    ref_feat = filter_features_by_depth(ref_feat, depth1_scaled)
    curr_feat = filter_features_by_depth(curr_feat, depth2_scaled)
    

    matches01 = matcher({"image0": ref_feat, "image1": curr_feat})
    feats0, feats1, matches01 = [
        rbd(x) for x in [ref_feat, curr_feat, matches01]
    ]  # remove batch dimension

    kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
    m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]


    print("Number of keypoints in ref image: ", len(kpts0))
    print("Number of keypoints in curr image: ", len(kpts1))
    print("shape of the image1: ", img1_torch_HWC.shape)
    print("shape of the image2: ", img2_torch_HWC.shape)

    print("matches shape: ", matches.shape)




    print("matches01 keys: ", matches01.keys())
    print("feats0 keys: ", feats0.keys())
    print("kpts0 keys: ", kpts0.shape)
    print("m_kpts0 keys:", m_kpts0.shape)
    print("ref_img shape: ", img1.shape)

    """ 
    Number of keypoints in ref image:  1000
    Number of keypoints in curr image:  1000
    shape of the image1:  torch.Size([1, 3, 720, 1280])
    shape of the image2:  torch.Size([1, 3, 720, 1280])
    matches shape:  torch.Size([640, 2])
    matches01 keys:  dict_keys(['matches0', 'matches1', 'matching_scores0', 'matching_scores1', 'stop', 'matches', 'scores', 'prune0', 'prune1'])
    feats0 keys:  dict_keys(['keypoints', 'keypoint_scores', 'descriptors', 'image_size'])
    kpts0 keys:  torch.Size([1000, 2])
    m_kpts0 keys: torch.Size([640, 2])
    ref_img shape:  (720, 1280, 3)
    """



    # Visualize matches
    output_img = visualize_matches(img1, img2, kpts0, kpts1, matches, add_text=True)
    log_image(entity=f"Matches between Frame {starting_frame} and Frame {ending_frame}", image=output_img)
    

    # Unproject keypoints to 3D points
    kps0 = kpts0[matches[..., 0]].int().cpu().numpy()
    kps1 = kpts1[matches[..., 1]].int().cpu().numpy()
    print("kps0 shape", kps0.shape)
    print("kps1 shape", kps1.shape)
    # Unproject keypoints to 3D points
    points0 = unproject_kps(depth1_scaled, kps0, camera, pose1, transform_to_world=True)
    points1 = unproject_kps(depth2_scaled, kps1, camera, pose2, transform_to_world=True)
    print("points0 shape", points0.shape)
    print("points1 shape", points1.shape)

    # Visualize the 3D points
    log_random_pc2(entity= "kps matched in frame 1", points=points0, colors="green", radius=0.04)
    log_random_pc2(entity= "kps matched in frame 2", points=points1, colors="blue", radius=0.04)


    # Apply delaunay triangulation to the keypoints of img1, then draw the triangulation on img1
    img1_delaunay, tri = delaunay_image_kps(img1, kps0)
    log_image(entity=f"Delaunay Triangulation on Frame {starting_frame}", image=img1_delaunay)

    # Create a graph from the delaunay triangulation and for each edge compute its distance in 3D and as text the depth value on top of the delaunay_image copy and then log it again
    delaunay_graph = convert_delauany_to_networkx(tri)

    annotated_image = img1_delaunay.copy()

    ########################### FOR VISUALIZATION OF DELAUNAY AND LENGTHS ############################
    # For each edge in the graph, compute 3D distance using the already unprojected points
    counter = 0
    for edge in delaunay_graph.edges:
        # Only process for every nth edge
        if counter % 20 != 0:
            counter += 1
            continue
        counter += 1
        idx1, idx2 = edge
        point1_idx = matches[idx1, 0].item()
        point2_idx = matches[idx2, 0].item()
        
        # Skip if we don't have valid indices
        if idx1 >= len(kps0) or idx2 >= len(kps0):
            continue
        
        # Calculate 3D distance between the points
        edge_length_3d = np.linalg.norm(points0[idx1] - points0[idx2])
        
        # Calculate midpoint of the edge for text placement
        midpoint_2d = ((kps0[idx1][0] + kps0[idx2][0]) // 2, 
                    (kps0[idx1][1] + kps0[idx2][1]) // 2)
        
        # Draw the 3D distance as text at the midpoint
        cv2.putText(annotated_image, f"{edge_length_3d:.2f}m", 
                    midpoint_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw the edge on the annotated image in red
        pt1 = tuple(kps0[idx1])
        pt2 = tuple(kps0[idx2])
        cv2.line(annotated_image, pt1, pt2, (0, 0, 255), 1)


    # Log the annotated image
    log_image(entity="Delaunay Triangulation with 3D Distances", image=annotated_image)
    ########################### FOR VISUALIZATION OF DELAUNAY AND LENGTHS ############################

    # Store edge distances in the graph for later use
    nx.set_edge_attributes(delaunay_graph, 
                        {edge: {'distance_3d': np.linalg.norm(points0[edge[0]] - points0[edge[1]])}
                        for edge in delaunay_graph.edges if edge[0] < len(points0) and edge[1] < len(points0)})
    # Store edge distances in the graph but for the other image
    nx.set_edge_attributes(delaunay_graph, 
                        {edge: {'distance_3d_other': np.linalg.norm(points1[edge[0]] - points1[edge[1]])} 
                        for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
    
    # Store the differences in the edge lengths in the graph
    nx.set_edge_attributes(delaunay_graph, 
                        {edge: {'distance_diff': np.abs(delaunay_graph.edges[edge]['distance_3d'] - delaunay_graph.edges[edge]['distance_3d_other'])} 
                        for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
    
    # Store for the edge, store a tuple with distance moved by the kps from frame1 to frame2 [node1motion, node2motion]
    nx.set_edge_attributes(delaunay_graph,
                        {edge: {'node1motion': np.linalg.norm(points0[edge[0]] - points1[edge[0]]), 
                                'node2motion': np.linalg.norm(points0[edge[1]] - points1[edge[1]])} 
                        for edge in delaunay_graph.edges if edge[0] < len(points1) and edge[1] < len(points1)})
    
    # Store for each edge, R*theta where R is the avg_length of the edge in frame1 and frame2 and theta is the angle between the two edges in 3D
    # First estimate edge vectors and then compute the angle between them
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
            angle = np.arccos(np.clip(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)), -1.0, 1.0))

            # Store the angle in the graph
            delaunay_graph.edges[edge]['angle_change'] = angle
            # Store the average length of the edge in frame1 and frame2
            avg_length = (np.linalg.norm(vec1) + np.linalg.norm(vec2)) / 2
            delaunay_graph.edges[edge]['avg_length'] = avg_length
            # Store the R*theta value in the graph
            delaunay_graph.edges[edge]['R_theta'] = avg_length * angle
            
    

    ## log a histogram like image to rerun - of the edge param - distance_diff
    # Get the edge lengths
    edge_lengths = [delaunay_graph.edges[edge]['distance_diff'] for edge in delaunay_graph.edges if 'distance_diff' in delaunay_graph.edges[edge]]
    # Create histogram of edge distance differences
    hist_img = visualize_edge_distance_histogram(delaunay_graph)

    # Log the histogram image to rerun
    log_image(entity="Edge Length Differences Histogram", image=hist_img)

    # Print every thing about the graph
    print("Delaunay Graph Info:")
    print("Delaunay Graph Edges with Attributes:")
    # for u, v, data in delaunay_graph.edges(data=True):
    #     print(f"Edge ({u}, {v}): {data}")
    # print("Delaunay Graph Nodes with Attributes:")
    # for node, data in delaunay_graph.nodes(data=True):
    #     print(f"Node {node}: {data}")
    # Print the number of edges and nodes
    print("Number of edges in the graph:", delaunay_graph.number_of_edges())
    print("Number of nodes in the graph:", delaunay_graph.number_of_nodes())
    # Print the number of connected components
    print("Number of connected components in the graph:", nx.number_connected_components(delaunay_graph))
    # Print the largest connected component
    largest_cc = max(nx.connected_components(delaunay_graph), key=len)
    print("Largest connected component size:", len(largest_cc))

    modified_delaunay_graph = delaunay_graph.copy()

    dynamic_edge_image = img1_delaunay.copy()

    for edge in modified_delaunay_graph.edges:

        # # For all the edges with distance_diff > threshold, draw them in blue
        # if delaunay_graph.edges[edge]['distance_diff'] > 0.6:
        #     pt1 = tuple(kps0[edge[0]])
        #     pt2 = tuple(kps0[edge[1]])
        #     cv2.line(dynamic_edge_image, pt1, pt2, (255, 0, 0), 1)
        #     # Remove that edge from the graph
        #     delaunay_graph.remove_edge(edge[0], edge[1])

        # elif max(delaunay_graph.edges[edge]['node1motion'], delaunay_graph.edges[edge]['node2motion']) > 0.35 and min(delaunay_graph.edges[edge]['node1motion'], delaunay_graph.edges[edge]['node2motion']) < 0.1:
        #     pt1 = tuple(kps0[edge[0]])
        #     pt2 = tuple(kps0[edge[1]])
        #     cv2.line(dynamic_edge_image, pt1, pt2, (255, 0, 0), 1)
        #     # Remove that edge from the graph
        #     delaunay_graph.remove_edge(edge[0], edge[1])


        if delaunay_graph.edges[edge]['R_theta'] > 0.25:
            pt1 = tuple(kps0[edge[0]])
            pt2 = tuple(kps0[edge[1]])
            cv2.line(dynamic_edge_image, pt1, pt2, (255, 0, 0), 1)
            # Remove that edge from the graph
            modified_delaunay_graph.remove_edge(edge[0], edge[1])
    # Log the dynamic edge image
    log_image(entity="Dynamic Edges in Delaunay Triangulation", image=dynamic_edge_image)

    print("Modified Delaunay Graph:", modified_delaunay_graph)
    # Visualizing connetected components
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
    # Log the connected components image
    log_image(entity="Connected Components in Delaunay Triangulation", image=connected_components_image)