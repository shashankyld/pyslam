# Class for dealing with two frames
# 1. Takes two frames as input,
# 2. Takes their images,
# 3. Has a param for number of features to be extracted, 
# 4. Extract features for both the images, 
# 5. Removes the points with the dynamic mask, if the number of points is less than half the number of intial # of features, extracts again with double the number of features
# 6. Matches the features 
# 7. Apply Delaunay triangulation on the reference frame (by default) or current frame matched points
# 8. Create a graph from delaunay triangulation
# 9. Unproject the points to the 3D space and 
# 10. Update properties of nodes and edges of the graph by comparing common edges in both frames - like edge length, angle, etc.
# 11. Modifies the graph by removing edges with large angle difference and large length difference and other relevant strategies
# 12. From the final graph, get connected components, for each component with more than N nodes, create a dynamic object except for the connected component with avg motion of nodes close to zero
# 13. Create a dynamic_objects instance and add the dynamic objects to it
# 14. Return the dynamic_objects instance 
# 15. If dynamic_objects instance is empty, set dynamic_objects_found mask to False, else set it to True
from dynamic_objects import DynamicObjects, DynamicObject
import torch
from thirdparty.LightGlue.lightglue import LightGlue, SuperPoint, DISK
from thirdparty.LightGlue.lightglue import viz2d
from utils_rerun import *
from thirdparty.LightGlue.lightglue.utils import rbd
from utils_delaunay import *
import networkx as nx
import sys
# Here 
# ref_id = -k_frames_away+1 usually
# cur_id = -1

class DelaunayDynamic:
    def __init__(self, num_features = 2000, effective_distance_threshold = 0.2, camera = None, slam = None):
        self.num_features = num_features
        self.effective_distance_threshold = effective_distance_threshold
        self.dynamic_objects = DynamicObjects()
        self.dynamic_mask = None
        self.camera = camera
        self.dynamic_objects_found = False
        self.graph = None
        self.recursion_depth = 0
        self.recursion_limit = 3
        self.slam = slam
        self.ref_frame = None
        self.cur_frame = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=self.num_features).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)
        # New attributes for feature storage
        self.ref_matched_data = {"keypoints": None, "descriptors": None, "keypoint_scores": None, "image_size": None}
        self.ref_id = None  # Stores the reference frame ID
        self.prune_delaunay_ref_frame_kps = False  # Flag to prune the reference frame
        self.prune_every_frame_kps_not_just_ref = False  # Flag to prune every frame, not just the reference frame
        self.max_gap_between_delaunay_ref_frame_and_cur_frame = 25  # Max gap between reference frame and current frame to update the reference frame
        
        # New parameter to control Delaunay triangulation on ref frame instead of cur frame
        self.use_ref_frame_for_delaunay = True  # Set to True to use ref frame for Delaunay
        self.delaunay_data = None  # Store the Delaunay triangulation data for reuse



    def _extract_features(self, ref_id, cur_id):
        """
        Extract features for the current frame and use stored matched features for the reference frame if available.
        """
        # For the first time setup self.ref_id 
        if self.ref_id is None:
            self.ref_id = ref_id
            if self.use_ref_frame_for_delaunay:
                self.delaunay_data = None
                print(f"First-time reference frame set (ID {ref_id}), initializing Delaunay data")
            
        # If reference ID changed, reset Delaunay data
        if self.ref_id != ref_id and self.use_ref_frame_for_delaunay:
            self.delaunay_data = None
            print(f"Reference frame changed from {self.ref_id} to {ref_id}, resetting Delaunay data")

        cur_frame = self.slam.map.get_frame(cur_id)
        dynamic_mask_cur = cur_frame.dynamic_mask
        self.cur_frame = cur_frame
        self.dynamic_mask = dynamic_mask_cur

        cur_torch_HWC = torch.from_numpy(cur_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cur_torch_HWC = cur_torch_HWC.to(self.device)

        # Get reference frame
        ref_frame = self.slam.map.get_frame(ref_id)
        self.ref_frame = ref_frame

        # Check if matched features for reference frame can be reused
        ref_feat = None
        
        # First, check if we have the same reference frame with cached features
        if self.ref_matched_data["keypoints"] is not None and self.ref_id == ref_id:
            print(f"Using precomputed matched features for reference frame {ref_id}")
            ref_feat = self.ref_matched_data
            try:
                print(f"Reusing stored matched features for reference frame {ref_id} with {ref_feat['keypoints'].shape[1]} keypoints")
            except Exception as e:
                print(f"Error reusing stored matched features for reference frame {ref_id}: {e}")
                ref_feat = None
        
        # Then, check if the reference frame has pre-calculated features
        elif hasattr(ref_frame, 'delaunay_matched_feat') and ref_frame.delaunay_matched_feat is not None:
            print(f"Using pre-stored matched features from reference frame {ref_id}")
            ref_feat = ref_frame.delaunay_matched_feat
            self.ref_matched_data = ref_feat
            self.ref_id = ref_id
            try:
                print(f"Using pre-stored features from reference frame {ref_id} with {ref_feat['keypoints'].shape[1]} keypoints")
            except Exception as e:
                print(f"Error using pre-stored features from reference frame {ref_id}: {e}")
                ref_feat = None
        
        # If no cached features are available, extract new ones
        if ref_feat is None:
            print(f"Extracting new features for reference frame {ref_id}")
            dynamic_mask_ref = ref_frame.dynamic_mask
            ref_torch_HWC = torch.from_numpy(ref_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            ref_torch_HWC = ref_torch_HWC.to(self.device)
            ref_feat = self.extractor.extract(ref_torch_HWC.to(self.device))

            ref_depth = ref_frame.depth_img
            ref_feat = self._filter_features_by_depth(ref_feat, ref_depth)
            ref_feat = self._filter_features_by_mask(ref_feat, dynamic_mask_ref)
            self.ref_id = ref_id
            print(f"Extracted new features for reference frame {ref_id} with {ref_feat['keypoints'].shape[1]} keypoints")

        # Extract features for current frame
        cur_feat = self.extractor.extract(cur_torch_HWC.to(self.device))
        cur_depth = cur_frame.depth_img
        cur_feat = self._filter_features_by_depth(cur_feat, cur_depth)
        cur_feat = self._filter_features_by_mask(cur_feat, dynamic_mask_cur)

        # Handle low keypoint count with recursion
        self.recursion_depth += 1
        if (ref_feat["keypoints"].shape[1] < self.num_features // 2 or 
            cur_feat["keypoints"].shape[1] < self.num_features // 2) and self.recursion_depth < self.recursion_limit:
            print("Low keypoint count, doubling num_features and re-extracting")
            self.extractor.conf.max_num_keypoints = self.num_features * 2
            
            # Only re-extract reference features if we don't have pre-stored ones
            if not (hasattr(ref_frame, 'delaunay_matched_feat') and ref_frame.delaunay_matched_feat is not None):
                ref_torch_HWC = torch.from_numpy(ref_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                ref_torch_HWC = ref_torch_HWC.to(self.device)
                ref_feat = self.extractor.extract(ref_torch_HWC.to(self.device))
                ref_feat = self._filter_features_by_depth(ref_feat, ref_frame.depth_img)
                ref_feat = self._filter_features_by_mask(ref_feat, ref_frame.dynamic_mask)
                
            cur_feat = self.extractor.extract(cur_torch_HWC.to(self.device))
            cur_feat = self._filter_features_by_depth(cur_feat, cur_depth)
            cur_feat = self._filter_features_by_mask(cur_feat, dynamic_mask_cur)

        print(f"Feature extraction recursion depth: {self.recursion_depth}")
        print(f"Extracting features with num_features: {self.extractor.conf.max_num_keypoints}")
        print(f"ref_feat keypoints: {ref_feat['keypoints'].shape}")
        print(f"cur_feat keypoints: {cur_feat['keypoints'].shape}")

        return ref_feat, cur_feat



    def _match_features(self, ref_feat, cur_feat):
        """
        Match features and store matched data for the reference frame in extraction-like format.
        """
        matches01 = self.matcher({"image0": ref_feat, "image1": cur_feat})
        feats0, feats1, matches01 = [rbd(x) for x in [ref_feat, cur_feat, matches01]]

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        # Store matched data for reference frame in extraction-like format
        if self.prune_delaunay_ref_frame_kps:
            self.ref_matched_data = {
                "keypoints": torch.unsqueeze(m_kpts0, 0),  # Shape: [1, N, 2]
                "descriptors": torch.unsqueeze(feats0["descriptors"][matches[..., 0]], 0),  # Shape: [1, N, D]
                "keypoint_scores": torch.unsqueeze(feats0["keypoint_scores"][matches[..., 0]], 0),  # Shape: [1, N]
                "image_size": ref_feat["image_size"],  # Shape: [1, 2]
                "keypoints_numpy": m_kpts0.int().cpu().numpy(),  # Numpy version for triangulation
                "triangulation": None,  # Will be filled in _apply_delaunay_triangulation_and_get_graph
                "delaunay_image": None  # Will be filled in _apply_delaunay_triangulation_and_get_graph
            }

            self.ref_frame.delaunay_matched_feat = self.ref_matched_data
            
            if self.prune_every_frame_kps_not_just_ref:
                self.cur_frame.delaunay_matched_feat = {
                    "keypoints": torch.unsqueeze(m_kpts1, 0),  # Shape: [1, N, 2]
                    "descriptors": torch.unsqueeze(feats1["descriptors"][matches[..., 1]], 0),  # Shape: [1, N, D]
                    "keypoint_scores": torch.unsqueeze(feats1["keypoint_scores"][matches[..., 1]], 0),  # Shape: [1, N]
                    "image_size": cur_feat["image_size"],  # Shape: [1, 2]
                    "keypoints_numpy": m_kpts1.int().cpu().numpy(),  # Numpy version for triangulation
                    "triangulation": None,  # For potential future use
                    "delaunay_image": None  # For potential future use
                }
                
            # If we're using a new reference frame, reset the delaunay data
            if self.ref_id != self.ref_frame.id and self.use_ref_frame_for_delaunay:
                self.delaunay_data = None
                print(f"New reference frame (ID {self.ref_frame.id}), resetting Delaunay data")
        else:
            # Just save all the ref_feat and cur_feat
            self.ref_matched_data = {
                "keypoints": torch.unsqueeze(kpts0, 0),  # Shape: [1, N, 2]
                "descriptors": torch.unsqueeze(feats0["descriptors"], 0),  # Shape: [1, N, D]
                "keypoint_scores": torch.unsqueeze(feats0["keypoint_scores"], 0),  # Shape: [1, N]
                "image_size": ref_feat["image_size"],  # Shape: [1, 2]
                "keypoints_numpy": kpts0.int().cpu().numpy(),  # Numpy version for triangulation
                "triangulation": None,  # Will be filled in _apply_delaunay_triangulation_and_get_graph
                "delaunay_image": None  # Will be filled in _apply_delaunay_triangulation_and_get_graph
            }





        print(f"Number of keypoints in ref image: {len(kpts0)}")
        print(f"Number of keypoints in curr image: {len(kpts1)}")
        print(f"Number of matches: {matches.shape[0]}")
        print(f"Stored {self.ref_matched_data['keypoints'].shape[1]} matched keypoints for reference frame")

        # Visualize matches
        output_img = self._visualize_matches(self.ref_frame.img, self.cur_frame.img, kpts0, kpts1, matches, add_text=True)
        log_image(entity=f"Matches between Frame curr and Frame k_frames_away", image=output_img)

        return m_kpts0, m_kpts1, matches

    def _extract_and_match_features(self, ref_id, cur_id):
        """
        Extract and match features between the reference and current frames.
        """
        # Extract features
        ref_feat, cur_feat = self._extract_features(ref_id, cur_id)

        # Match features
        m_kpts0, m_kpts1, matches = self._match_features(ref_feat, cur_feat)
        print(m_kpts0.shape, m_kpts1.shape, matches.shape)
        return ref_feat, cur_feat, m_kpts0, m_kpts1, matches

    

    def _apply_delaunay_triangulation_and_get_graph(self, ref_id, cur_id):
        """
        Apply Delaunay triangulation on the matched keypoints and get the graph.
        """
        ref_frame = self.slam.map.get_frame(ref_id)
        cur_frame = self.slam.map.get_frame(cur_id)
        # Extract and match features
        ref_feat, cur_feat, m_kpts0, m_kpts1, matches = self._extract_and_match_features(ref_id, cur_id)
        # 1. Prepare keypoints for Delaunay triangulation
        m_kpts0_np = m_kpts0.int().cpu().numpy()
        m_kpts1_np = m_kpts1.int().cpu().numpy()

        print("m_kpts0_np shape: ", m_kpts0_np.shape)
        print("m_kpts1_np shape: ", m_kpts1_np.shape)
        
        # 2. Apply Delaunay triangulation to the matched keypoints
        # Check if we need to compute a new triangulation
        compute_new_triangulation = False
        
        # Compute new triangulation if:
        # 1. This is a new reference frame (ref_id != self.ref_id) or the first time
        # 2. We don't have stored triangulation data
        # 3. We're using the old behavior (use_ref_frame_for_delaunay = False)
        if self.ref_id != ref_id or self.delaunay_data is None or not self.use_ref_frame_for_delaunay:
            compute_new_triangulation = True
        
        if compute_new_triangulation:
            if self.use_ref_frame_for_delaunay:
                # Apply triangulation on reference frame
                img_delaunay, tri = delaunay_image_kps(ref_frame.img, m_kpts0_np)
                log_entity = "world/matched_kps/ref_frame/delaunay_triangulation"
                print("Computing new Delaunay triangulation on reference frame")
                
                # Store triangulation data for reuse
                if self.ref_id == ref_id:
                    self.delaunay_data = {
                        "triangulation": tri,
                        "image": img_delaunay
                    }
            else:
                # Original behavior: apply triangulation on current frame
                img_delaunay, tri = delaunay_image_kps(cur_frame.img, m_kpts1_np)
                log_entity = "world/matched_kps/cur_frame/delaunay_triangulation"
                print("Computing new Delaunay triangulation on current frame")
                
            if True:
                log_image(log_entity, img_delaunay)
        else:
            # Reuse stored triangulation
            print("Reusing stored Delaunay triangulation from reference frame")
            tri = self.delaunay_data["triangulation"]
            img_delaunay = self.delaunay_data["image"]
            
            # Log the reused image
            if True:
                log_image("world/matched_kps/ref_frame/reused_delaunay_triangulation", img_delaunay)
        
        # 3. Create a graph from the Delaunay triangulation
        delaunay_graph = convert_delauany_to_networkx(tri)
        
        # When using reference frame for Delaunay, we need to ensure we only keep
        # edges connecting keypoints that are common between both frames
        if self.use_ref_frame_for_delaunay:
            # Create a copy of the graph to modify
            filtered_graph = delaunay_graph.copy()
            
            # Get total number of matched keypoints
            num_matched_keypoints = m_kpts0_np.shape[0]
            
            # Remove edges where one or both endpoints are not in the common set
            edges_to_remove = []
            for edge in filtered_graph.edges():
                if (edge[0] >= num_matched_keypoints or edge[1] >= num_matched_keypoints):
                    edges_to_remove.append(edge)
            
            filtered_graph.remove_edges_from(edges_to_remove)
            print(f"Removed {len(edges_to_remove)} edges connecting non-common keypoints")
            
            # Use filtered graph instead of original
            delaunay_graph = filtered_graph
            
            # Visualize the filtered graph (only for debugging)
            if True:
                filtered_img = ref_frame.img.copy()
                for edge in delaunay_graph.edges():
                    if edge[0] < m_kpts0_np.shape[0] and edge[1] < m_kpts0_np.shape[0]:
                        pt1 = tuple(m_kpts0_np[edge[0]])
                        pt2 = tuple(m_kpts0_np[edge[1]])
                        cv2.line(filtered_img, pt1, pt2, (0, 255, 0), 1)
                
                log_image("world/matched_kps/ref_frame/filtered_delaunay", filtered_img)
        
        # Store triangulation in ref_matched_data when using reference frame
        if self.use_ref_frame_for_delaunay and compute_new_triangulation:
            self.ref_matched_data["triangulation"] = tri
            self.ref_matched_data["delaunay_image"] = img_delaunay
            
        return ref_feat, cur_feat, m_kpts0_np, m_kpts1_np, matches, delaunay_graph, img_delaunay
        

    
    def _update_graph_properties(self, ref_id, cur_id):
        """
        Update the graph properties based on the matched keypoints and their 3D coordinates.
        """
        ref_feat, cur_feat, m_kpts0_np, m_kpts1_np, matches,delaunay_graph, img_delaunay = self._apply_delaunay_triangulation_and_get_graph(ref_id, cur_id)
        ref_frame = self.slam.map.get_frame(ref_id)
        cur_frame = self.slam.map.get_frame(cur_id)
        ref_depth = ref_frame.depth_img 
        cur_depth = cur_frame.depth_img
        ref_frame_Tcw = ref_frame.pose
        cur_frame_Tcw = cur_frame.pose
        ref_frame_Twc = np.linalg.inv(ref_frame_Tcw)
        cur_Twc = np.linalg.inv(cur_frame_Tcw)
        # 4. Unproject keypoints to 3D points
        points0, z_1 = self._unproject_kps(ref_depth, 
                                m_kpts0_np, self.camera, ref_frame_Twc, transform_to_world=True)
        points1, z_2 = self._unproject_kps(cur_depth, 
                                m_kpts1_np, self.camera, cur_Twc, transform_to_world=True)
        
        print("points0 shape: ", points0.shape)
        print("points1 shape: ", points1.shape)
        print("points0 depth zero:", z_1 )
        print("points1 depth zero:", z_2 )

        # Store 3D points in ref_matched_data
        self.ref_matched_data["points3d"] = points0  # Shape: [N, 3]


        # Visualize the 3D points
        log_random_pc2(entity= "world/slam/kps matched in k_frames_away", points=points0, colors="green", radius=0.04)
        log_random_pc2(entity= "world/slam/kps matched in cur_frame", points=points1, colors="blue", radius=0.04)

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
            {edge: {'effective_distance': np.sqrt((delaunay_graph.edges[edge]['distance_3d'] * np.cos(delaunay_graph.edges[edge]['angle_change']) - 
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

        # Detect dynamic edges and remove them from the graph
        modified_delaunay_graph = delaunay_graph.copy()
        dynamic_edge_image = img_delaunay.copy()

        effective_distance_threshold = self.effective_distance_threshold

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
        connected_components = self._get_connected_components(modified_delaunay_graph)
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

        update_ref_frame_flag = self.update_ref_frame_flag(ref_id, cur_id)        

        return update_ref_frame_flag

    def update_ref_frame_flag(self, ref_id, cur_id):
        """
        Update the reference frame ID and reset the recursion depth.
        if gap between ref_id and cur_id is more than 25, if object is found - dynamic connected component with more than 5 nodes and clear motion is found.
        """
        if abs(cur_id - ref_id) > self.max_gap_between_delaunay_ref_frame_and_cur_frame:
            # Reset the recursion depth
            self.recursion_depth = 0
            print("Resetting recursion depth to 0, and shifting the delaunay_ref_frame due to large gap between ref_id and cur_id")
            
            # Reset Delaunay data when reference frame changes
            if self.use_ref_frame_for_delaunay:
                self.delaunay_data = None
                print("Resetting Delaunay data due to reference frame change")
            
            return True

    def _filter_features_by_depth(self, ref_feat, depth_scaled, max_depth=6):
        """
        Filters keypoints from reference features where depth is zero and depth is less than Xm.
        
        Args:
            ref_feat: Dictionary containing feature data (keypoints, keypoint_scores, descriptors)
            depth_scaled: Scaled depth map (numpy array)
        
        Returns:
            ref_feat: Filtered reference features
        """
        max_depth = 6 #m
        ref_feat_kps = ref_feat["keypoints"][0].int().cpu().numpy()
        valid_indices = [
            i for i, kp in enumerate(ref_feat_kps)
            if depth_scaled[int(kp[1]), int(kp[0])] != 0 and depth_scaled[int(kp[1]), int(kp[0])] < max_depth
        ]
        
        if len(valid_indices) < len(ref_feat_kps):
            print(f"Removing {len(ref_feat_kps) - len(valid_indices)} keypoints with zero depth or depth > {max_depth}m")
        ref_feat["keypoints"] = ref_feat["keypoints"][:, valid_indices]
        ref_feat["keypoint_scores"] = ref_feat["keypoint_scores"][:, valid_indices]
        ref_feat["descriptors"] = ref_feat["descriptors"][:, valid_indices]

        
    
        return ref_feat
    
    def _filter_features_by_mask(self, ref_feat, mask):
        """
        Filters keypoints from reference features where the mask is zero.
        
        Args:
            ref_feat: Dictionary containing feature data (keypoints, keypoint_scores, descriptors)
            mask: Mask image (numpy array)
        
        Returns:
            ref_feat: Filtered reference features
        """
        ref_feat_kps = ref_feat["keypoints"][0].int().cpu().numpy()
        valid_indices = [
            i for i, kp in enumerate(ref_feat_kps)
            if mask[int(kp[1]), int(kp[0])] == 0
        ]
        
        if len(valid_indices) < len(ref_feat_kps):
            print(f"Removing {len(ref_feat_kps) - len(valid_indices)} keypoints with zero mask")
        ref_feat["keypoints"] = ref_feat["keypoints"][:, valid_indices]
        ref_feat["keypoint_scores"] = ref_feat["keypoint_scores"][:, valid_indices]
        ref_feat["descriptors"] = ref_feat["descriptors"][:, valid_indices]
        

        return ref_feat
    
    # def _filter_features_for_being_outlier_compared_to_surrounding_pc(self, ref_feat, point_cloud, threshold=0.1):
        
    def _visualize_matches(self, img0, img1, kpts0, kpts1, matches, color=(0, 255, 0), thickness=2, radius=6,  add_text = False):
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

    def _unproject_kps(self, depth_img, kps, camera, pose, transform_to_world=False):
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
        visited_xy = set()
        for kp in kps:
            x, y = int(kp[0]), int(kp[1])
            if (x, y) in visited_xy:
                print(f"Duplicate keypoint at ({x}, {y})")
                continue
            visited_xy.add((x, y))

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
            return points_world[:3].T, depth_zero_count
        
        print(f"Number of keypoints with zero depth: {depth_zero_count}")
        print(f"Number of valid keypoints: {len(points)}")
        print("total keypoints", len(kps))
        
        

        return points, depth_zero_count
    

    def _get_connected_components(self,G):
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