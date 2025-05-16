# Class for dealing with two frames
# 1. Takes two frames as input,
# 2. Takes their images,
# 3. Has a param for number of features to be extracted, 
# 4. Extract features for both the images, 
# 5. Removes the points with the dynamic mask, if the number of points is less than half the number of intial # of features, extracts again with double the number of features
# 6. Matches the features 
# 7. Apply Delaunay triangulation on the current frame matached points
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

class DelaunayDynamic:
    def __init__(self, num_features = 1000, effective_distance_threshold = 0.1):
        self.num_features = num_features
        self.effective_distance_threshold = effective_distance_threshold
        self.dynamic_objects = DynamicObjects()
        self.dynamic_mask = None

        self.dynamic_objects_found = False
        self.graph = None
        self.recursion_depth = 0
        self.recursion_limit = 3

        self.ref_frame = None
        self.cur_frame = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = SuperPoint(max_num_keypoints=num_features).eval().to(self.device)
        self.matcher = LightGlue(features="superpoint").eval().to(self.device)

    def _extract_features(self, ref_frame, cur_frame, dynamic_mask): 
        """
        Extract features from the reference and current frames.
        """
        # Set known attributes
        self.ref_frame = ref_frame
        self.cur_frame = cur_frame
        self.dynamic_mask = dynamic_mask

        ref_torch_HWC = torch.from_numpy(ref_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        cur_torch_HWC = torch.from_numpy(cur_frame.img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        ref_torch_HWC = ref_torch_HWC.to(self.device)
        cur_torch_HWC = cur_torch_HWC.to(self.device)
        
        ref_feat = self.extractor.extract(ref_torch_HWC.to(self.device))
        cur_feat = self.extractor.extract(cur_torch_HWC.to(self.device))

        print("ref_feat keypoints before filtering with depth: ", ref_feat["keypoints"].shape)
        print("cur_feat keypoints before filtering with depth: ", cur_feat["keypoints"].shape)

        ref_depth = ref_frame.depth_img
        cur_depth = cur_frame.depth_img

        # Print max and min of ref_depth and curr_depth
        print("ref_depth max: ", ref_depth.max())   
        print("ref_depth min: ", ref_depth.min())
        print("curr_depth max: ", cur_depth.max())
        print("curr_depth min: ", cur_depth.min())

        ref_feat = self._filter_features_by_depth(ref_feat, ref_depth)
        cur_feat = self._filter_features_by_depth(cur_feat, cur_depth)

        # Print # of keypoints after filtering
        print("ref_feat keypoints after filtering depth: ", ref_feat["keypoints"].shape)
        print("curr_feat keypoints after filtering depth: ", cur_feat["keypoints"].shape)

        # Remove features that are masked out
        ref_feat = self._filter_features_by_mask(ref_feat, dynamic_mask)
        cur_feat = self._filter_features_by_mask(cur_feat, dynamic_mask)
        print("ref_feat keypoints after filtering mask: ", ref_feat["keypoints"].shape)
        print("curr_feat keypoints after filtering mask: ", cur_feat["keypoints"].shape)

        # Increment the recursion depth
        self.recursion_depth += 1

        # Check if the number of keypoints is less than half the number of features
        if ref_feat["keypoints"].shape[1] < self.num_features // 2 and self.recursion_depth < self.recursion_limit:
            print("Number of keypoints is less than half the number of features, extracting again with double the number of features")
            self.num_features *= 2
            ref_feat = self.extractor.extract(ref_torch_HWC.to(self.device))
            cur_feat = self.extractor.extract(cur_torch_HWC.to(self.device))
            ref_feat = self._filter_features_by_depth(ref_feat, ref_depth)
            cur_feat = self._filter_features_by_depth(cur_feat, cur_depth)
            ref_feat = self._filter_features_by_mask(ref_feat, dynamic_mask)
            cur_feat = self._filter_features_by_mask(cur_feat, dynamic_mask)
            
        # Print the number of keypoints after filtering
        print("Feature extraction recursion depth: ", self.recursion_depth)
        print("Extracting features with num_features: ", self.num_features)
        print("ref_feat keypoints after filtering with: ", ref_feat["keypoints"].shape)
        print("curr_feat keypoints after filtering: ", cur_feat["keypoints"].shape)

        return ref_feat, cur_feat

    def _match_features(self, ref_feat, cur_feat):
        """
        Match features between the reference and current frames.
        """
        matches01 = self.matcher({"image0": ref_feat, "image1": cur_feat})
        feats0, feats1, matches01 = [
            rbd(x) for x in [ref_feat, cur_feat, matches01]
        ]  # remove batch dimension

        kpts0, kpts1, matches = feats0["keypoints"], feats1["keypoints"], matches01["matches"]
        m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

        print("Number of keypoints in ref image: ", len(kpts0))
        print("Number of keypoints in curr image: ", len(kpts1))
        print("matches shape: ", matches.shape)

        # Visualize matches
        output_img = self._visualize_matches(self.ref_frame.img,self.cur_frame.img,  kpts0, kpts1, matches, add_text=True)
        log_image(entity=f"Matches between Frame curr and Frame k_frames_away", image=output_img)

        return m_kpts0, m_kpts1, matches

    def _extract_and_match_features(self, ref_frame, cur_frame, dynamic_mask):
        """
        Extract and match features between the reference and current frames.
        """
        # Extract features
        ref_feat, cur_feat = self._extract_features(ref_frame, cur_frame, dynamic_mask)

        # Match features
        m_kpts0, m_kpts1, matches = self._match_features(ref_feat, cur_feat)
        print(m_kpts0.shape, m_kpts1.shape, matches.shape)
        return ref_feat, cur_feat, m_kpts0, m_kpts1, matches


    def _apply_delaunay_triangulation_and_get_graph(self, cur_frame, m_kpts0, m_kpts1):
        """
        Apply Delaunay triangulation on the matched keypoints and get the graph.
        """
        # 1. Prepare keypoints for Delaunay triangulation
        m_kpts0_np = m_kpts0.int().cpu().numpy()
        m_kpts1_np = m_kpts1.int().cpu().numpy()

        print("m_kpts0_np shape: ", m_kpts0_np.shape)
        print("m_kpts1_np shape: ", m_kpts1_np.shape)
        
        # 2. Apply Delaunay triangulation to the matched keypoints
        img_delaunay, tri = delaunay_image_kps(cur_frame.img, m_kpts1_np)
        if True:
            log_image("world/matched_kps/cur_frame/delaunay_triangulation", img_delaunay)
        
        # 3. Create a graph from the Delaunay triangulation
        delaunay_graph = convert_delauany_to_networkx(tri)
        return delaunay_graph


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
