import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Make sure thirdparty/sam2 is in sys.path
SLAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))

# Import SAM2 video predictor
from sam2.build_sam import build_sam2_video_predictor

class SAM2KeyframeProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tmp_folder = os.path.join(SLAM_ROOT, "tmp_sam2_frames")
        os.makedirs(self.tmp_folder, exist_ok=True)
        
        # Initialize SAM2 model
        # Change to SAM2 directory to make relative paths work with Hydra
        original_dir = os.getcwd()
        os.chdir(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))
        
        try:
            # Load SAM2 predictor
            sam2_checkpoint = os.path.join("checkpoints", "sam2.1_hiera_large.pt")
            model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
            
            if not os.path.exists(sam2_checkpoint):
                print(f"Warning: SAM2 checkpoint not found at {sam2_checkpoint}")
                print("SAM2 integration will not work. Please download the checkpoint.")
                self.predictor = None
            else:
                print("Loading SAM2 predictor...")
                self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=self.device)
                print("SAM2 predictor loaded successfully!")
        finally:
            # Change back to original directory
            os.chdir(original_dir)
            
        self.inference_state = None
        self.obj_id = 1  # Using a single object ID for simplicity

    def clear_tmp_folder(self):
        """Clear temporary folder for frame images"""
        # Clear existing files in tmp folder
        for file in os.listdir(self.tmp_folder):
            file_path = os.path.join(self.tmp_folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

    def init_inference_state(self):
        """Initialize SAM2 inference state with images in tmp folder"""
        if self.predictor is None:
            return None
            
        # Check if there are images in the tmp folder
        if not any(f.endswith('.jpg') for f in os.listdir(self.tmp_folder)):
            print("No images found in temporary folder. Cannot initialize inference state.")
            return None
            
        # Initialize or reset inference state
        if self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
            
        self.inference_state = self.predictor.init_state(video_path=self.tmp_folder)
        return self.inference_state
    
    def save_keyframe_images(self, keyframes):
        """Save all keyframe images to temporary folder"""
        for i, kf in enumerate(keyframes):
            if kf.img is not None:
                # Convert to RGB for PIL compatibility
                rgb_img = cv2.cvtColor(kf.img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_img)
                
                # Save image with frame index (as expected by SAM2)
                frame_path = os.path.join(self.tmp_folder, f"{i:05d}.jpg")
                pil_img.save(frame_path)
                print(f"Saved keyframe {kf.id} as {frame_path}")
    
    def generate_prompts_from_mask(self, mask, num_points=50):
        """Generate SAM2 prompts (points) from a binary mask"""
        if mask is None:
            return None, None
            
        # Find white pixels in the mask (dynamic objects)
        white_pixels = np.where(mask > 0) ## BLACK PIXELS is mask == 0
        if len(white_pixels[0]) == 0:
            # No dynamic objects detected
            return None, None
            
        # Randomly sample points from the mask
        if len(white_pixels[0]) <= num_points:
            # Use all white pixels if fewer than num_points
            sample_indices = range(len(white_pixels[0]))
        else:
            # Randomly sample num_points from white pixels
            sample_indices = np.random.choice(len(white_pixels[0]), num_points, replace=False)
            
        # Create points array with x,y coordinates (SAM2 expects x,y format)
        points = np.column_stack([white_pixels[1][sample_indices], white_pixels[0][sample_indices]])
        # All points are positive indicators (label 1)
        labels = np.ones(len(points), dtype=np.int32)
        
        return points, labels
    
    def process_keyframes(self, keyframes):
        """Process all keyframes with SAM2 video segmentation"""
        if self.predictor is None:
            print("SAM2 predictor not initialized. Skipping keyframe processing.")
            return keyframes
        
        if len(keyframes) == 0:
            print("No keyframes to process. Skipping SAM2 processing.")
            return keyframes
            
        # Clear temporary folder
        self.clear_tmp_folder()
        
        # Save all keyframe images to tmp folder
        self.save_keyframe_images(keyframes)
        
        # Check if we have any images to process
        if not any(f.endswith('.jpg') for f in os.listdir(self.tmp_folder)):
            print("No keyframe images saved. Skipping SAM2 processing.")
            return keyframes
            
        # Initialize inference state with the saved images
        self.init_inference_state()
        if self.inference_state is None:
            print("Failed to initialize SAM2 inference state. Skipping processing.")
            return keyframes
            
        # Add prompts to the inference state from each keyframe's dynamic mask
        has_prompts = False
        for i, kf in enumerate(keyframes):
            # Check if keyframe has a dynamic mask from MaskRCNN
            if kf.dynamic_mask is not None:
                print("Valid dynamic mask found in keyframe.")
                # Generate points from the dynamic mask
                points, labels = self.generate_prompts_from_mask(kf.dynamic_mask)
                
                # If we found valid points, add them to SAM2
                if points is not None and len(points) > 0:
                    # Convert points and labels to torch tensors
                    points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)  # [1, N, 2]
                    labels_tensor = torch.tensor(labels, dtype=torch.int32).unsqueeze(0)  # [1, N]
                    
                    # Store the prompts in the keyframe for future reference
                    kf.add_sam2_prompt(self.obj_id, points, labels)
                    
                    try:
                        # Add the prompts to SAM2 inference state
                        print(f"Adding {len(points)} prompts to keyframe {kf.id} at frame index {i}")
                        self.predictor.add_new_points_or_box(
                            inference_state=self.inference_state,
                            frame_idx=i,
                            obj_id=self.obj_id,
                            points=points_tensor,
                            labels=labels_tensor
                        )
                        has_prompts = True
                    except Exception as e:
                        print(f"Error adding prompts to SAM2: {e}")
        
        # Only propagate if we have added at least one prompt
        if not has_prompts:
            print("No valid prompts found in any keyframe. Skipping propagation.")
            return keyframes
        
        # Propagate segmentation masks across all frames
        try:
            print("Propagating segmentation masks across keyframes...")
            video_segments = {}
            for out_obj_ids in self.predictor.propagate_in_video(self.inference_state):
                # Check if the output is a tuple with frame_idx (newer SAM2 API)
                if isinstance(out_obj_ids, tuple) and len(out_obj_ids) >= 3:
                    out_frame_idx, out_obj_ids, out_mask_logits = out_obj_ids
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
        except Exception as e:
            print(f"Error during mask propagation: {e}")
            return keyframes
        
        # Update each keyframe with its refined segmentation mask
        for i, kf in enumerate(keyframes):
            if i in video_segments and self.obj_id in video_segments[i]:
                refined_mask = video_segments[i][self.obj_id]
                
                # Convert to uint8 mask (required by KeyFrame class)
                binary_mask = refined_mask.astype(np.uint8) * 255
                print("Shape of the binary mask:", binary_mask.shape)
                
                # Store the refined mask in the keyframe
                kf.set_sam2_prediction(self.obj_id, binary_mask)
                
                # Also update the dynamic_mask for consistency
                kf.dynamic_mask = binary_mask[0]
                
                print(f"Updated dynamic mask for keyframe {kf.id}")
                
                # Visualize the mask (for debugging)
                cv2.imshow(f"SAM2 Refined Mask - KF {kf.id}", binary_mask[0])
                cv2.waitKey(1)
        
        return keyframes