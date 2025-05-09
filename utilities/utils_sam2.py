import os
import shutil
import tempfile
import numpy as np
import cv2
from PIL import Image
import torch

def create_temp_symlink_folder(dataset_images_path_dir, start_idx, end_idx, temp_folder_name="sam2_temp_symlinks"):
    """
    Creates a clean temporary folder with symlinks to selected images from a dataset directory.

    Args:
        dataset_images_path_dir (str): Path to the folder containing RGB images.
        start_idx (int): Start frame index (inclusive).
        end_idx (int): End frame index (exclusive).
        temp_folder_name (str): Name for the temp folder (under /tmp). Default: 'sam2_temp_symlinks'.

    Returns:
        str: Path to the temporary folder with symlinks.
    """
    # Create sorted list of image filenames
    images_paths_ordered = sorted(os.listdir(dataset_images_path_dir))

    # Validate indices
    if start_idx < 0 or end_idx > len(images_paths_ordered):
        raise IndexError(f"Frame range [{start_idx}, {end_idx}) is out of bounds for {len(images_paths_ordered)} images.")

    # Create or clean up temp folder
    temp_root = os.path.join(tempfile.gettempdir(), temp_folder_name)
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root)

    # Create symlinks
    for fname in images_paths_ordered[start_idx:end_idx]: # Includes the start index, excludes the end index
        src_path = os.path.join(dataset_images_path_dir, fname)
        dst_path = os.path.join(temp_root, fname)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source image not found: {src_path}")
        os.symlink(os.path.abspath(src_path), dst_path)

    return temp_root

def  create_temp_actual_folder(dataset_images_path_dir, start_idx, end_idx, temp_folder_name="sam2_temp_symlinks"):
    """
    Creates a clean temporary folder with actual copies of selected images from a dataset directory.

    Args:
        dataset_images_path_dir (str): Path to the folder containing RGB images.
        start_idx (int): Start frame index (inclusive).
        end_idx (int): End frame index (exclusive).
        temp_folder_name (str): Name for the temp folder (under /tmp). Default: 'sam2_temp_symlinks'.

    Returns:
        str: Path to the temporary folder with actual copies.
    """
    # Create sorted list of image filenames
    images_paths_ordered = sorted(os.listdir(dataset_images_path_dir))

    # Validate indices
    if start_idx < 0 or end_idx > len(images_paths_ordered):
        raise IndexError(f"Frame range [{start_idx}, {end_idx}) is out of bounds for {len(images_paths_ordered)} images.")

    # Create or clean up temp folder
    temp_root = os.path.join(tempfile.gettempdir(), temp_folder_name)
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    os.makedirs(temp_root)

    # Copy files
    for fname in images_paths_ordered[start_idx:end_idx]: # Includes the start index, excludes the end index
        src_path = os.path.join(dataset_images_path_dir, fname)
        dst_path = os.path.join(temp_root, fname)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source image not found: {src_path}")
        shutil.copy(src_path, dst_path)

    return temp_root



def sam2_logits_to_masks(out_mask_logits, img, threshold=0.0):
    """
    Convert SAM2 output logits to a list of binary masks matching image shape.
    
    Args:
        out_mask_logits: Output tensor from SAM2 model (shape [N, H, W])
        img: Reference image to match shape
        threshold: Value to threshold logits at (default: 0.0)
    
    Returns:
        List of binary masks as np.uint8 arrays with shape matching img's H,W (single channel)
    """
    masks = []
    
    # Handle case where out_mask_logits is None
    if out_mask_logits is None:
        return masks
    
    # Check if img has valid dimensions
    if img is None or img.shape[0] <= 0 or img.shape[1] <= 0:
        print(f"Warning: Invalid image shape: {img.shape if img is not None else 'None'}")
        return masks
    
    target_height, target_width = img.shape[0], img.shape[1]
    print(f"Target image shape: {target_height}x{target_width}")
    
    # Process each mask in the logits
    for i in range(len(out_mask_logits)):
        # Get logits for this mask
        mask_logits = out_mask_logits[i]
        
        # Convert to binary mask by thresholding and move to CPU
        binary_mask = (mask_logits > threshold).cpu().numpy().astype(np.uint8)
        print(f"Mask {i} shape before resize: {binary_mask.shape}")
        # remove the channel dimension if present - currently it is [1, H, W], convert to [H, W]
        if len(binary_mask.shape) == 3 and binary_mask.shape[0] == 1:
            binary_mask = binary_mask[0]
        elif len(binary_mask.shape) != 2:
            print(f"Warning: Unexpected mask shape {binary_mask.shape} for mask {i}. Expected [H, W] or [1, H, W].")
            continue
        

        
        # Check if the shape matches the target shape
        if binary_mask.shape != (target_height, target_width):
            # Ensure target dimensions are valid
            if target_width <= 0 or target_height <= 0:
                print(f"Skipping mask resize: invalid target dimensions ({target_width}x{target_height})")
                continue
                
            # Print shapes for debugging
            print(f"Resizing mask from {binary_mask.shape} to {target_height}x{target_width}")
            
            # Resize if needed
            binary_mask = cv2.resize(binary_mask, (target_width, target_height), 
                                    interpolation=cv2.INTER_NEAREST)
        
        masks.append(binary_mask)
    
    return masks