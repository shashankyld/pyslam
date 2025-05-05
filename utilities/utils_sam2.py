import os
import shutil
import tempfile

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
    for fname in images_paths_ordered[start_idx:end_idx]:
        src_path = os.path.join(dataset_images_path_dir, fname)
        dst_path = os.path.join(temp_root, fname)
        if not os.path.exists(src_path):
            raise FileNotFoundError(f"Source image not found: {src_path}")
        os.symlink(os.path.abspath(src_path), dst_path)

    return temp_root
