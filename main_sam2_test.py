import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

# --- Add thirdparty/sam2 to sys.path ---
SLAM_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))

# --- Import SAM 2 classes ---
from sam2.build_sam import build_sam2_video_predictor

# --- Helper Functions ---
def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id % 10
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

# --- Set Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Load SAM 2 Video Predictor ---
# Store current directory to restore it later
original_dir = os.getcwd()

# Change to the SAM2 directory to make relative paths work with Hydra
os.chdir(os.path.join(SLAM_ROOT, "thirdparty", "sam2"))


# Use relative paths that work with Hydra's package system
sam2_checkpoint = os.path.join("checkpoints", "sam2.1_hiera_tiny.pt")
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"  # Hydra will look for this relative to the sam2 package

# Make sure the checkpoint exists
if not os.path.exists(sam2_checkpoint):
    raise FileNotFoundError(f"Checkpoint file not found at: {os.path.join(os.getcwd(), sam2_checkpoint)}")

# Load the predictor
try:
    predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
    print("SAM2 predictor loaded successfully!")
finally:
    # Change back to the original directory
    os.chdir(original_dir)

# --- Set Up Video Sequence ---
# video_dir = os.path.join(SLAM_ROOT, "videos", "bedroom")  # Adjust as needed
# Video dir : /home/shashank/Documents/UniBonn/Sem4/ThesisPrep/pyslam/data/TUM/rgbd_bonn_crowd/rgb/1548339819.87426.png
video_dir = os.path.abspath("/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/pyslam/data/TUM/rgbd_bonn_crowd/rgb_jpg/")
if not os.path.exists(video_dir):
    os.makedirs(video_dir, exist_ok=True)
    print(f"Created video directory: {video_dir}")
    print("Please add video frames (.jpg/.jpeg) to this directory")
    exit(0)

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
]
print("frame names: ", frame_names)
if not frame_names:
    print(f"No image frames found in {video_dir}")
    print("Please add video frames (.jpg/.jpeg) to this directory")
    exit(0)

frame_names.sort(key=lambda p: float(os.path.splitext(p)[0]))
# --- Initialize Inference State ---
inference_state = predictor.init_state(video_path=video_dir)

# --- Add Object Prompt (e.g., clicks) ---
ann_frame_idx = 0
ann_obj_id = 1
points = np.array([[210, 350]], dtype=np.float32)
labels = np.array([1], np.int32)
_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    points=points,
    labels=labels
)

# --- Visualize Result ---
plt.figure(figsize=(9, 6))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_points(points, labels, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
plt.show()

# --- Propagate Across Video ---
video_segments = {}
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# --- Visualize Propagated Results ---
vis_frame_stride = 30
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)
    plt.show()

print("SAM 2 integration with SLAM project complete!")