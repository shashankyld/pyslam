import subprocess
import base64
import os

# Convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Prepare inputs
image_path = "/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/pyslam/data/TUM/rgbd_bonn_crowd/rgb/1548339828.48902.png"
prompts = "[(100, 200), (150, 250)]"  # List of prompts
checkpoint = "/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/sam2/sam2.1_hiera_tiny.pt"
model_cfg = "/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/sam2/sam2/configs/sam2/sam2_hiera_t.yaml"

# Convert the image to base64
image_data = image_to_base64(image_path)

# Construct the command to run the script
command = [
    "conda", "run", "--name", "sam2_env", "python", "/home/shashank/Documents/UniBonn/Sem4/ThesisPrep/sam2/sam2_predictor.py",  # Use conda run to activate sam2_env
    "--checkpoint", checkpoint,
    "--model_cfg", model_cfg,
    "--image_data", image_data,
    "--prompts", prompts
]

# Run the command and capture the output
subprocess.run(command, check=True)
