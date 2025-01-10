import cv2
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import numpy as np

class MaskRCNNUtils:
    def __init__(self, threshold=0.8, device='cuda'):
        """
        Initializes MaskRCNNUtils with specified threshold and device.

        Args:
            threshold (float): Confidence threshold for detections (default: 0.8).
            device (str): Device to run the model on ('cpu' or 'cuda') (default: 'cpu').
        """
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT 
        self.model = maskrcnn_resnet50_fpn(weights=weights)
        self.model.to(device)  # Move the model to the specified device
        self.model.eval()  # Set the model to evaluation mode
        self.threshold = threshold
        self.device = device

    def detect_humans(self, image):
        """
        Detects humans in an image using Mask R-CNN from torchvision.

        Args:
            image: The input image.

        Returns:
            A list of dictionaries, where each dictionary represents a detected human 
            and contains bounding box, mask, and class information.
        """
        image_tensor = F.to_tensor(image).to(self.device)  # Move image tensor to the device
        with torch.no_grad():
            predictions = self.model([image_tensor])

        humans = []
        for i in range(len(predictions[0]['boxes'])):
            if predictions[0]['scores'][i] > self.threshold and predictions[0]['labels'][i] == 1:  # Filter for humans (label 1 is 'person')
                human = {
                    'box': predictions[0]['boxes'][i].detach().cpu().numpy(),
                    'mask': predictions[0]['masks'][i, 0].detach().cpu().numpy(),
                    'score': predictions[0]['scores'][i].item()
                }
                humans.append(human)
        return humans

    def visualize_humans(self, image, humans):
        """
        Visualizes the detected humans on the image by drawing bounding boxes 
        and overlaying semi-transparent masks.

        Args:
            image: The input image.
            humans: A list of human detection results (output of detect_humans).
        """
        for human in humans:
            box = human['box'].astype(int)
            mask = human['mask']

            # Draw bounding box
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

            # Overlay mask (semi-transparent)
            rgb_mask = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            image = cv2.addWeighted(image, 1, rgb_mask, 0.5, 0)  # Blend the image and the mask

        return image
    
    def human_mask(self, image):
        """
        Returns a mask of the detected humans as a NumPy array.

        Args:
            image: The input image.

        Returns:
            A NumPy array representing the binary mask of the detected humans.
        """
        
        humans = self.detect_humans(image)  # Call detect_humans internally
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        for human in humans:
            # Use the mask from the detection results directly
            mask = np.where(human['mask'] > 0.5, 255, mask)  
        return mask