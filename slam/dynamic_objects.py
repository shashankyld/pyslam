# Implimenting a class for dynamic objects in the SLAM system
"""
1. Create both dynamic_objects and dynamic_object classes.
2. The dynamic_objects class should contain a dictionary of dynamic_object instances.
3. The dynamic_object class should contain the following attributes:
    - id : int
    - mask : np.zeros_like(img)[:, :, 0], where Image shape:  (720, 1280, 3)
    - prompts : list of 2D points from delaunay triangulation for SAM2 prompting
"""
import numpy as np
import cv2

class DynamicObject:
    def __init__(self, id, prompts, mask):
        self.id = id
        self.prompts = prompts
        self.mask = mask

class DynamicObjects:
    def __init__(self, mask_size=(720, 1280)):
        self.objects = {}
        self.combined_mask = None
        self.combined_prompts = []
        self.mask_size = mask_size
        

    def add_object(self, dynamic_object):
        if dynamic_object.id not in self.objects:
            self.objects[dynamic_object.id] = dynamic_object
        else:
            raise ValueError(f"Object with id {dynamic_object.id} already exists.")

    def remove_object(self, id):
        if id in self.objects:
            del self.objects[id]
        else:
            raise ValueError(f"Object with id {id} does not exist.")

    def get_object(self, id):
        return self.objects.get(id, None)
    
    def get_combined_mask(self):
        if self.combined_mask is None:
            self.combined_mask = np.zeros(self.mask_size, dtype=np.uint8)
            for obj in self.objects.values(): 
                self.combined_mask = np.maximum(self.combined_mask, obj.mask)

        # Dilate the combined mask to fill in gaps
        kernel = np.ones((5, 5), np.uint8)
        self.combined_mask = cv2.dilate(self.combined_mask, kernel, iterations=5)
        return self.combined_mask
    
    def get_combined_prompts(self):
        if not self.combined_prompts:
            for obj in self.objects.values():
                self.combined_prompts.extend(obj.prompts)
        return self.combined_prompts

    
