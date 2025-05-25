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
from copy import deepcopy

class DynamicObject:
    def __init__(self, id, prompts, mask):
        self.id = id
        self.prompts = prompts
        self.mask = mask
        self.frame_id = None  # To track the frame in which this object was detected
        self.frame = None
class DynamicObjects:
    def __init__(self, mask_size=(480, 640)):
        self.objects = {}
        self.combined_mask = None
        self.combined_prompts = []
        self.mask_size = mask_size
        self.frame_id = None
        self.frame = None
        
    def __iter__(self):
        """Allow iteration over the objects."""
        return iter(self.objects.values())
    
    def __len__(self):
        """Return the number of objects."""
        return len(self.objects)
    
    def append(self, dynamic_object):
        """Add a dynamic object to the collection (alias for add_object)."""
        self.add_object(dynamic_object)

    def add_object(self, dynamic_object):
        if dynamic_object.id not in self.objects:
            self.objects[dynamic_object.id] = dynamic_object
        else:
            # Update the existing object instead of raising an error
            # Merge prompts and update mask if needed
            existing_obj = self.objects[dynamic_object.id]
            existing_obj.prompts.extend(dynamic_object.prompts)
            if dynamic_object.mask is not None:
                existing_obj.mask = dynamic_object.mask

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

    def update_from_sam2_results(self, sam2_results, sam2_run_ids):
        """
        Update dynamic objects from SAM2 segmentation results.
        
        Args:
            sam2_results: Dictionary mapping SAM2 frame indices to object masks
                {sam2_frame_idx: {obj_id: binary_mask, ...}, ...}
            sam2_run_ids: Mapping between SAM2 indices and actual frame IDs
                {sam2_frame_idx: actual_frame_id, ...}
                
        This method updates objects only for the matching frame_id.
        """
        if not sam2_results or not sam2_run_ids:
            return
            
        for sam2_run_frame_idx, frame_results in sam2_results.items():
            if self.frame_id == sam2_run_ids[sam2_run_frame_idx]:
                for obj_id, binary_mask in frame_results.items():
                    # Convert mask to correct format if needed
                    if binary_mask.dtype != np.uint8:
                        mask = binary_mask.astype(np.uint8) * 255
                    else:
                        mask = binary_mask
                        
                    # Ensure mask has correct dimensions
                    if mask.shape != self.mask_size:
                        mask = cv2.resize(mask, (self.mask_size[1], self.mask_size[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                    
                    # Update existing object or create new one
                    if obj_id in self.objects:
                        # Update existing object
                        self.objects[obj_id].mask = mask
                    else:
                        # Create new object with empty prompts list
                        new_obj = DynamicObject(id=obj_id, prompts=[], mask=mask)
                        self.add_object(new_obj)
                
                # Reset combined mask since objects were updated
                self.combined_mask = None
                self.combined_prompts = []
                
                # Regenerate combined mask immediately
                self.get_combined_mask()
                
                # We found the matching frame, so we can stop
                break

    def copy(self):
        """
        Create a deep copy of the DynamicObjects instance using deepcopy.
        
        Returns:
            A new DynamicObjects instance with copies of all objects.
        """
        return deepcopy(self)