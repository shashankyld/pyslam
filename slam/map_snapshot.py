import time
import numpy as np
from collections import OrderedDict
from threading import RLock
from utils_sys import Printer

class MapSnapshot:
    """
    Class for maintaining serialized snapshots of maps for the last N timestamps.
    """
    def __init__(self, max_snapshots=10):
        self._lock = RLock()
        self.max_snapshots = max_snapshots
        self.snapshots = OrderedDict()

    def add_snapshot(self, current_map, timestamp=None):
        with self._lock:
            if timestamp is None:
                timestamp = time.time()
            
            try:
                Printer.blue(f"Creating map snapshot at timestamp {timestamp}")
                
                # Create a serialized representation of the map
                snapshot = {
                    'timestamp': timestamp,
                    'num_frames': current_map.num_frames(),
                    'num_keyframes': current_map.num_keyframes(),
                    'num_map_points': current_map.num_points(),
                    'num_local_map_points': current_map.local_map.num_points(),
                    'frames': {},
                    'keyframes': {},
                    'map_points': {},
                    'local_map_points': {}
                }
                
                # Store frame data
                for i in range(current_map.num_frames()):
                    frame = current_map.get_frame(i - current_map.num_frames())
                    if frame is not None:
                        frame_data = {
                            'timestamp': frame.timestamp,
                            'id': getattr(frame, 'id', None),
                            'pose': frame.pose.copy() if frame.pose is not None else None,
                            'kps_count': len(frame.kps) if frame.kps is not None else 0,
                            'kpsu_count': len(frame.kpsu) if frame.kpsu is not None else 0
                        }
                        snapshot['frames'][frame.timestamp] = frame_data
                
                # Store keyframe data
                keyframes = current_map.get_keyframes()
                for kf in keyframes:
                    if kf is not None:
                        kf_id = getattr(kf, 'id', kf.timestamp)  # Use timestamp if no id
                        kf_data = {
                            'timestamp': kf.timestamp,
                            'id': kf_id,
                            'pose': kf.pose.copy() if kf.pose is not None else None,
                            'kps_count': len(kf.kps) if kf.kps is not None else 0,
                            'kpsu_count': len(kf.kpsu) if kf.kpsu is not None else 0
                        }
                        snapshot['keyframes'][kf_id] = kf_data
                
                # Store global map points with descriptors
                points, colors = current_map.get_points_as_np()
                descriptors = None
                try:
                    map_points = current_map.get_points()
                    if map_points and len(map_points) > 0:
                        descriptors = [mp.des for mp in map_points if mp is not None and mp.des is not None]
                        if descriptors and len(descriptors) > 0:
                            descriptors = np.array(descriptors)
                except Exception as e:
                    Printer.orange(f"Could not extract descriptors from map points: {str(e)}")
                
                if points is not None and len(points) > 0:
                    snapshot['map_points'] = {
                        'points': points.copy(),
                        'colors': colors.copy() if colors is not None else None,
                        'descriptors': descriptors.copy() if descriptors is not None else None
                    }
                else:
                    snapshot['map_points'] = {
                        'points': None,
                        'colors': None,
                        'descriptors': None
                    }
                
                # Store local map points with descriptors
                local_points, local_colors = current_map.local_map.get_points_as_np()
                local_descriptors = None
                try:
                    local_map_points = current_map.local_map.get_points()
                    if local_map_points and len(local_map_points) > 0:
                        local_descriptors = [mp.des for mp in local_map_points if mp is not None and mp.des is not None]
                        if local_descriptors and len(local_descriptors) > 0:
                            local_descriptors = np.array(local_descriptors)
                except Exception as e:
                    Printer.orange(f"Could not extract descriptors from local map points: {str(e)}")
                
                if local_points is not None and len(local_points) > 0:
                    snapshot['local_map_points'] = {
                        'points': local_points.copy(),
                        'colors': local_colors.copy() if local_colors is not None else None,
                        'descriptors': local_descriptors.copy() if local_descriptors is not None else None
                    }
                else:
                    snapshot['local_map_points'] = {
                        'points': None,
                        'colors': None,
                        'descriptors': None
                    }
                
                # Store snapshot
                self.snapshots[timestamp] = snapshot
                
                # Remove oldest snapshots if we exceed the maximum
                while len(self.snapshots) > self.max_snapshots:
                    oldest_timestamp = next(iter(self.snapshots))
                    del self.snapshots[oldest_timestamp]
                    Printer.blue(f"Removed oldest snapshot at timestamp {oldest_timestamp}")
                
                return timestamp
                
            except Exception as e:
                Printer.red(f"Error creating map snapshot: {str(e)}")
                return None
                
    def get_snapshot(self, timestamp=None):
        """Get a snapshot at the specified timestamp or the newest one if None"""
        with self._lock:
            if not self.snapshots:
                Printer.orange("No map snapshots available")
                return None
            if timestamp is None:
                return self.snapshots[next(reversed(self.snapshots))]
            return self.snapshots.get(timestamp, None)

    def get_snapshot_at_index(self, index=-1):
        """Get a snapshot by index (-1 for newest, 0 for oldest)"""
        with self._lock:
            if not self.snapshots:
                Printer.orange("No map snapshots available")
                return None, None
            timestamps = list(self.snapshots.keys())
            if index < -len(timestamps) or index >= len(timestamps):
                Printer.orange(f"Index {index} out of range")
                return None, None
            timestamp = timestamps[index]
            return timestamp, self.snapshots[timestamp]

    def get_timestamps(self):
        """Get all timestamps with available snapshots"""
        with self._lock:
            return list(self.snapshots.keys())

    def clear(self):
        """Clear all snapshots"""
        with self._lock:
            self.snapshots.clear()
            Printer.blue("Cleared all map snapshots")

    def get_map_points_from_snapshot(self, timestamp_or_index, local=False):
        """
        Get map points from a snapshot.
        
        Args:
            timestamp_or_index: The timestamp of the snapshot, or an index
            local: Whether to get local map points instead of global
            
        Returns:
            Tuple of (points, colors) or (None, None) if not found
        """
        if isinstance(timestamp_or_index, int):
            timestamp, snapshot = self.get_snapshot_at_index(timestamp_or_index)
        else:
            snapshot = self.get_snapshot(timestamp_or_index)
            
        if snapshot is None:
            return None, None
            
        key = 'local_map_points' if local else 'map_points'
        if key not in snapshot:
            return None, None
            
        return snapshot[key]['points'], snapshot[key]['colors']

    def get_common_points_by_descriptor(self, timestamp_or_index1, timestamp_or_index2, local=False, ratio_test_threshold=0.8):
        """
        Get common map points between two snapshots by comparing descriptors.
        
        Args:
            timestamp_or_index1: The timestamp or index of the first snapshot
            timestamp_or_index2: The timestamp or index of the second snapshot
            local: Whether to get local map points instead of global
            ratio_test_threshold: Threshold for the ratio test (lower is stricter)
            
        Returns:
            Tuple of (points1, colors1, points2, colors2, matched_indices) for the common points,
            or (None, None, None, None, None) if not found or no common points
        """
        import numpy as np
        import cv2

        # Get first snapshot
        if isinstance(timestamp_or_index1, int):
            timestamp1, snapshot1 = self.get_snapshot_at_index(timestamp_or_index1)
        else:
            snapshot1 = self.get_snapshot(timestamp_or_index1)
            timestamp1 = timestamp_or_index1
        
        # Get second snapshot
        if isinstance(timestamp_or_index2, int):
            timestamp2, snapshot2 = self.get_snapshot_at_index(timestamp_or_index2)
        else:
            snapshot2 = self.get_snapshot(timestamp_or_index2)
            timestamp2 = timestamp_or_index2
        
        if snapshot1 is None or snapshot2 is None:
            Printer.orange("One or both snapshots not found")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        key = 'local_map_points' if local else 'map_points'
        
        if key not in snapshot1 or key not in snapshot2:
            Printer.orange(f"Key {key} not found in snapshots")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        # Get points, colors and descriptors
        points1 = snapshot1[key]['points']
        colors1 = snapshot1[key]['colors']
        descriptors1 = snapshot1[key].get('descriptors', None)
        
        points2 = snapshot2[key]['points']
        colors2 = snapshot2[key]['colors']
        descriptors2 = snapshot2[key].get('descriptors', None)
        
        if points1 is None or points2 is None or len(points1) == 0 or len(points2) == 0:
            Printer.orange("No points found in one or both snapshots")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        if descriptors1 is None or descriptors2 is None:
            Printer.orange("Descriptors not available, falling back to position-based matching")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
        
        try:
            # Ensure descriptors are numpy arrays with the right type
            if isinstance(descriptors1, list):
                descriptors1 = np.array(descriptors1)
            if isinstance(descriptors2, list):
                descriptors2 = np.array(descriptors2)
                
            # Ensure descriptors are in uint8 format for binary descriptor matching
            if descriptors1.dtype != np.uint8:
                descriptors1 = descriptors1.astype(np.uint8)
            if descriptors2.dtype != np.uint8:
                descriptors2 = descriptors2.astype(np.uint8)
            
            # Create descriptor matcher
            FLANN_INDEX_LSH = 6
            index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            
            # Find 2 best matches for each descriptor in descriptors1
            matches = matcher.knnMatch(descriptors1, descriptors2, k=2)
            
            # Apply ratio test
            good_matches = []
            for i, match_pair in enumerate(matches):
                if len(match_pair) >= 2:  # Ensure we have at least 2 matches for the ratio test
                    m, n = match_pair
                    if m.distance < ratio_test_threshold * n.distance:
                        good_matches.append((m.queryIdx, m.trainIdx))
            
            if len(good_matches) == 0:
                Printer.orange("No matching descriptors found")
                return np.array([]), np.array([]), np.array([]), np.array([]), []
            
            # Extract matched indices
            matched_indices1 = [match[0] for match in good_matches]
            matched_indices2 = [match[1] for match in good_matches]
            
            # Extract matching points and colors
            matched_points1 = points1[matched_indices1]
            matched_colors1 = colors1[matched_indices1] if colors1 is not None else None
            matched_points2 = points2[matched_indices2]
            matched_colors2 = colors2[matched_indices2] if colors2 is not None else None
            
            Printer.blue(f"Found {len(good_matches)} common points between snapshots using descriptor matching")
            
            return matched_points1, matched_colors1, matched_points2, matched_colors2, good_matches
            
        except Exception as e:
            Printer.red(f"Error in descriptor matching: {str(e)}")
            return np.array([]), np.array([]), np.array([]), np.array([]), []
