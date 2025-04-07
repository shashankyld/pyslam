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

