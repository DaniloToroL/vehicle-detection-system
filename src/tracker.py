from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict
from .config import AppConfig
from .detector import Detection

@dataclass
class TrackedObject:
    """Data structure for a tracked object."""
    track_id: int
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str
    centroid: Tuple[float, float]

class ObjectTracker:
    """
    Wrapper for DeepSORT tracker to maintain object identities across frames.
    """
    def __init__(self, config: AppConfig):
        """
        Initialize the tracker with configuration.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.tracker = DeepSort(
            max_age=config.tracker.max_age,
            n_init=config.tracker.n_init,
            max_iou_distance=config.tracker.max_iou_distance,
            embedder="mobilenet",  # Using MobileNet for feature extraction
            embedder_gpu=False  # Set to True if you have GPU
        )
        # Store trajectories: {track_id: [centroid1, centroid2, ...]}
        self.trajectories: Dict[int, List[Tuple[float, float]]] = defaultdict(list)
        
        # Track all unique IDs seen (for total vehicle count)
        self.all_tracked_ids: set = set()
        
        print("Tracker initialized.")

    def update(self, detections: List[Detection], frame: np.ndarray) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of Detection objects from the detector
            frame: Current frame (required for DeepSORT feature extraction)
            
        Returns:
            List of TrackedObject objects with unique IDs
        """
        # Convert detections to DeepSORT format
        # DeepSORT expects: ([bbox], confidence, class_name)
        raw_detections = []
        for det in detections:
            # DeepSORT expects [x, y, w, h] format
            x1, y1, x2, y2 = det.bbox
            w = x2 - x1
            h = y2 - y1
            raw_detections.append((
                [x1, y1, w, h],
                det.confidence,
                det.class_name
            ))
        
        # Update tracker with frame for feature extraction
        tracks = self.tracker.update_tracks(raw_detections, frame=frame)
        
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
                
            track_id = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            
            # Record this ID
            self.all_tracked_ids.add(track_id)
            
            # Calculate centroid
            centroid = (
                (ltrb[0] + ltrb[2]) / 2,
                (ltrb[1] + ltrb[3]) / 2
            )
            
            # Store trajectory
            self.trajectories[track_id].append(centroid)
            
            # Limit trajectory length to last 30 points
            if len(self.trajectories[track_id]) > 30:
                self.trajectories[track_id] = self.trajectories[track_id][-30:]
            
            # Get class information from the original detection
            class_name = track.get_det_class() if hasattr(track, 'get_det_class') else "unknown"
            # Map class name to class_id (simplified mapping)
            class_id_map = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7, "person": 0}
            class_id = class_id_map.get(class_name, 0)
            
            tracked_objects.append(TrackedObject(
                track_id=track_id,
                bbox=[ltrb[0], ltrb[1], ltrb[2], ltrb[3]],
                confidence=track.get_det_conf() if hasattr(track, 'get_det_conf') else 0.0,
                class_id=class_id,
                class_name=class_name,
                centroid=centroid
            ))
        
        return tracked_objects
    
    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """
        Get the trajectory for a specific track ID.
        
        Args:
            track_id: The track ID
            
        Returns:
            List of (x, y) centroids representing the object's path
        """
        return self.trajectories.get(track_id, [])
    
    def get_total_vehicle_count(self) -> int:
        """
        Get the total number of unique vehicles tracked.
        
        Returns:
            Total count of unique vehicle IDs
        """
        return len(self.all_tracked_ids)
