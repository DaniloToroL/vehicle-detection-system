from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from datetime import datetime
from .tracker import TrackedObject
from .utils import line_intersection
from .config import AppConfig

@dataclass
class CrossingEvent:
    """Data structure for a line crossing event."""
    track_id: int
    timestamp: str
    direction: str  # "up", "down", "left", "right"
    class_name: str
    centroid: Tuple[float, float]

class LineCounter:
    """
    Detects and counts objects crossing a virtual line.
    """
    def __init__(self, config: AppConfig):
        """
        Initialize the line counter.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        x1, y1, x2, y2 = config.line.coordinates
        self.line_start = (x1, y1)
        self.line_end = (x2, y2)
        self.direction_type = config.line.direction
        
        # Track previous positions
        self.previous_positions: Dict[int, Tuple[float, float]] = {}
        
        # Track which IDs have already crossed (to prevent double counting)
        self.crossed_ids: Set[int] = set()
        
        # Crossing events
        self.crossing_events: List[CrossingEvent] = []
        
        # Counters
        self.count_up = 0
        self.count_down = 0
        self.count_left = 0
        self.count_right = 0
        
        print(f"Line counter initialized with line from {self.line_start} to {self.line_end}")

    def update(self, tracked_objects: List[TrackedObject]) -> None:
        """
        Update the counter with new tracked objects.
        
        Args:
            tracked_objects: List of tracked objects from the current frame
        """
        for obj in tracked_objects:
            track_id = obj.track_id
            current_pos = obj.centroid
            
            # If we have a previous position, check for crossing
            if track_id in self.previous_positions:
                prev_pos = self.previous_positions[track_id]
                
                # Check if the trajectory crosses the line
                if line_intersection(prev_pos, current_pos, self.line_start, self.line_end):
                    # Only count if this ID hasn't crossed yet
                    if track_id not in self.crossed_ids:
                        direction = self._determine_direction(prev_pos, current_pos)
                        self._record_crossing(track_id, direction, obj.class_name, current_pos)
                        self.crossed_ids.add(track_id)
            
            # Update previous position
            self.previous_positions[track_id] = current_pos
    
    def _determine_direction(self, 
                            prev_pos: Tuple[float, float], 
                            current_pos: Tuple[float, float]) -> str:
        """
        Determine the direction of crossing.
        
        Args:
            prev_pos: Previous centroid position
            current_pos: Current centroid position
            
        Returns:
            Direction string: "up", "down", "left", or "right"
        """
        if self.direction_type == "vertical":
            # For vertical lines, check if moving up or down
            if current_pos[1] < prev_pos[1]:
                return "up"
            else:
                return "down"
        else:  # horizontal
            # For horizontal lines, check if moving left or right
            if current_pos[0] < prev_pos[0]:
                return "left"
            else:
                return "right"
    
    def _record_crossing(self, 
                        track_id: int, 
                        direction: str, 
                        class_name: str,
                        centroid: Tuple[float, float]) -> None:
        """
        Record a crossing event.
        
        Args:
            track_id: The track ID
            direction: The direction of crossing
            class_name: The class name of the object
            centroid: The centroid position at crossing
        """
        # Create event
        event = CrossingEvent(
            track_id=track_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            direction=direction,
            class_name=class_name,
            centroid=centroid
        )
        self.crossing_events.append(event)
        
        # Update counters
        if direction == "up":
            self.count_up += 1
        elif direction == "down":
            self.count_down += 1
        elif direction == "left":
            self.count_left += 1
        elif direction == "right":
            self.count_right += 1
        
        print(f"Crossing detected: {class_name} (ID: {track_id}) going {direction}")
    
    def get_total_count(self) -> int:
        """Get the total number of crossings."""
        return self.count_up + self.count_down + self.count_left + self.count_right
    
    def get_statistics(self) -> Dict[str, int]:
        """
        Get crossing statistics.
        
        Returns:
            Dictionary with crossing counts
        """
        return {
            "total": self.get_total_count(),
            "up": self.count_up,
            "down": self.count_down,
            "left": self.count_left,
            "right": self.count_right
        }
    
    def reset(self) -> None:
        """Reset all counters and tracking data."""
        self.previous_positions.clear()
        self.crossed_ids.clear()
        self.crossing_events.clear()
        self.count_up = 0
        self.count_down = 0
        self.count_left = 0
        self.count_right = 0
