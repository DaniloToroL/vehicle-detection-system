"""
Helper utilities for the vehicle detection system.
"""
import numpy as np
from typing import Tuple, List
import cv2

def calculate_centroid(bbox: List[float]) -> Tuple[float, float]:
    """
    Calculate the centroid of a bounding box.
    
    Args:
        bbox: Bounding box as [x1, y1, x2, y2]
        
    Returns:
        Tuple of (center_x, center_y)
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def line_intersection(p1: Tuple[float, float], 
                      p2: Tuple[float, float],
                      line_start: Tuple[float, float],
                      line_end: Tuple[float, float]) -> bool:
    """
    Check if line segment p1-p2 intersects with line segment line_start-line_end.
    
    Args:
        p1: First point of trajectory segment (x, y)
        p2: Second point of trajectory segment (x, y)
        line_start: Start point of counting line (x, y)
        line_end: End point of counting line (x, y)
        
    Returns:
        True if segments intersect, False otherwise
    """
    def ccw(A, B, C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
    
    A, B = p1, p2
    C, D = line_start, line_end
    
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """
    Get a consistent color for each class ID.
    
    Args:
        class_id: The class ID
        
    Returns:
        BGR color tuple
    """
    colors = {
        0: (255, 0, 0),      # person - blue
        2: (0, 255, 0),      # car - green
        3: (0, 255, 255),    # motorcycle - yellow
        5: (0, 0, 255),      # bus - red
        7: (255, 0, 255),    # truck - magenta
    }
    return colors.get(class_id, (200, 200, 200))

def draw_text_with_background(frame: np.ndarray, 
                               text: str, 
                               position: Tuple[int, int],
                               font_scale: float = 0.6,
                               thickness: int = 2,
                               text_color: Tuple[int, int, int] = (255, 255, 255),
                               bg_color: Tuple[int, int, int] = (0, 0, 0)) -> None:
    """
    Draw text with a background rectangle for better visibility.
    
    Args:
        frame: The image to draw on
        text: The text to draw
        position: (x, y) position for the text
        font_scale: Font scale
        thickness: Text thickness
        text_color: Text color in BGR
        bg_color: Background color in BGR
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    
    x, y = position
    
    # Draw background rectangle
    cv2.rectangle(
        frame,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        -1
    )
    
    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness
    )
