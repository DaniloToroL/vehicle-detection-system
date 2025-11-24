from ultralytics import YOLO
import numpy as np
import torch
from typing import List
from dataclasses import dataclass
from .config import AppConfig

# Fix for PyTorch 2.4+ weights_only=True default
# Monkeypatch torch.load to default weights_only=False if not specified
_original_load = torch.load

def _safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)

torch.load = _safe_load


@dataclass
class Detection:
    """Data structure for a single object detection."""
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    class_name: str

class ObjectDetector:
    """
    Wrapper for YOLOv11 model to handle object detection.
    """
    def __init__(self, config: AppConfig):
        """
        Initialize the detector with configuration.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.device = config.model.device
        print(f"Loading model: {config.model.name} on device: {self.device}...")
        self.model = YOLO(config.model.name)
        # Move model to specified device (GPU/CPU)
        self.model.to(self.device)
        self.classes = set(config.model.classes)
        self.conf_threshold = config.model.confidence_threshold
        print(f"Model loaded successfully on {self.device.upper()}.")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run YOLOv11 detection on a single frame.
        
        Args:
            frame: Input image/frame as numpy array (BGR)
            
        Returns:
            List of Detection objects
        """
        # Run inference on specified device (GPU/CPU)
        # verbose=False suppresses the default printing to stdout
        results = self.model(
            frame, 
            verbose=False, 
            conf=self.conf_threshold, 
            device=self.device,
            iou=self.config.model.iou_threshold)[0]

        detections = []
        
        # Process results
        for box in results.boxes:
            class_id = int(box.cls[0])
            
            # Filter by class
            if class_id not in self.classes:
                continue
                
            # Get coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            class_name = self.model.names[class_id]
            
            detections.append(Detection(
                bbox=[x1, y1, x2, y2],
                confidence=conf,
                class_id=class_id,
                class_name=class_name
            ))
            
        return detections
