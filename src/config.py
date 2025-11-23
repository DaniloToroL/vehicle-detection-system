import yaml
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import os

@dataclass
class ModelConfig:
    name: str = "yolo11n.pt"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 0])
    device: str = "cpu"

@dataclass
class TrackerConfig:
    max_age: int = 30
    n_init: int = 3
    max_iou_distance: float = 0.7

@dataclass
class LineConfig:
    coordinates: List[int] = field(default_factory=lambda: [0, 500, 1280, 500])
    direction: str = "vertical"

@dataclass
class VideoConfig:
    input_path: str = "samples/traffic.mp4"
    output_path: str = "output/result.mp4"
    show_display: bool = True
    save_output: bool = True
    fps: int = 30
    resolution: List[int] = field(default_factory=lambda: [1280, 720])

@dataclass
class VisualizationConfig:
    line_color: List[int] = field(default_factory=lambda: [0, 0, 255])
    line_thickness: int = 2
    text_color: List[int] = field(default_factory=lambda: [255, 255, 255])
    text_scale: float = 0.8
    box_thickness: int = 2

@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    line: LineConfig = field(default_factory=LineConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def load(cls, config_path: str = "config/config.yaml") -> 'AppConfig':
        """Load configuration from a YAML file."""
        if not os.path.exists(config_path):
            print(f"Warning: Config file {config_path} not found. Using defaults.")
            return cls()

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        if not config_dict:
            return cls()

        # Helper to safely get nested dicts
        def get_section(section: str) -> dict:
            return config_dict.get(section, {})

        return cls(
            model=ModelConfig(**get_section('model')),
            tracker=TrackerConfig(**get_section('tracker')),
            line=LineConfig(**get_section('line')),
            video=VideoConfig(**get_section('video')),
            visualization=VisualizationConfig(**get_section('visualization'))
        )
