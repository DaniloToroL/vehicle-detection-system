# Vehicle Detection System

A professional-grade system for detecting and tracking vehicles and people in videos with cross-line counting capabilities.

## Features

- **High-accuracy detection** using YOLOv11 (state-of-the-art object detection)
- **Robust tracking** with DeepSORT for maintaining object identities
- **Cross-line counting** with directional tracking
- **Real-time visualization** with bounding boxes, IDs, and trajectories
- **Flexible configuration** via YAML files or CLI arguments
- **Clean, modular code** following best practices

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Navigate to the project directory:**
   ```bash
   cd d:\dev\robotipy\yoao\vehicle-detection-system
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The YOLOv11 model will be downloaded automatically on first run.

## Quick Start

### Basic Usage

Process a video with default settings:

```bash
py main.py --input samples/traffic.mp4
```

### Custom Line Position

Define a custom counting line (x1, y1, x2, y2):

```bash
py main.py --input video.mp4 --line 100,300,500,300
```

### Using a Configuration File

```bash
py main.py --config config/config.yaml --input video.mp4
```

### Save Without Display

Process without showing the video (faster):

```bash
py main.py --input video.mp4 --no-display
```

## Configuration

Edit `config/config.yaml` to customize:

- **Model settings**: Model size (yolo11n/s/m/l/x), confidence threshold, device (CPU/GPU)
- **Tracking settings**: Max age, initialization frames, IOU threshold
- **Line settings**: Coordinates, direction
- **Visualization**: Colors, line thickness, text size

## Output

The system generates:

1. **Annotated video** (saved to `output/` folder)
   - Bounding boxes around detected objects
   - Unique IDs for each tracked object
   - Object trajectories
   - Counting line visualization
   - Real-time statistics overlay

2. **Console statistics**
   - Total crossings
   - Directional counts (up/down/left/right)
   - Crossing events log

## Project Structure

```
vehicle-detection-system/
├── src/
│   ├── config.py           # Configuration management
│   ├── detector.py         # YOLOv11 detection wrapper
│   ├── tracker.py          # DeepSORT tracking
│   ├── line_counter.py     # Cross-line detection
│   ├── video_processor.py  # Main processing pipeline
│   └── utils.py            # Helper functions
├── config/
│   └── config.yaml         # Configuration file
├── models/                 # Model weights (auto-downloaded)
├── samples/                # Sample videos
├── output/                 # Processed videos
├── tests/                  # Unit tests
├── main.py                 # CLI entry point
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Command-Line Options

```
Options:
  -i, --input PATH      Path to input video file
  -o, --output PATH     Path to output video file
  -c, --config PATH     Path to configuration file
  --line TEXT          Counting line coordinates as "x1,y1,x2,y2"
  --no-display         Disable real-time display
  --no-save           Do not save output video
  --help              Show this message and exit
```

## Supported Object Classes

- **Vehicles**: Car, Truck, Bus, Motorcycle
- **People**: Person

(Based on COCO dataset classes)

## Performance

- **CPU**: ~10-15 FPS (YOLOv11n)
- **GPU**: ~30-60 FPS (YOLOv11n)

Model size options (speed vs accuracy trade-off):
- `yolo11n.pt` - Nano (fastest)
- `yolo11s.pt` - Small
- `yolo11m.pt` - Medium
- `yolo11l.pt` - Large
- `yolo11x.pt` - Extra Large (most accurate)

## Troubleshooting

### Video won't open
- Ensure the video path is correct
- Try converting the video to MP4 format

### Low FPS
- Use a smaller model (yolo11n.pt)
- Reduce video resolution
- Use GPU if available (set `device: "cuda"` in config)

### Missing detections
- Lower the `confidence_threshold` in config
- Use a larger model (yolo11m/l/x.pt)

### Double counting
- Adjust the `max_iou_distance` in tracker settings
- Ensure the counting line is positioned appropriately

## License

This project uses the following open-source libraries:
- Ultralytics YOLOv11 (AGPL-3.0)
- DeepSORT (GPL-3.0)
- OpenCV (Apache 2.0)

## Support

For issues or questions, please check the configuration settings and ensure all dependencies are properly installed.
