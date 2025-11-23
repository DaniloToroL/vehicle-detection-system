import cv2
import numpy as np
import os
from typing import Optional
from tqdm import tqdm
from .config import AppConfig
from .detector import ObjectDetector
from .tracker import ObjectTracker
from .line_counter import LineCounter
from .utils import get_color_for_class, draw_text_with_background

# Set higher FFmpeg read attempts for videos with multiple streams
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '100000'

class VideoProcessor:
    """
    End-to-end video processing pipeline for vehicle detection and counting.
    """
    def __init__(self, config: AppConfig):
        """
        Initialize the video processor.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        
        # Initialize components
        print("Initializing components...")
        self.detector = ObjectDetector(config)
        self.tracker = ObjectTracker(config)
        self.line_counter = LineCounter(config)
        print("Video processor ready.")
    
    def process_video(self, 
                     input_path: Optional[str] = None,
                     output_path: Optional[str] = None) -> None:
        """
        Process a video file with detection, tracking, and counting.
        
        Args:
            input_path: Path to input video (overrides config)
            output_path: Path to output video (overrides config)
        """
        # Use provided paths or fall back to config
        input_path = input_path or self.config.video.input_path
        output_path = output_path or self.config.video.output_path
        
        print(f"Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height} @ {fps} FPS, {total_frames} frames")
        
        # Initialize video writer if saving output
        writer = None
        if self.config.video.save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            print(f"Output will be saved to: {output_path}")
        
        # Process frames
        frame_count = 0
        frames_written = 0
        with tqdm(total=total_frames, desc="Processing") as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Run detection
                detections = self.detector.detect(frame)
                
                # Update tracker with frame
                tracked_objects = self.tracker.update(detections, frame)
                
                # Update line counter
                self.line_counter.update(tracked_objects)
                
                # Visualize
                annotated_frame = self._visualize(frame, tracked_objects)
                
                # Write frame if saving (do this BEFORE display to ensure all frames are written)
                if writer is not None:
                    writer.write(annotated_frame)
                    frames_written += 1
                
                # Show display if enabled
                if self.config.video.show_display:
                    cv2.imshow('Vehicle Detection', annotated_frame)
                    # Don't break on 'q' - just close the window but continue processing
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nDisplay closed by user. Processing continues...")
                        cv2.destroyAllWindows()
                        self.config.video.show_display = False  # Disable further display
                
                pbar.update(1)
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if self.config.video.show_display:
            cv2.destroyAllWindows()
        
        # Verify frame counts
        print(f"\n✅ Frames processed: {frame_count}/{total_frames}")
        if writer:
            print(f"✅ Frames written to output: {frames_written}")
        
        # Print statistics
        self._print_statistics()
    
    def _visualize(self, frame: np.ndarray, tracked_objects: list) -> np.ndarray:
        """
        Draw visualizations on the frame.
        
        Args:
            frame: Input frame
            tracked_objects: List of TrackedObject instances
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw counting line
        line_color = tuple(self.config.visualization.line_color)
        thickness = self.config.visualization.line_thickness
        x1, y1, x2, y2 = self.config.line.coordinates
        cv2.line(annotated, (int(x1), int(y1)), (int(x2), int(y2)), 
                line_color, thickness)
        
        # Draw tracked objects
        for obj in tracked_objects:
            # Get color for this class
            color = get_color_for_class(obj.class_id)
            
            # Draw bounding box with thicker line
            x1, y1, x2, y2 = [int(v) for v in obj.bbox]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 
                         self.config.visualization.box_thickness)
            
            # Draw label with larger text for better visibility
            label = f"{obj.class_name.upper()} #{obj.track_id}"
            draw_text_with_background(
                annotated, 
                label, 
                (x1, y1 - 5),
                font_scale=0.7,  # Larger font
                thickness=2,
                text_color=tuple(self.config.visualization.text_color),
                bg_color=color
            )
            
            # Trajectory drawing disabled - just recognition without tracking lines
            # trajectory = self.tracker.get_trajectory(obj.track_id)
            # if len(trajectory) > 1:
            #     points = np.array(trajectory, dtype=np.int32)
            #     cv2.polylines(annotated, [points], False, color, 2)
        
        # Draw statistics panel
        stats = self.line_counter.get_statistics()
        total_vehicles = self.tracker.get_total_vehicle_count()
        
        y_offset = 30
        
        # Total vehicles detected
        text = f"Total Vehicles: {total_vehicles}"
        draw_text_with_background(
            annotated,
            text,
            (10, y_offset),
            font_scale=0.8,
            thickness=2,
            text_color=(255, 255, 255),
            bg_color=(0, 128, 0)  # Green background
        )
        y_offset += 40
        
        # Line crossing statistics
        for key, value in stats.items():
            if key == 'total':
                continue  # Skip total for line crossings, we show it separately
            text = f"Crossed {key.capitalize()}: {value}"
            draw_text_with_background(
                annotated,
                text,
                (10, y_offset),
                font_scale=0.7,
                thickness=2
            )
            y_offset += 30
        
        return annotated
    
    def _print_statistics(self) -> None:
        """Print final statistics."""
        print("\n" + "="*50)
        print("PROCESSING COMPLETE")
        print("="*50)
        
        # Total vehicles
        total_vehicles = self.tracker.get_total_vehicle_count()
        print(f"\n📊 TOTAL UNIQUE VEHICLES DETECTED: {total_vehicles}")
        
        # Line crossing stats
        stats = self.line_counter.get_statistics()
        print(f"\n🚦 LINE CROSSING STATISTICS:")
        print(f"  Total crossings: {stats['total']}")
        print(f"  Up: {stats['up']}")
        print(f"  Down: {stats['down']}")
        print(f"  Left: {stats['left']}")
        print(f"  Right: {stats['right']}")
        print(f"\n📝 Crossing events logged: {len(self.line_counter.crossing_events)}")
        print("="*50)
