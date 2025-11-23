#!/usr/bin/env python3
"""
Vehicle Detection System - Main Entry Point

Process videos to detect, track, and count vehicles and people crossing a line.
"""

# IMPORTANT: Set this BEFORE importing any modules that use OpenCV
# OpenCV reads this environment variable when it's first loaded
import os
os.environ['OPENCV_FFMPEG_READ_ATTEMPTS'] = '100000'  # Increased for multi-stream videos
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'

import sys
from pathlib import Path
import click

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.config import AppConfig
from src.video_processor import VideoProcessor

@click.command()
@click.option(
    '--input', '-i',
    type=click.Path(exists=True),
    help='Path to input video file'
)
@click.option(
    '--output', '-o',
    type=click.Path(),
    help='Path to output video file'
)
@click.option(
    '--config', '-c',
    type=click.Path(exists=True),
    default='config/config.yaml',
    help='Path to configuration file (default: config/config.yaml)'
)
@click.option(
    '--line',
    type=str,
    help='Counting line coordinates as "x1,y1,x2,y2" (e.g., "100,300,500,300")'
)
@click.option(
    '--no-display',
    is_flag=True,
    help='Disable real-time display'
)
@click.option(
    '--no-save',
    is_flag=True,
    help='Do not save output video'
)
def main(input, output, config, line, no_display, no_save):
    """
    Vehicle Detection System
    
    Detect, track, and count vehicles and people in videos.
    
    Example usage:
    
        python main.py --input samples/traffic.mp4
        
        python main.py --input video.mp4 --line 100,300,500,300
        
        python main.py --config custom_config.yaml --input video.mp4
    """
    print("="*60)
    print("VEHICLE DETECTION SYSTEM")
    print("="*60)
    
    # Load configuration
    app_config = AppConfig.load(config)
    
    # Override config with CLI arguments
    if input:
        app_config.video.input_path = input
    if output:
        app_config.video.output_path = output
    if line:
        try:
            coords = [int(x.strip()) for x in line.split(',')]
            if len(coords) != 4:
                raise ValueError("Line must have exactly 4 coordinates")
            app_config.line.coordinates = coords
        except Exception as e:
            click.echo(f"Error parsing line coordinates: {e}", err=True)
            sys.exit(1)
    if no_display:
        app_config.video.show_display = False
    if no_save:
        app_config.video.save_output = False
    
    # Validate input file exists
    if not os.path.exists(app_config.video.input_path):
        click.echo(f"Error: Input file not found: {app_config.video.input_path}", err=True)
        click.echo("\nPlease provide a valid input video using --input option.", err=True)
        sys.exit(1)
    
    # Create output directory if needed
    if app_config.video.save_output:
        output_dir = os.path.dirname(app_config.video.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    # Process video
    try:
        processor = VideoProcessor(app_config)
        processor.process_video()
    except KeyboardInterrupt:
        click.echo("\n\nProcessing interrupted by user.")
        sys.exit(0)
    except Exception as e:
        click.echo(f"\n\nError during processing: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
