"""
Utility script to check video properties and verify frame counts.
"""
import cv2
import sys

def check_video(video_path):
    """Check video properties."""
    print(f"\n{'='*60}")
    print(f"VIDEO ANALYSIS: {video_path}")
    print(f"{'='*60}")
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}")
        return
    
    # Get properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    
    # Decode fourcc
    fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    print(f"\n📊 Properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Codec (FourCC): {fourcc_str}")
    print(f"  Reported frames: {total_frames}")
    
    # Count actual frames
    print(f"\n⏳ Counting actual frames...")
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    print(f"  Actual frames read: {frame_count}")
    
    # Calculate duration
    if fps > 0:
        duration = frame_count / fps
        print(f"\n⏱️  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    # Verify
    if frame_count == total_frames:
        print(f"\n✅ Frame count matches! Video is complete.")
    else:
        print(f"\n⚠️  Frame count mismatch!")
        print(f"     Expected: {total_frames}")
        print(f"     Got: {frame_count}")
        print(f"     Difference: {total_frames - frame_count}")
    
    cap.release()
    print(f"{'='*60}\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py check_video.py <video_path>")
        print("\nExample:")
        print("  py check_video.py samples/input.mp4")
        print("  py check_video.py output/result.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    check_video(video_path)
