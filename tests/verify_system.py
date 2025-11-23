"""
System verification script to check all components are working.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from src.config import AppConfig
        from src.detector import ObjectDetector, Detection
        from src.tracker import ObjectTracker, TrackedObject
        from src.line_counter import LineCounter, CrossingEvent
        from src.video_processor import VideoProcessor
        from src.utils import calculate_centroid, line_intersection, get_color_for_class
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    try:
        from src.config import AppConfig
        config = AppConfig.load("config/config.yaml")
        print(f"✅ Config loaded: Model={config.model.name}, Device={config.model.device}")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_detector_init():
    """Test detector initialization."""
    print("\nTesting detector initialization...")
    try:
        from src.config import AppConfig
        from src.detector import ObjectDetector
        config = AppConfig()
        detector = ObjectDetector(config)
        print("✅ Detector initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Detector initialization failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    try:
        from src.utils import calculate_centroid, line_intersection, get_color_for_class
        
        # Test centroid
        centroid = calculate_centroid([0, 0, 100, 100])
        assert centroid == (50.0, 50.0), "Centroid calculation failed"
        
        # Test line intersection
        intersects = line_intersection((0, 0), (100, 100), (0, 100), (100, 0))
        assert intersects == True, "Line intersection failed"
        
        # Test color
        color = get_color_for_class(2)
        assert color == (0, 255, 0), "Color mapping failed"
        
        print("✅ All utility functions working")
        return True
    except Exception as e:
        print(f"❌ Utility test failed: {e}")
        return False

def main():
    print("="*60)
    print("VEHICLE DETECTION SYSTEM - VERIFICATION")
    print("="*60)
    
    results = []
    results.append(test_imports())
    results.append(test_config())
    results.append(test_detector_init())
    results.append(test_utils())
    
    print("\n" + "="*60)
    if all(results):
        print("✅ ALL TESTS PASSED - System is ready!")
        print("="*60)
        print("\nNext steps:")
        print("1. Place a video file in the samples/ folder")
        print("2. Run: py main.py --input samples/your_video.mp4")
        print("="*60)
        return 0
    else:
        print("❌ SOME TESTS FAILED - Please check the errors above")
        print("="*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
