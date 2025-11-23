import sys
import os

# Add the project root to the python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import AppConfig

def test_setup():
    print("Testing project setup...")
    
    # Test 1: Check directories
    dirs = ["src", "config", "models", "samples", "output", "tests"]
    for d in dirs:
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), d)
        if os.path.exists(path):
            print(f"✅ Directory {d} exists")
        else:
            print(f"❌ Directory {d} missing")

    # Test 2: Check config loading
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config", "config.yaml")
        config = AppConfig.load(config_path)
        print(f"✅ Config loaded successfully. Model: {config.model.name}")
    except Exception as e:
        print(f"❌ Config loading failed: {e}")

    print("\nSetup verification complete.")

if __name__ == "__main__":
    test_setup()
