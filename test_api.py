"""
Test script for YOLO Similarity Detection API

This script tests the FastAPI application by uploading test images
and checking the similarity detection functionality.

Directory Structure for Test Images:
deploy/
├── test_images/
│   ├── pickup/          # Put pickup location images here
│   │   ├── image1.jpg
│   │   ├── image2.png
│   │   └── ...
│   └── warehouse/       # Put warehouse images here
│       ├── image1.jpg
│       ├── image2.png
│       └── ...
├── app.py
├── test_api.py         # This script
└── ...
"""

import requests
import json
import os
from pathlib import Path
import time

# API Configuration
API_BASE_URL = "http://127.0.0.1:8000"
PICKUP_FOLDER = "test_images/pickup"
WAREHOUSE_FOLDER = "test_images/warehouse"

def create_test_directories():
    """Create test image directories if they don't exist"""
    pickup_dir = Path(PICKUP_FOLDER)
    warehouse_dir = Path(WAREHOUSE_FOLDER)
    
    pickup_dir.mkdir(parents=True, exist_ok=True)
    warehouse_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Created test directories:")
    print(f"   - {pickup_dir.absolute()}")
    print(f"   - {warehouse_dir.absolute()}")
    print(f"\n💡 Place your test images in these folders before running the test!")

def check_api_health():
    """Check if the API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Health Check: {data['message']}")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            return True
        else:
            print(f"❌ API Health Check Failed: Status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("❌ API is not running! Please start the FastAPI server first.")
        print("   Run: python app.py")
        return False
    except Exception as e:
        print(f"❌ Error checking API health: {e}")
        return False

def get_image_files(folder_path):
    """Get all image files from a folder"""
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    return list(set(image_files))  # Remove duplicates

def test_similarity_detection():
    """Test the similarity detection endpoint"""
    print("\n🔍 Testing Similarity Detection...")
    
    # Get image files
    pickup_images = get_image_files(PICKUP_FOLDER)
    warehouse_images = get_image_files(WAREHOUSE_FOLDER)
    
    if not pickup_images:
        print(f"❌ No images found in {PICKUP_FOLDER}")
        print(f"   Please add some test images to: {Path(PICKUP_FOLDER).absolute()}")
        return None
    
    if not warehouse_images:
        print(f"❌ No images found in {WAREHOUSE_FOLDER}")
        print(f"   Please add some test images to: {Path(WAREHOUSE_FOLDER).absolute()}")
        return None
    
    print(f"📸 Found {len(pickup_images)} pickup images and {len(warehouse_images)} warehouse images")
    
    # Prepare files for upload
    pickup_files = []
    warehouse_files = []
    
    try:
        # Open pickup images
        for img_path in pickup_images:
            pickup_files.append(('pickup_images', (img_path.name, open(img_path, 'rb'), 'image/jpeg')))
        
        # Open warehouse images
        for img_path in warehouse_images:
            warehouse_files.append(('warehouse_images', (img_path.name, open(img_path, 'rb'), 'image/jpeg')))
        
        # Prepare form data
        files = pickup_files + warehouse_files
        data = {'similarity_threshold': 0.75}
        
        print("📤 Uploading images and running similarity detection...")
        print("   This may take a few minutes depending on image count and size...")
        
        # Make API request
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/detect-similarity-upload/",
            files=files,
            data=data,
            timeout=300  # 5 minutes timeout
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Similarity detection completed in {end_time - start_time:.2f} seconds")
            print(f"   Run ID: {result['run_id']}")
            print(f"   Status: {result['status']}")
            print(f"   Message: {result['message']}")
            
            # Print report summary
            if 'report' in result and result['report']:
                report = result['report']
                print(f"\n📊 Detection Results:")
                print(f"   - Total matches found: {len(report)}")
                
                for i, match in enumerate(report[:3], 1):  # Show first 3 matches
                    print(f"   - Match {i}: {match['pickup']['class']} ↔ {match['warehouse']['class']}")
                    print(f"     Similarity: {match['similarity_score']:.3f}")
                    print(f"     Same class: {match['class_match']}")
                
                if len(report) > 3:
                    print(f"   ... and {len(report) - 3} more matches")
            
            print(f"\n📥 Download instructions:")
            print(f"   {result['download_instructions']}")
            
            return result
        else:
            print(f"❌ API request failed: Status {response.status_code}")
            print(f"   Error: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The detection process may take longer for large images.")
        return None
    except Exception as e:
        print(f"❌ Error during similarity detection: {e}")
        return None
    finally:
        # Close all opened files
        for _, (_, file_obj, _) in pickup_files + warehouse_files:
            if hasattr(file_obj, 'close'):
                file_obj.close()

def test_download_yolo_files():
    """Test YOLO files download endpoint"""
    print("\n📥 Testing YOLO files download...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/download-yolo-files/")
        if response.status_code == 200:
            result = response.json()
            print(f"✅ YOLO files check: {result['message']}")
            return True
        else:
            print(f"❌ YOLO files download failed: Status {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Error testing YOLO download: {e}")
        return False

def save_test_results(result, filename="test_results.json"):
    """Save test results to a JSON file"""
    if result:
        try:
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"\n💾 Test results saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving results: {e}")

def main():
    """Main test function"""
    print("🚀 YOLO Similarity Detection API Test")
    print("=" * 50)
    
    # Create test directories
    create_test_directories()
    
    # Check if API is running
    if not check_api_health():
        return
    
    # Test YOLO files download
    test_download_yolo_files()
    
    # Test similarity detection
    result = test_similarity_detection()
    
    # Save results
    if result:
        save_test_results(result)
        print("\n🎉 All tests completed successfully!")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()
