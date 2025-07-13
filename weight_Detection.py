from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi import APIRouter
from typing import List, Optional
import cv2
import pytesseract
import easyocr
import numpy as np
import re
import os
import logging
from PIL import Image
import time
import io
import base64
from pydantic import BaseModel
import asyncio
from concurrent.futures import ThreadPoolExecutor
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
# app = FastAPI(
#     title="Weight Detection API",
#     description="API for detecting weights from digital display images",
#     version="1.0.0"
# )

router = APIRouter(
    prefix="/weight_detection",
    tags=["Weight Detection"]
)

# Initialize EasyOCR once globally
reader = easyocr.Reader(['en'], gpu=False)  # Set to True if GPU available

# Request/Response Models
class WeightDetectionRequest(BaseModel):
    expected_weights: List[float]
    tolerance: Optional[float] = 0.1  # 10% tolerance by default

class ImageResult(BaseModel):
    image_name: str
    detected_weight: str
    confidence: float
    method: str
    processing_time: float
    matches_expected: bool
    closest_expected_weight: Optional[float]
    difference: Optional[float]
    blur_info: dict

class WeightDetectionResponse(BaseModel):
    total_images: int
    successful_detections: int
    results: List[ImageResult]
    summary: dict

class OptimizedDisplayDetector:
    """Optimized detector for digital displays with multiple color support"""
    
    @staticmethod
    def detect_red_orange_numbers(image):
        """Optimized detection for red/orange numbers"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Simplified color ranges
        ranges = [
            ([0, 50, 50], [15, 255, 255]),      # Red range 1
            ([165, 50, 50], [180, 255, 255]),   # Red range 2
            ([10, 80, 80], [25, 255, 255]),     # Orange range
        ]
        
        # Combine masks efficiently
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # Single morphological operation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, None
        
        # Get largest contour above minimum area
        valid_contours = [c for c in contours if cv2.contourArea(c) > 50]
        if not valid_contours:
            return None, None
        
        largest_contour = max(valid_contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add padding
        padding = max(10, min(w, h) // 4)
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.shape[1] - x, w + 2 * padding)
        h = min(image.shape[0] - y, h + 2 * padding)
        
        roi = image[y:y+h, x:x+w]
        return (x, y, w, h), roi
    
    @staticmethod
    def detect_colored_display_region(image, color_name, hsv_lower, hsv_upper):
        """Detect specific colored digital display regions"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, hsv_lower, hsv_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            return (x, y, w, h)
        return None
    
    @staticmethod
    def detect_bright_regions(image):
        """Detect bright regions that might contain displays"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find bright regions
        _, bright_mask = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 200:  # Minimum area
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.5 < aspect_ratio < 10:  # Reasonable aspect ratio
                    regions.append((x, y, w, h))
        
        return regions

class BlurDetector:
    """Detect and measure blur in images"""
    
    @staticmethod
    def detect_blur(image):
        """Detect blur using multiple methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        gray = gray.astype(np.uint8)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).mean()
        blur_score = {
            'laplacian_variance': laplacian_var,
            'gradient_magnitude': gradient_magnitude,
            'is_blurred': laplacian_var < 100 or gradient_magnitude < 10
        }
        return blur_score

class OptimizedImageProcessor:
    """Optimized image processing for OCR"""
    
    @staticmethod
    def create_enhanced_versions(image, max_versions=6):
        """Create limited enhanced versions for OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        versions = []
        
        # 1. Original resized (3x scale)
        resized = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        versions.append(resized)
        
        # 2. CLAHE enhanced and resized
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_enhanced = clahe.apply(gray)
        clahe_resized = cv2.resize(clahe_enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        versions.append(clahe_resized)
        
        # 3. Binary threshold
        thresh_val = np.mean(gray) + np.std(gray)
        _, binary = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
        binary_resized = cv2.resize(binary, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        versions.append(binary_resized)
        
        # 4. Inverted binary
        inverted = cv2.bitwise_not(binary)
        inverted_resized = cv2.resize(inverted, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        versions.append(inverted_resized)
        
        # 5. Adaptive threshold
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        adaptive_resized = cv2.resize(adaptive, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
        versions.append(adaptive_resized)
        
        # 6. Edge enhanced
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        edge_enhanced = cv2.filter2D(gray, -1, kernel)
        edge_enhanced = np.clip(edge_enhanced, 0, 255).astype(np.uint8)
        edge_resized = cv2.resize(edge_enhanced, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        versions.append(edge_resized)
        
        return versions[:max_versions]

def optimized_tesseract_ocr(image):
    """Optimized Tesseract OCR with fewer configurations"""
    configs = [
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789.',
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.',
        r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.',
        r'--oem 3 --psm 13 -c tessedit_char_whitelist=0123456789.',
    ]
    
    results = set()
    
    for config in configs:
        try:
            text = pytesseract.image_to_string(image, config=config).strip()
            if text:
                # Clean and extract numbers
                cleaned = re.sub(r'[^\d.]', '', text)
                if cleaned and len(cleaned) >= 1:
                    results.add(cleaned)
                
                # Extract number patterns
                numbers = re.findall(r'\d+\.?\d*', text)
                results.update(num for num in numbers if len(num) >= 1)
        except Exception as e:
            logger.debug(f"Tesseract config failed: {e}")
            continue
    
    return list(results)

def optimized_easyocr_detection(image):
    """Optimized EasyOCR with fewer configurations"""
    results = []
    
    try:
        # Single configuration with good balance
        ocr_results = reader.readtext(image, detail=1, width_ths=0.5, height_ths=0.7, paragraph=False)
        
        for bbox, text, conf in ocr_results:
            if conf > 0.1:  # Lower threshold to catch more
                cleaned = re.sub(r'[^\d.]', '', text)
                if cleaned:
                    results.append((cleaned, conf))
                
                # Also try original text
                if text.strip() != cleaned:
                    results.append((text.strip(), conf))
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")
    
    return results

def format_weight_smart(raw_text):
    """Smart weight formatting"""
    if not raw_text:
        return None
    
    text = str(raw_text).strip().replace(' ', '').replace('O', '0').replace('o', '0')
    
    # Remove non-digit characters except decimal point
    digits_only = re.sub(r'[^\d.]', '', text)
    if not digits_only or len(digits_only) < 1:
        return None
    
    # Handle multiple decimal points
    if digits_only.count('.') > 1:
        parts = digits_only.split('.')
        digits_only = parts[0] + '.' + ''.join(parts[1:])
    
    try:
        num = float(digits_only)
        if 0.001 <= num <= 999999: 
            return digits_only
    except:
        pass
    
    # Try adding decimal point for long numbers without decimal
    if '.' not in digits_only and len(digits_only) > 3:
        for pos in [3, 2, 1]:
            if len(digits_only) > pos:
                formatted = digits_only[:-pos] + '.' + digits_only[-pos:]
                try:
                    num = float(formatted)
                    if 0.001 <= num <= 999999:
                        return formatted
                except:
                    continue
    
    return None

def detect_weight_comprehensive(image):
    """Comprehensive weight detection combining both methods"""
    start_time = time.time()
    
    detector = OptimizedDisplayDetector()
    processor = OptimizedImageProcessor()
    blur_detector = BlurDetector()
    
    # Get blur information
    blur_info = blur_detector.detect_blur(image)
    
    # Get regions to process
    regions_to_process = []
    
    # Try red/orange detection first
    red_result = detector.detect_red_orange_numbers(image)
    if red_result[0] is not None:
        regions_to_process.append(('red_orange', red_result[1]))
        logger.info("Red/orange region detected")
    
    # Try green detection
    green_region = detector.detect_colored_display_region(
        image, "green", np.array([40, 50, 50]), np.array([80, 255, 255])
    )
    if green_region:
        x, y, w, h = green_region
        roi = image[y:y+h, x:x+w]
        regions_to_process.append(('green_display', roi))
        logger.info("Green region detected")
    
    # Try blue detection
    blue_region = detector.detect_colored_display_region(
        image, "blue", np.array([90, 50, 50]), np.array([130, 255, 255])
    )
    if blue_region:
        x, y, w, h = blue_region
        roi = image[y:y+h, x:x+w]
        regions_to_process.append(('blue_display', roi))
        logger.info("Blue region detected")
    
    # Add bright regions as fallback
    bright_regions = detector.detect_bright_regions(image)
    for i, (x, y, w, h) in enumerate(bright_regions[:2]):  # Limit to 2 regions
        roi = image[y:y+h, x:x+w]
        regions_to_process.append((f'bright_{i}', roi))
    
    # Fallback to full image if no regions
    if not regions_to_process:
        regions_to_process.append(('full_image', image))
    
    all_detections = []
    
    for region_name, roi in regions_to_process:
        # Create enhanced versions (limited to 4 for speed)
        enhanced_versions = processor.create_enhanced_versions(roi, max_versions=4)
        
        for i, enhanced_img in enumerate(enhanced_versions):
            # Run OCR methods
            tesseract_results = optimized_tesseract_ocr(enhanced_img)
            easyocr_results = optimized_easyocr_detection(enhanced_img)
            
            # Process results
            for result in tesseract_results:
                formatted = format_weight_smart(result)
                if formatted:
                    confidence = 0.8 if 'display' in region_name or 'red_orange' in region_name else 0.6
                    all_detections.append((formatted, confidence, f'Tesseract-{region_name}_v{i+1}'))
            
            for text, conf in easyocr_results:
                formatted = format_weight_smart(text)
                if formatted:
                    if 'display' in region_name or 'red_orange' in region_name:
                        conf = min(1.0, conf * 1.2)
                    all_detections.append((formatted, conf, f'EasyOCR-{region_name}_v{i+1}'))
    
    # Get best result
    results = {
        'best_detection': 'Not detected',
        'confidence': 0,
        'method': 'None',
        'processing_time': time.time() - start_time,
        'total_detections': len(all_detections),
        'blur_info': blur_info
    }
    
    if all_detections:
        # Sort by confidence and prefer longer numbers
        all_detections.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
        
        best = all_detections[0]
        results['best_detection'] = best[0]
        results['confidence'] = best[1]
        results['method'] = best[2]
        
        # Log top detections
        logger.info("Top detections:")
        for i, (text, conf, method) in enumerate(all_detections[:3]):
            logger.info(f"  {i+1}. {text} (conf: {conf:.2f}, method: {method})")
    
    return results

def check_weight_match(detected_weight_str, expected_weights, tolerance=0.1):
    """Check if detected weight matches any expected weight within tolerance"""
    try:
        detected_weight = float(detected_weight_str)
    except (ValueError, TypeError):
        return False, None, None
    
    closest_weight = None
    min_difference = float('inf')
    
    for expected in expected_weights:
        difference = abs(detected_weight - expected)
        relative_difference = difference / expected if expected != 0 else float('inf')
        
        if difference < min_difference:
            min_difference = difference
            closest_weight = expected
        
        if relative_difference <= tolerance:
            return True, expected, difference
    
    return False, closest_weight, min_difference

async def process_single_image(image_data: bytes, image_name: str, expected_weights: List[float], tolerance: float):
    """Process a single image asynchronously"""
    try:
        # Convert bytes to opencv image
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Could not decode image")
        
        # Detect weight
        detection_result = detect_weight_comprehensive(image)
        
        # Check if detected weight matches expected weights
        matches, closest_expected, difference = check_weight_match(
            detection_result['best_detection'], 
            expected_weights, 
            tolerance
        )
        
        return ImageResult(
            image_name=image_name,
            detected_weight=detection_result['best_detection'],
            confidence=detection_result['confidence'],
            method=detection_result['method'],
            processing_time=detection_result['processing_time'],
            matches_expected=matches,
            closest_expected_weight=closest_expected,
            difference=difference,
            blur_info=detection_result['blur_info']
        )
        
    except Exception as e:
        logger.error(f"Error processing image {image_name}: {e}")
        return ImageResult(
            image_name=image_name,
            detected_weight="Error",
            confidence=0.0,
            method="Error",
            processing_time=0.0,
            matches_expected=False,
            closest_expected_weight=None,
            difference=None,
            blur_info={"error": str(e)}
        )

@router.post("/detect-weights", response_model=WeightDetectionResponse)
async def detect_weights(
    images: List[UploadFile] = File(...),
    expected_weights: str = Form(...),
    tolerance: float = Form(0.1)
):
    """
    Detect weights from multiple images and compare with expected weights
    
    - **images**: List of image files to process
    - **expected_weights**: Comma-separated list of expected weights (e.g., "1.5,2.3,4.7")
    - **tolerance**: Tolerance for weight matching (default: 0.1 = 10%)
    """
    try:
        # Parse expected weights
        expected_weights_list = [float(w.strip()) for w in expected_weights.split(',')]
        
        # Validate files
        if not images:
            raise HTTPException(status_code=400, detail="No images provided")
        
        # Process images concurrently
        tasks = []
        for image_file in images:
            image_data = await image_file.read()
            task = process_single_image(image_data, image_file.filename, expected_weights_list, tolerance)
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
        
        # Calculate summary statistics
        successful_detections = sum(1 for r in results if r.detected_weight != "Error" and r.detected_weight != "Not detected")
        matching_detections = sum(1 for r in results if r.matches_expected)
        total_confidence = sum(r.confidence for r in results if r.confidence > 0)
        avg_confidence = total_confidence / successful_detections if successful_detections > 0 else 0
        
        summary = {
            "total_images": len(images),
            "successful_detections": successful_detections,
            "matching_detections": matching_detections,
            "match_rate": matching_detections / len(images) if len(images) > 0 else 0,
            "average_confidence": avg_confidence,
            "expected_weights": expected_weights_list,
            "tolerance_used": tolerance
        }
        
        return WeightDetectionResponse(
            total_images=len(images),
            successful_detections=successful_detections,
            results=results,
            summary=summary
        )
        
    except Exception as e:
        logger.error(f"Error in detect_weights endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "Weight Detection API is running"}

@router.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Weight Detection API",
        "version": "1.0.0",
        "endpoints": {
            "/detect-weights": "POST - Detect weights from images",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        }
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(router, host="0.0.0.0", port=8000)