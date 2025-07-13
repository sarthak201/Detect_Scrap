import cv2
import numpy as np
import logging
import os
from typing import List, Dict, Tuple
from datetime import datetime
import shutil
from pathlib import Path
import json
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImprovedYOLOSimilarityDetector:
    """
    Improved YOLO detector that finds similar NON-LIVING objects between pickup and warehouse folders
    with better duplicate handling and main object detection
    """
    
    def __init__(self, pickup_folder: str, warehouse_folder: str, output_folder: str = "similar_matches"):
        self.pickup_folder = pickup_folder
        self.warehouse_folder = warehouse_folder
        self.output_folder = output_folder
        self.similarity_threshold = 0.75
        
        # Create output directories
        self.create_output_directories()
        
        # YOLO model paths
        self.yolo_weights = "yolo_files/yolov4.weights"
        self.yolo_config = "yolo_files/yolov4.cfg"
        self.yolo_names = "yolo_files/coco.names"
        
        # Define NON-LIVING object classes only (exclude all living things)
        self.allowed_classes = {
            # Vehicles
            'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            # Furniture & Items
            'chair', 'couch', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            # Sports & Tools
            'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            # Bags & Containers
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
            # Electronics
            'stop sign', 'parking meter', 'bench', 'fire hydrant'
        }
        
        # Initialize YOLO
        self.setup_yolo()
        
        # Storage for detected objects
        self.pickup_objects = []
        self.warehouse_objects = []
        self.similar_matches = []
        self.used_pickup_objects = set()  # Track used objects to prevent duplicates
        self.used_warehouse_objects = set()
    
    def create_output_directories(self):
        """Create directories for saving similar objects and unmatched flagged images"""
        base_path = Path(self.output_folder)
        base_path.mkdir(exist_ok=True)
        
        # Create subdirectories for matches
        (base_path / "pickup_matches").mkdir(exist_ok=True)
        (base_path / "warehouse_matches").mkdir(exist_ok=True)
        (base_path / "combined_matches").mkdir(exist_ok=True)
        (base_path / "reports").mkdir(exist_ok=True)
        
        # Create flag directory for unmatched images
        (base_path / "flag").mkdir(exist_ok=True)
        (base_path / "flag" / "pickup_unmatched").mkdir(exist_ok=True)
        (base_path / "flag" / "warehouse_unmatched").mkdir(exist_ok=True)
        
        logger.info(f"Output directories created at: {base_path}")
    
    def setup_yolo(self):
        """Setup YOLO model"""
        try:
            # Check if YOLO files exist
            missing_files = []
            for file in [self.yolo_weights, self.yolo_config, self.yolo_names]:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                logger.error("YOLO files missing:")
                for file in missing_files:
                    logger.error(f"  - {file}")
                logger.error("Please download YOLO v4 files or update paths")
                raise FileNotFoundError("YOLO model files not found")
            
            # Load YOLO
            self.yolo_net = cv2.dnn.readNet(self.yolo_weights, self.yolo_config)
            self.yolo_layer_names = self.yolo_net.getLayerNames()
            self.yolo_output_layers = [self.yolo_layer_names[i - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
            
            # Load class names
            with open(self.yolo_names, "r") as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
            
            logger.info("✅ YOLO model loaded successfully!")
            logger.info(f"✅ Loaded {len(self.yolo_classes)} object classes")
            logger.info(f"✅ Filtering for {len(self.allowed_classes)} non-living object classes")
            self.yolo_available = True
            
        except Exception as e:
            logger.error(f"❌ YOLO setup failed: {e}")
            self.yolo_available = False
            raise
    
    def detect_main_object_in_image(self, image_path: str, folder_type: str) -> Dict:
        """Detect the MAIN (largest, most confident) non-living object in image"""
        if not self.yolo_available:
            return None
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Could not load image: {image_path}")
                return None
            
            height, width, channels = image.shape
            
            # Prepare blob for YOLO
            blob = cv2.dnn.blobFromImage(image, 0.00392, (608, 608), (0, 0, 0), True, crop=False)
            self.yolo_net.setInput(blob)
            outputs = self.yolo_net.forward(self.yolo_output_layers)
            
            # Process detections
            valid_detections = []
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > 0.6:  # Higher confidence threshold
                        # Get class name
                        class_name = self.yolo_classes[class_id] if class_id < len(self.yolo_classes) else 'object'
                        
                        # Only keep non-living objects
                        if class_name not in self.allowed_classes:
                            continue
                        
                        # Get bounding box coordinates
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        # Skip very small objects
                        if w < 80 or h < 80:
                            continue
                        
                        # Calculate area and combined score
                        area = w * h
                        combined_score = confidence * np.sqrt(area) / 1000  # Area-weighted confidence
                        
                        valid_detections.append({
                            'bbox': [x, y, w, h],
                            'confidence': float(confidence),
                            'class_id': class_id,
                            'class_name': class_name,
                            'area': area,
                            'combined_score': combined_score
                        })
            
            if not valid_detections:
                logger.info(f"No valid non-living objects found in {image_path}")
                return None
            
            # Apply Non-Maximum Suppression to remove overlapping detections
            boxes = [det['bbox'] for det in valid_detections]
            confidences = [det['confidence'] for det in valid_detections]
            
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
            
            if len(indexes) == 0:
                return None
            
            # From remaining detections after NMS, pick the one with highest combined score
            nms_detections = [valid_detections[i] for i in indexes.flatten()]
            main_detection = max(nms_detections, key=lambda x: x['combined_score'])
            
            # Extract the main object
            x, y, w, h = main_detection['bbox']
            
            # Ensure coordinates are within image bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, width - x)
            h = min(h, height - y)
            
            # Extract ROI
            roi = image[y:y+h, x:x+w]
            
            if roi.size == 0:
                return None
            
            # Generate unique ID for this detection
            detection_id = f"{folder_type}_{os.path.splitext(os.path.basename(image_path))[0]}_main"
            
            main_object = {
                'id': detection_id,
                'bbox': (x, y, w, h),
                'confidence': main_detection['confidence'],
                'class': main_detection['class_name'],
                'roi': roi,
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'folder_type': folder_type,
                'area': main_detection['area'],
                'features': None  # Will be computed later
            }
            
            logger.info(f"Found main object '{main_detection['class_name']}' "
                       f"(conf: {main_detection['confidence']:.3f}, "
                       f"area: {main_detection['area']}) in {os.path.basename(image_path)}")
            
            return main_object
            
        except Exception as e:
            logger.error(f"Detection failed for {image_path}: {e}")
            return None
    
    def process_folder(self, folder_path: str, folder_type: str) -> List[Dict]:
        """Process all images in a folder and detect MAIN objects only"""
        logger.info(f"🔍 Processing {folder_type} folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            logger.error(f"Folder not found: {folder_path}")
            return []
        
        main_objects = []
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        
        # Get all unique image files using a set
        image_files = set()
        for ext in image_extensions:
            image_files.update(Path(folder_path).glob(f"*{ext}"))
            image_files.update(Path(folder_path).glob(f"*{ext.upper()}"))
        image_files = list(image_files)  # Convert back to list for processing
        
        logger.info(f"Found {len(image_files)} unique images in {folder_type} folder")
        
        processed = 0
        for image_path in image_files:
            main_object = self.detect_main_object_in_image(str(image_path), folder_type)
            if main_object:
                main_objects.append(main_object)
            processed += 1
            
            if processed % 5 == 0:
                logger.info(f"Processed {processed}/{len(image_files)} images in {folder_type}")
        
        logger.info(f"✅ Detected {len(main_objects)} main objects in {folder_type} folder")
        return main_objects
    
    def extract_enhanced_features(self, roi) -> np.ndarray:
        """Extract enhanced features optimized for object similarity"""
        try:
            if roi.size == 0:
                return np.zeros(512)
            
            # Resize to standard size
            roi_resized = cv2.resize(roi, (128, 128))
            
            # Convert to different color spaces
            roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
            roi_hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            roi_lab = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2LAB)
            
            features = []
            
            # 1. Enhanced color histograms (more bins for better discrimination)
            hist_h = cv2.calcHist([roi_hsv], [0], None, [36], [0, 180])  # Hue
            hist_s = cv2.calcHist([roi_hsv], [1], None, [32], [0, 256])  # Saturation
            hist_v = cv2.calcHist([roi_hsv], [2], None, [32], [0, 256])  # Value
            
            # Normalize and add
            for hist in [hist_h, hist_s, hist_v]:
                hist_norm = cv2.normalize(hist, hist).flatten()
                features.extend(hist_norm)
            
            # 2. L*a*b* color features (perceptually uniform)
            hist_l = cv2.calcHist([roi_lab], [0], None, [32], [0, 256])
            hist_a = cv2.calcHist([roi_lab], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([roi_lab], [2], None, [32], [0, 256])
            
            for hist in [hist_l, hist_a, hist_b]:
                hist_norm = cv2.normalize(hist, hist).flatten()
                features.extend(hist_norm)
            
            # 3. Enhanced texture features using Gabor-like filters
            gray_float = roi_gray.astype(np.float32)
            
            # Multiple orientation gradients
            texture_features = []
            for angle in [0, 45, 90, 135]:
                # Create oriented gradient
                if angle == 0:
                    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
                elif angle == 45:
                    kernel = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=np.float32)
                elif angle == 90:
                    kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
                else:  # 135
                    kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=np.float32)
                
                filtered = cv2.filter2D(gray_float, -1, kernel)
                texture_hist, _ = np.histogram(filtered.flatten(), bins=16, range=(-255, 255))
                texture_hist = texture_hist / np.sum(texture_hist) if np.sum(texture_hist) > 0 else texture_hist
                texture_features.extend(texture_hist)
            
            features.extend(texture_features)
            
            # 4. Shape context features
            edges = cv2.Canny(roi_gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            shape_features = []
            if contours:
                # Get the largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                
                # Shape descriptors
                area = cv2.contourArea(largest_contour)
                perimeter = cv2.arcLength(largest_contour, True)
                
                if perimeter > 0 and area > 0:
                    # Compactness
                    compactness = (perimeter * perimeter) / (4 * np.pi * area)
                    
                    # Aspect ratio
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = w / h if h > 0 else 1
                    
                    # Extent (object area / bounding box area)
                    extent = area / (w * h) if w * h > 0 else 0
                    
                    # Solidity (object area / convex hull area)
                    hull = cv2.convexHull(largest_contour)
                    hull_area = cv2.contourArea(hull)
                    solidity = area / hull_area if hull_area > 0 else 0
                    
                    shape_features = [compactness, aspect_ratio, extent, solidity]
                else:
                    shape_features = [0, 1, 0, 0]
            else:
                shape_features = [0, 1, 0, 0]
            
            features.extend(shape_features)
            
            # 5. Statistical moments
            moments_features = []
            for channel in [roi_gray, roi_hsv[:,:,0], roi_hsv[:,:,1], roi_hsv[:,:,2]]:
                moments_features.extend([
                    np.mean(channel),
                    np.std(channel),
                    float(np.mean((channel - np.mean(channel))**3)),  # Skewness
                    float(np.mean((channel - np.mean(channel))**4))   # Kurtosis
                ])
            
            features.extend(moments_features)
            
            # Convert to numpy array and handle any NaN values
            feature_vector = np.array(features, dtype=np.float32)
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
            
            # L2 normalization
            norm = np.linalg.norm(feature_vector)
            if norm > 0:
                feature_vector = feature_vector / norm
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(512)
    
    def compute_features_for_objects(self, objects: List[Dict]):
        """Compute features for all objects"""
        logger.info(f"🧠 Computing enhanced features for {len(objects)} objects...")
        
        for i, obj in enumerate(objects):
            obj['features'] = self.extract_enhanced_features(obj['roi'])
            
            if (i + 1) % 10 == 0:
                logger.info(f"Computed features for {i + 1}/{len(objects)} objects")
    
    def find_best_matches(self) -> List[Dict]:
        """Find best matches using improved algorithm to prevent duplicates"""
        logger.info("🔍 Finding best similar objects (1:1 matching)...")
        
        if not self.pickup_objects or not self.warehouse_objects:
            logger.warning("No objects found in one or both folders")
            return []
        
        # Get feature matrices
        pickup_features = np.array([obj['features'] for obj in self.pickup_objects])
        warehouse_features = np.array([obj['features'] for obj in self.warehouse_objects])
        
        # Compute similarity matrix
        similarity_matrix = cosine_similarity(pickup_features, warehouse_features)
        
        logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
        logger.info(f"Max similarity: {np.max(similarity_matrix):.3f}")
        logger.info(f"Min similarity: {np.min(similarity_matrix):.3f}")
        
        # Find best matches using Hungarian-like approach (greedy best-first)
        matches = []
        used_pickup = set()
        used_warehouse = set()
        
        # Get all potential matches above threshold
        potential_matches = []
        for i in range(len(self.pickup_objects)):
            for j in range(len(self.warehouse_objects)):
                similarity = similarity_matrix[i][j]
                if similarity >= self.similarity_threshold:
                    potential_matches.append((i, j, similarity))
        
        # Sort by similarity (highest first)
        potential_matches.sort(key=lambda x: x[2], reverse=True)
        
        logger.info(f"Found {len(potential_matches)} potential matches above threshold {self.similarity_threshold}")
        
        # Greedily select best matches (each object can only be matched once)
        for pickup_idx, warehouse_idx, similarity in potential_matches:
            if pickup_idx not in used_pickup and warehouse_idx not in used_warehouse:
                pickup_obj = self.pickup_objects[pickup_idx]
                warehouse_obj = self.warehouse_objects[warehouse_idx]
                
                # Additional class-aware similarity check
                class_bonus = 0.0
                if pickup_obj['class'] == warehouse_obj['class']:
                    class_bonus = 0.1  # Bonus for same class
                
                final_similarity = similarity + class_bonus
                
                match = {
                    'pickup_object': pickup_obj,
                    'warehouse_object': warehouse_obj,
                    'similarity_score': float(final_similarity),
                    'raw_similarity': float(similarity),
                    'class_match': pickup_obj['class'] == warehouse_obj['class'],
                    'match_id': f"match_{len(matches):04d}"
                }
                
                matches.append(match)
                used_pickup.add(pickup_idx)
                used_warehouse.add(warehouse_idx)
                
                logger.info(f"Match {len(matches)}: {pickup_obj['class']} ↔ {warehouse_obj['class']} "
                           f"(similarity: {final_similarity:.3f})")
        
        logger.info(f"✅ Found {len(matches)} unique best matches")
        return matches
    
    def save_similar_objects(self, matches: List[Dict]):
        """Save similar objects to output folders"""
        logger.info(f"💾 Saving {len(matches)} similar object matches...")
        
        pickup_dir = Path(self.output_folder) / "pickup_matches"
        warehouse_dir = Path(self.output_folder) / "warehouse_matches"
        combined_dir = Path(self.output_folder) / "combined_matches"
        
        match_report = []
        
        for match in matches:
            match_id = match['match_id']
            similarity = match['similarity_score']
            
            pickup_obj = match['pickup_object']
            warehouse_obj = match['warehouse_object']
            
            # Save pickup object
            pickup_filename = f"{match_id}_pickup_{pickup_obj['class']}_{similarity:.3f}.jpg"
            pickup_save_path = pickup_dir / pickup_filename
            cv2.imwrite(str(pickup_save_path), pickup_obj['roi'])
            
            # Save warehouse object
            warehouse_filename = f"{match_id}_warehouse_{warehouse_obj['class']}_{similarity:.3f}.jpg"
            warehouse_save_path = warehouse_dir / warehouse_filename
            cv2.imwrite(str(warehouse_save_path), warehouse_obj['roi'])
            
            # Create combined image
            combined_img = self.create_enhanced_combined_image(pickup_obj, warehouse_obj, match)
            combined_filename = f"{match_id}_combined_{similarity:.3f}.jpg"
            combined_save_path = combined_dir / combined_filename
            cv2.imwrite(str(combined_save_path), combined_img)
            
            # Add to report
            match_info = {
                'match_id': match_id,
                'similarity_score': similarity,
                'raw_similarity': match['raw_similarity'],
                'class_match': match['class_match'],
                'pickup': {
                    'class': pickup_obj['class'],
                    'confidence': pickup_obj['confidence'],
                    'image': pickup_obj['image_name'],
                    'bbox': pickup_obj['bbox'],
                    'area': pickup_obj['area']
                },
                'warehouse': {
                    'class': warehouse_obj['class'],
                    'confidence': warehouse_obj['confidence'],
                    'image': warehouse_obj['image_name'],
                    'bbox': warehouse_obj['bbox'],
                    'area': warehouse_obj['area']
                }
            }
            match_report.append(match_info)
        
        # Save detailed match report
        report_path = Path(self.output_folder) / "reports" / f"similarity_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(match_report, f, indent=2)
        
        # Save summary report
        summary = {
            'total_pickup_objects': len(self.pickup_objects),
            'total_warehouse_objects': len(self.warehouse_objects),
            'total_matches': len(matches),
            'similarity_threshold': self.similarity_threshold,
            'class_matches': sum(1 for m in matches if m['class_match']),
            'average_similarity': np.mean([m['similarity_score'] for m in matches]) if matches else 0,
            'pickup_objects_matched': len([m for m in matches]),
            'warehouse_objects_matched': len([m for m in matches])
        }
        
        summary_path = Path(self.output_folder) / "reports" / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Saved all matches and reports to {self.output_folder}")
    
    def flag_unmatched_images(self, matches: List[Dict]) -> Dict:
        """Flag and save images that don't have matches"""
        logger.info("🏁 Flagging unmatched images...")
        
        flag_dir = Path(self.output_folder) / "flag"
        pickup_flag_dir = flag_dir / "pickup_unmatched"
        warehouse_flag_dir = flag_dir / "warehouse_unmatched"
        
        # Track matched image paths
        matched_pickup_paths = set()
        matched_warehouse_paths = set()
        
        for match in matches:
            matched_pickup_paths.add(match['pickup_object']['image_path'])
            matched_warehouse_paths.add(match['warehouse_object']['image_path'])
        
        # Find unmatched pickup images
        unmatched_pickup = []
        for obj in self.pickup_objects:
            if obj['image_path'] not in matched_pickup_paths:
                unmatched_pickup.append(obj)
                # Copy original image to flag directory
                src_path = obj['image_path']
                dest_filename = f"unmatched_pickup_{obj['image_name']}"
                dest_path = pickup_flag_dir / dest_filename
                try:
                    shutil.copy2(src_path, dest_path)
                    logger.info(f"Flagged unmatched pickup image: {obj['image_name']}")
                except Exception as e:
                    logger.error(f"Failed to copy {src_path} to flag directory: {e}")
        
        # Find unmatched warehouse images
        unmatched_warehouse = []
        for obj in self.warehouse_objects:
            if obj['image_path'] not in matched_warehouse_paths:
                unmatched_warehouse.append(obj)
                # Copy original image to flag directory
                src_path = obj['image_path']
                dest_filename = f"unmatched_warehouse_{obj['image_name']}"
                dest_path = warehouse_flag_dir / dest_filename
                try:
                    shutil.copy2(src_path, dest_path)
                    logger.info(f"Flagged unmatched warehouse image: {obj['image_name']}")
                except Exception as e:
                    logger.error(f"Failed to copy {src_path} to flag directory: {e}")
        
        # Create unmatched report
        unmatched_report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_pickup_images': len(self.pickup_objects),
                'total_warehouse_images': len(self.warehouse_objects),
                'matched_pickup_images': len(matched_pickup_paths),
                'matched_warehouse_images': len(matched_warehouse_paths),
                'unmatched_pickup_images': len(unmatched_pickup),
                'unmatched_warehouse_images': len(unmatched_warehouse),
                'total_matches_found': len(matches)
            },
            'unmatched_pickup_details': [
                {
                    'image_name': obj['image_name'],
                    'image_path': obj['image_path'],
                    'detected_class': obj['class'],
                    'confidence': obj['confidence'],
                    'flagged_copy_path': str(pickup_flag_dir / f"unmatched_pickup_{obj['image_name']}")
                }
                for obj in unmatched_pickup
            ],
            'unmatched_warehouse_details': [
                {
                    'image_name': obj['image_name'],
                    'image_path': obj['image_path'],
                    'detected_class': obj['class'],
                    'confidence': obj['confidence'],
                    'flagged_copy_path': str(warehouse_flag_dir / f"unmatched_warehouse_{obj['image_name']}")
                }
                for obj in unmatched_warehouse
            ]
        }
        
        # Save unmatched report
        reports_dir = Path(self.output_folder) / "reports"
        unmatched_report_path = reports_dir / "unmatched_images_report.json"
        with open(unmatched_report_path, 'w') as f:
            json.dump(unmatched_report, f, indent=2)
        
        logger.info(f"🏁 Unmatched images report:")
        logger.info(f"   - Unmatched pickup images: {len(unmatched_pickup)}")
        logger.info(f"   - Unmatched warehouse images: {len(unmatched_warehouse)}")
        logger.info(f"   - Flagged images saved to: {flag_dir}")
        logger.info(f"   - Report saved to: {unmatched_report_path}")
        
        return unmatched_report
    
    def serialize_matches(self, matches: List[Dict]) -> List[Dict]:
        """Convert matches to JSON-serializable format by removing numpy arrays and OpenCV objects"""
        serialized_matches = []
        
        for match in matches:
            pickup_obj = match['pickup_object']
            warehouse_obj = match['warehouse_object']
            
            # Create clean objects without numpy arrays or OpenCV objects
            clean_pickup = {
                'id': pickup_obj['id'],
                'bbox': [int(x) for x in pickup_obj['bbox']],  # Convert to list of ints
                'confidence': float(pickup_obj['confidence']),
                'class': pickup_obj['class'],
                'image_path': pickup_obj['image_path'],
                'image_name': pickup_obj['image_name'],
                'folder_type': pickup_obj['folder_type'],
                'area': int(pickup_obj['area'])
            }
            
            clean_warehouse = {
                'id': warehouse_obj['id'],
                'bbox': [int(x) for x in warehouse_obj['bbox']],  # Convert to list of ints
                'confidence': float(warehouse_obj['confidence']),
                'class': warehouse_obj['class'],
                'image_path': warehouse_obj['image_path'],
                'image_name': warehouse_obj['image_name'],
                'folder_type': warehouse_obj['folder_type'],
                'area': int(warehouse_obj['area'])
            }
            
            serialized_match = {
                'pickup_object': clean_pickup,
                'warehouse_object': clean_warehouse,
                'similarity_score': float(match['similarity_score']),
                'raw_similarity': float(match['raw_similarity']),
                'class_match': bool(match['class_match']),
                'match_id': match['match_id']
            }
            
            serialized_matches.append(serialized_match)
        
        return serialized_matches
    
    def create_enhanced_combined_image(self, pickup_obj, warehouse_obj, match):
        """Create an enhanced combined image with detailed information"""
        # Resize both ROIs to same height
        height = 250
        pickup_roi = pickup_obj['roi']
        warehouse_roi = warehouse_obj['roi']
        
        pickup_aspect = pickup_roi.shape[1] / pickup_roi.shape[0]
        warehouse_aspect = warehouse_roi.shape[1] / warehouse_roi.shape[0]
        
        pickup_resized = cv2.resize(pickup_roi, (int(height * pickup_aspect), height))
        warehouse_resized = cv2.resize(warehouse_roi, (int(height * warehouse_aspect), height))
        
        # Create combined image with more space for text
        total_width = pickup_resized.shape[1] + warehouse_resized.shape[1] + 80
        combined = np.ones((height + 120, total_width, 3), dtype=np.uint8) * 255
        
        # Place images
        combined[40:40+height, 20:20+pickup_resized.shape[1]] = pickup_resized
        combined[40:40+height, pickup_resized.shape[1]+60:pickup_resized.shape[1]+60+warehouse_resized.shape[1]] = warehouse_resized
        
        # Add enhanced labels
        cv2.putText(combined, "PICKUP", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(combined, f"{pickup_obj['class']} ({pickup_obj['confidence']:.2f})", 
                   (20, height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(combined, "WAREHOUSE", (pickup_resized.shape[1]+60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(combined, f"{warehouse_obj['class']} ({warehouse_obj['confidence']:.2f})", 
                   (pickup_resized.shape[1]+60, height + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Similarity info
        cv2.putText(combined, f"Similarity: {match['similarity_score']:.3f}", 
                   (20, height + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
        
        # Class match indicator
        class_text = "SAME CLASS" if match['class_match'] else "DIFF CLASS"
        color = (0, 200, 0) if match['class_match'] else (0, 100, 200)
        cv2.putText(combined, class_text, (20, height + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return combined
    
    def run_similarity_detection(self):
        """Main function to run the complete similarity detection process"""
        logger.info("🚀 Starting Enhanced YOLO Non-Living Object Similarity Detection")
        
        try:
            # Step 1: Process pickup folder (detect main objects only)
            self.pickup_objects = self.process_folder(self.pickup_folder, "pickup")
            
            # Step 2: Process warehouse folder (detect main objects only)
            self.warehouse_objects = self.process_folder(self.warehouse_folder, "warehouse")
            
            if not self.pickup_objects:
                logger.error("No non-living main objects detected in pickup folder")
                # Still flag unmatched images even if no pickup objects
                if self.warehouse_objects:
                    unmatched_report = self.flag_unmatched_images([])
                    return {
                        'matches': [],
                        'unmatched_report': unmatched_report,
                        'summary': {
                            'total_matches': 0,
                            'pickup_objects': 0,
                            'warehouse_objects': len(self.warehouse_objects),
                            'average_similarity': 0.0,
                            'similarity_threshold': float(self.similarity_threshold)
                        }
                    }
                return {'error': 'No objects detected in pickup folder'}
            
            if not self.warehouse_objects:
                logger.error("No non-living main objects detected in warehouse folder")
                # Flag unmatched pickup images
                unmatched_report = self.flag_unmatched_images([])
                return {
                    'matches': [],
                    'unmatched_report': unmatched_report,
                    'summary': {
                        'total_matches': 0,
                        'pickup_objects': len(self.pickup_objects),
                        'warehouse_objects': 0,
                        'average_similarity': 0.0,
                        'similarity_threshold': float(self.similarity_threshold)
                    }
                }
            
            # Step 3: Compute enhanced features
            self.compute_features_for_objects(self.pickup_objects)
            self.compute_features_for_objects(self.warehouse_objects)
            
            # Step 4: Find best matches (1:1 mapping)
            self.similar_matches = self.find_best_matches()
            
            if not self.similar_matches:
                logger.warning(f"No similar objects found above threshold {self.similarity_threshold}")
                logger.info("Try lowering the similarity threshold or check if objects are truly similar")
                # Still flag unmatched images even if no matches found
                unmatched_report = self.flag_unmatched_images([])
                return {
                    'matches': [],
                    'unmatched_report': unmatched_report,
                    'summary': {
                        'total_matches': 0,
                        'pickup_objects': len(self.pickup_objects),
                        'warehouse_objects': len(self.warehouse_objects),
                        'average_similarity': 0.0,
                        'similarity_threshold': float(self.similarity_threshold)
                    }
                }
            
            # Step 5: Save results
            self.save_similar_objects(self.similar_matches)
            
            # Step 6: Flag unmatched images
            unmatched_report = self.flag_unmatched_images(self.similar_matches)
            
            # Print detailed summary
            logger.info("="*70)
            logger.info("🎉 ENHANCED SIMILARITY DETECTION COMPLETE!")
            logger.info(f"📊 Pickup main objects detected: {len(self.pickup_objects)}")
            logger.info(f"📊 Warehouse main objects detected: {len(self.warehouse_objects)}")
            logger.info(f"📊 Similar matches found: {len(self.similar_matches)}")
            logger.info(f"📊 Unmatched pickup images: {unmatched_report['summary']['unmatched_pickup_images']}")
            logger.info(f"📊 Unmatched warehouse images: {unmatched_report['summary']['unmatched_warehouse_images']}")
            logger.info(f"📊 Class matches: {sum(1 for m in self.similar_matches if m['class_match'])}")
            logger.info(f"📊 Similarity threshold: {self.similarity_threshold}")
            if self.similar_matches:
                avg_sim = np.mean([m['similarity_score'] for m in self.similar_matches])
                logger.info(f"📊 Average similarity: {avg_sim:.3f}")
            logger.info(f"📁 Results saved to: {self.output_folder}")
            logger.info(f"🏁 Flagged unmatched images saved to: {self.output_folder}/flag/")
            logger.info("="*70)
            
            # Print individual matches
            logger.info("\n🔍 MATCH DETAILS:")
            for i, match in enumerate(self.similar_matches, 1):
                pickup_obj = match['pickup_object']
                warehouse_obj = match['warehouse_object']
                logger.info(f"  {i}. {pickup_obj['image_name']} ({pickup_obj['class']}) ↔ "
                           f"{warehouse_obj['image_name']} ({warehouse_obj['class']}) "
                           f"- Similarity: {match['similarity_score']:.3f}")
            
            # Return comprehensive results for API (JSON-serializable)
            return {
                'matches': self.serialize_matches(self.similar_matches),
                'unmatched_report': unmatched_report,
                'summary': {
                    'total_matches': len(self.similar_matches),
                    'pickup_objects': len(self.pickup_objects),
                    'warehouse_objects': len(self.warehouse_objects),
                    'average_similarity': float(np.mean([m['similarity_score'] for m in self.similar_matches])) if self.similar_matches else 0.0,
                    'similarity_threshold': float(self.similarity_threshold),
                    'class_matches': sum(1 for m in self.similar_matches if m['class_match'])
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Similarity detection failed: {e}")
            raise


# Usage example
if __name__ == "__main__":
    # Configuration
    PICKUP_FOLDER = "pickup"      # Path to pickup folder
    WAREHOUSE_FOLDER = "warehouse" # Path to warehouse folder
    OUTPUT_FOLDER = "similar_matches"  # Output folder for results
    
    # Create detector instance
    detector = ImprovedYOLOSimilarityDetector(
        pickup_folder=PICKUP_FOLDER,
        warehouse_folder=WAREHOUSE_FOLDER,
        output_folder=OUTPUT_FOLDER
    )
    
    # Adjust similarity threshold if needed (0.75 = 75% similar)
    # Lower values (0.6-0.7) will find more matches
    # Higher values (0.8-0.9) will be more selective
    detector.similarity_threshold = 0.75
    
    # Run the detection
    detector.run_similarity_detection()