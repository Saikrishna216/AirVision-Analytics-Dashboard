from ultralytics import YOLO
import cv2
import math
import pytesseract
import numpy as np
from PIL import Image
import string
import re

# Set your tesseract path if needed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

YOLO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",  "sports ball","kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse","remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink","refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

VEHICLE_CLASS_IDS = [
    YOLO_CLASSES.index(cls) for cls in ["car", "truck", "bus", "motorcycle"]
]

FACE_WIDTH_CM = 15
CAMERA_FOV = 70

def load_yolo_model(model_path):
    return YOLO(model_path)

def estimate_distance(face_width_pixels, frame_width):
    focal_length = (frame_width * 0.5) / math.tan(CAMERA_FOV * 0.5 * math.pi/180)
    distance = (FACE_WIDTH_CM * focal_length) / face_width_pixels
    return distance / 100

def run_vehicle_count(model, frame, return_count=False):
    results = model(frame)
    vehicle_classes = {2, 3, 5, 7}
    vehicle_count = 0
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls in vehicle_classes:
                vehicle_count += 1
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                vehicle_type = YOLO_CLASSES[cls]
                label = f"{vehicle_type}: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 25), (200, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, f"Vehicles: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    if return_count:
        return frame, vehicle_count
    else:
        return frame

def run_face_recognition(face_embedder, frame, boxes=None, names=None, confidences=None):
    frame_height, frame_width = frame.shape[:2]
    if boxes is None or names is None or confidences is None : 
        boxes, names, confidences = face_embedder.detect_and_recognize(frame)
    recognized_count = sum(1 for name in names if name != "Unknown")
    unknown_count = len(names) - recognized_count
    for (x1, y1, x2, y2), name, conf in zip(boxes, names, confidences):
        face_width = x2 - x1
        distance = estimate_distance(face_width, frame_width)
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name}: {conf:.2f}, {distance:.1f}m"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, f"Recognized Faces: {recognized_count}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Unknown Faces: {unknown_count}", (10, 70),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return frame

def enhance_license_plate(plate_img):
    h, w = plate_img.shape[:2]
    
    # Resize to a consistent height while maintaining aspect ratio
    new_height = 200
    new_width = int(new_height * (w/h))
    resized = cv2.resize(plate_img, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Try multiple enhancement methods
    
    # 1. Standard sharpening for cleaner edges
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    
    # 2. Bilateral filter to preserve edges while removing noise
    bilateral = cv2.bilateralFilter(sharpened, 11, 17, 17)
    
    # 3. Apply CLAHE for better contrast in varied lighting
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    
    # 4. Different thresholding techniques
    _, otsu = cv2.threshold(bilateral, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY, 11, 2)
    
    # 5. Morphological operations for connecting broken characters
    kernel_v = np.ones((3,1), np.uint8)  # Vertical kernel
    kernel_h = np.ones((1,3), np.uint8)  # Horizontal kernel
    
    # Close operation to connect character parts
    morph_v = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel_v)
    morph_h = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel_h)
    
    # Edge enhancement for better character definition
    edges = cv2.Canny(gray, 100, 200)
    dilated_edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
    
    return {
        "adaptive": adaptive,
        "morph": morph_v,
        "morph_h": morph_h,
        "otsu": otsu,
        "bilateral": bilateral,
        "edges": dilated_edges,
        "clahe": clahe_img,
        "original_gray": gray,
        "sharpened": sharpened,
        "color": resized
    }

def ocr_license_plate(plate_img):
    enhanced_versions = enhance_license_plate(plate_img)
    results = []
    confidence_scores = []
    
    # Try plate-specific preprocessing and configurations
    configs = [
        # Standard config with increased contrast analysis
        r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_do_invert=0 -c textord_heavy_nr=1',
        
        # Config for cleaner plates with spacing
        r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tessedit_do_invert=0',
        
        # Config for more noisy/damaged plates
        r'--oem 3 --psm 13 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 -c tesseract_do_invert=0 -c edges_boxarea=0.9'
    ]
    
    # Try more preprocessed images with different techniques
    for img_type in ["morph", "adaptive", "bilateral", "original_gray"]:
        img = enhanced_versions[img_type]
        for config in configs:
            try:
                # Try image scaling variations
                for scale_factor in [1.0, 1.5, 0.8]:
                    if scale_factor != 1.0:
                        scaled_img = cv2.resize(img, (0,0), fx=scale_factor, fy=scale_factor)
                    else:
                        scaled_img = img
                        
                    pil_img = Image.fromarray(scaled_img)
                    ocr_data = pytesseract.image_to_data(pil_img, config=config, 
                                                       output_type=pytesseract.Output.DICT)
                    
                    # Process text more intelligently
                    text_parts = []
                    avg_confidence = 0
                    confidence_count = 0
                    
                    for i in range(len(ocr_data['text'])):
                        if ocr_data['text'][i].strip():
                            text_parts.append(ocr_data['text'][i])
                            avg_confidence += ocr_data['conf'][i]
                            confidence_count += 1
                    
                    # Join and clean up the text
                    plate_text = ''.join(text_parts)
                    avg_confidence = avg_confidence / confidence_count if confidence_count > 0 else 0
                    
                    # Lower threshold for initial detection to catch more potential plates
                    if plate_text and len(plate_text) >= 4 and avg_confidence > 25:
                        # Clean up text for license plates
                        plate_text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
                        results.append(plate_text)
                        confidence_scores.append(avg_confidence)
            except Exception as e:
                pass
    
    if not results:  # If standard OCR failed, try fuzzy OCR
        try:
            # Last resort - try a very permissive approach
            text = pytesseract.image_to_string(
                enhanced_versions["color"], 
                config='--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
            )
            if text:
                text = re.sub(r'[^A-Z0-9]', '', text.upper())
                if len(text) >= 4:
                    results.append(text)
                    confidence_scores.append(30)  # Default confidence
        except:
            pass
    
    if results:
        best_idx = confidence_scores.index(max(confidence_scores))
        return results[best_idx], confidence_scores[best_idx]
    
    return None, 0

# --- Plate post-processing and formatting ---

dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}

def format_license(text):
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char, 5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    for j in range(min(7, len(text))):
        if text[j] in mapping.get(j, {}).keys():
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_

def normalize_and_validate_plate(plate_text):
    """Advanced normalization for license plates using context-aware algorithms"""
    # Remove spaces and convert to uppercase
    plate_text = plate_text.upper().replace(' ', '')
    
    # Common OCR substitutions for license plates with positional context
    ocr_substitutions = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1',
        '8': 'B', 'B': '8',
        '5': 'S', 'S': '5',
        '2': 'Z', 'Z': '2',
        '6': 'G', 'G': '6',
        'D': '0', 'Q': '0',
        'T': '7', '7': 'T'
    }
    
    # Apply a more sophisticated contextual correction based on Indian plate formats
    corrected = ""
    for i, char in enumerate(plate_text):
        # First two positions are typically state code letters
        if i < 2:
            if char in "0123456789":
                # Convert digits in first positions to letters
                if char == '0': corrected += 'O'
                elif char == '1': corrected += 'I'
                elif char == '8': corrected += 'B'
                elif char == '5': corrected += 'S'
                elif char == '2': corrected += 'Z'
                elif char == '6': corrected += 'G'
                else: corrected += char
            else:
                corrected += char
        # Positions 2-3 are typically district code digits
        elif 2 <= i <= 3:
            if char in "OIZSBG":
                # Convert letters in number positions to digits
                if char == 'O': corrected += '0'
                elif char == 'I': corrected += '1'
                elif char == 'Z': corrected += '2'
                elif char == 'S': corrected += '5'
                elif char == 'B': corrected += '8'
                elif char == 'G': corrected += '6'
                else: corrected += char
            else:
                corrected += char
        # Positions 4-5 are typically series letters
        elif 4 <= i <= 5:
            if char in "0123456789":
                # Convert digits back to letters in series positions
                if char == '0': corrected += 'O'
                elif char == '1': corrected += 'I'
                elif char == '8': corrected += 'B'
                elif char == '5': corrected += 'S'
                elif char == '2': corrected += 'Z'
                elif char == '6': corrected += 'G'
                else: corrected += char
            else:
                corrected += char
        # Last positions are typically registration number
        else:
            if char in "OIZS":
                # Convert letters to digits in number positions
                if char == 'O': corrected += '0'
                elif char == 'I': corrected += '1'
                elif char == 'Z': corrected += '2'
                elif char == 'S': corrected += '5'
                else: corrected += char
            else:
                corrected += char
    
    return corrected

def format_indian_plate(text):
    """
    Format and validate Indian license plate numbers with more flexible matching
    """
    # Remove all spaces and convert to uppercase
    text = text.upper().replace(' ', '')
    
    # Try multiple patterns for Indian plates with increasing flexibility
    patterns = [
        # Standard format: 2 letters + 2 digits + 2 letters + 4 digits (KA 01 MP 6985)
        r'^([A-Z]{2})(\d{2})([A-Z]{2})(\d{4})$',
        
        # More flexible: 2 letters + 2 digits + 1-2 letters + 1-4 digits (KJ 23 UI 4567)
        r'^([A-Z]{2})(\d{2})([A-Z]{1,2})(\d{1,4})$',
        
        # Even more flexible: any 2 chars + 2 digits + any 1-2 chars + any 1-4 digits
        r'^([A-Z0-9]{2})(\d{2})([A-Z0-9]{1,2})(\d{1,4})$'
    ]
    
    for pattern in patterns:
        match = re.match(pattern, text)
        if match:
            # Format as KJ 23 UI 4567
            return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
    
    # If no pattern matched, see if we can still extract something that looks like an Indian plate
    # This is a last resort for plates with unusual formats
    if len(text) >= 7 and any(c.isalpha() for c in text) and any(c.isdigit() for c in text):
        # Try to find patterns like XX00XX0000 in a messier string
        match = re.search(r'([A-Z0-9]{2})(\d{2})([A-Z0-9]{1,2})(\d{1,4})', text)
        if match:
            return f"{match.group(1)} {match.group(2)} {match.group(3)} {match.group(4)}"
    
    return None

def run_license_plate_recognition(model, frame, return_debug=False):
    """
    Run license plate detection and recognition on a frame.
    First detects plates, draws bounding boxes, then runs OCR on each plate.
    """
    results = model(frame, verbose=False)
    detected_plates = []
    debug_imgs = []
    for result in results:
        boxes = result.boxes
        car_plates = []
        
        # Step 1 & 2: Detect license plates and draw bounding boxes immediately
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            
            # Draw bounding box for every detected plate region
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Plate: {conf:.2f}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            
            # Step 3: Extract plate image and run OCR
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
                
            plate_text, confidence = ocr_license_plate(plate_img)
            
            # If text is detected with reasonable confidence
            if plate_text and len(plate_text) >= 4:
                normalized = normalize_and_validate_plate(plate_text)
                formatted_plate = format_indian_plate(normalized)
                
                # Choose color and text based on whether it's a valid Indian format
                if formatted_plate:
                    box_color = (0, 255, 0)  # Green for valid plates
                    display_plate = formatted_plate
                    label = f"Valid: {formatted_plate}"
                else:
                    box_color = (0, 165, 255)  # Orange for invalid plates
                    display_plate = normalized
                    label = f"Text: {normalized}"
                
                # Draw bounding box with appropriate color
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Display the appropriate label
                text_y = y1-10 if y1-10 > 10 else y1+20  # Ensure text is visible
                cv2.putText(frame, label, (x1, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
                
                car_plates.append((display_plate, confidence, (x1, y1, x2, y2), plate_img))
                
        # Step 4: For each car, keep only the plate with the highest confidence
        if car_plates:
            best_plate = max(car_plates, key=lambda x: x[1])
            display_plate, confidence, (x1, y1, x2, y2), orig_img = best_plate
            
            # Add to the detected plates list if not already there
            if display_plate not in detected_plates:
                detected_plates.append(display_plate)
                
            # Add debug images if requested
            if return_debug:
                # Process the image for better visualization
                debug_img = enhance_license_plate(orig_img)
                debug_imgs.append((cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), 
                                  cv2.cvtColor(debug_img["otsu"], cv2.COLOR_GRAY2RGB)))
                
    if return_debug:
        return frame, detected_plates, debug_imgs
    else:
        return frame, detected_plates
