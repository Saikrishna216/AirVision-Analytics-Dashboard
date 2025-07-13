import streamlit as st
import cv2
import tempfile
import os
import time
import uuid
from utils.inference import load_yolo_model, run_vehicle_count, run_face_recognition, run_license_plate_recognition , format_indian_plate
from utils.face_embedding import FaceEmbedder
import math

# Make sure the models directory exists
os.makedirs("models", exist_ok=True)
os.makedirs("dataset", exist_ok=True)

st.set_page_config(layout="wide")
st.title("🔍AirVision Analytics App")

# Select input source
source = st.radio("Select Video Source", ["Webcam", "Upload Video", "Drone Feed (URL)"])

# Multi-select toggles
options = st.multiselect(
    "Select Analysis Modules",
    ["Vehicle Count", "License Plate Recognition", "Face Recognition", "Crowd Analysis"]
)

# Load video source
video_source = None
if source == "Upload Video":
    video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        video_source = tfile.name
elif source == "Drone Feed (URL)":
    drone_url = st.text_input("Enter Drone Video Stream URL (e.g., http://...)")
    if drone_url:
        video_source = drone_url
else:
    video_source = 0  # webcam

# Model paths
model_paths = {
    "Vehicle Count": "models/yolov8n.pt",
    "License Plate Recognition": "models/Licence.pt",
    "Face Recognition": "models/yolov8n-face.pt",
    "Crowd Analysis": "models/crowd_analysis.pt"
}

# Create Face Embedder if needed
face_embedder = None
if "Face Recognition" in options:
    with st.spinner("Loading face recognition model..."):
        # Initialize FaceEmbedder
        face_embedder = FaceEmbedder(
            detection_model_path=model_paths["Face Recognition"],
            embedding_model_path="models/facenet.pt",
            known_faces_dir="dataset",
            threshold=0.4
        )

# Define model loading and inference functions
def load_model(model_path):
    if "Face Recognition" in model_path:
        return None  # Face recognition is handled separately
    return load_yolo_model(model_path)

# Load only selected models
models = {}
for opt in options:
    if opt != "Face Recognition":  # Skip face recognition, which is loaded separately
        models[opt] = load_model(model_paths[opt])

# Add these UI containers before starting video processing
if st.button("Start Video Processing"):
    if video_source is not None:
        # Create persistent UI elements first
        stframe = st.empty()
        
        # Create persistent sidebar elements
        with st.sidebar:
            fps_info = st.empty()
            process_every_n_container = st.empty()
            debug_checkbox_container = st.empty()
            
            # Create persistent containers for license plates
            if "License Plate Recognition" in options:
                st.header("Detected License Plates")
                plates_container = st.empty()
                
                st.header("License Plate Debug")
                debug_container = st.container()
                debug_col1, debug_col2 = debug_container.columns(2)

        # Create persistent containers for face recognition before processing
        if "Face Recognition" in options:
            with st.sidebar:
                st.header("Face Recognition")
                faces_container = st.empty()
                faces_text_container = st.empty()

        # Now use the containers for initial values
        fps_info.text(f"Processing video...")
        process_every_n = process_every_n_container.slider("Process every N frames", 1, 10, 8)
        show_debug = debug_checkbox_container.checkbox("Show License Plate Debug Images", value=False)
        
        # Initialize the video capture
        cap = cv2.VideoCapture(video_source)
        
        # Get original video FPS
        original_fps = cap.get(cv2.CAP_PROP_FPS) 
        if original_fps <= 0:
            original_fps = 30  # Default if FPS cannot be determined
        
        # Update FPS information
        fps_info.text(f"Video FPS: {original_fps:.1f}")
        
        # Calculate frame time in seconds
        frame_time = 1.0 / original_fps
        
        # Add process counter for skipping frames
        frame_count = 0
        
        # Initialize variables to store face recognition results
        face_results = {
            "boxes": [],
            "names": [],
            "confidences": []
        }
        last_face_detection_time = time.time()
        face_detection_interval = 1.0  # seconds between face detection updates

        # Add this before the video processing loop
        detected_plates = []  # Will store (plate_text, timestamp) pairs
        
        # Timing variables for maintaining frame rate
        last_frame_time = time.time()

        # Add a global vehicle counter outside the loop
        vehicle_count = 0

        # Add persistent variables
        last_plates_displayed = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Create a copy of the frame for display
            display_frame = frame.copy()
            current_time = time.time()

            # Print timestamp and frame information at regular intervals
            if frame_count % 30 == 0:  # Print every 30 frames
                print(f"\n--- Processing Frame {frame_count} | Time: {time.strftime('%H:%M:%S')} ---")

            # Process models every Nth frame, but always display the latest vehicle count
            if frame_count % process_every_n == 0:
                for opt in options:
                    if opt == "Vehicle Count" and opt in models:
                        # Update vehicle count only on processing frames
                        display_frame, vehicle_count = run_vehicle_count(models[opt], display_frame, return_count=True)
                        print(f"Vehicle count: {vehicle_count}")
                    elif opt == "License Plate Recognition" and opt in models:
                        display_frame, plates, debug_imgs = run_license_plate_recognition(models[opt], display_frame, return_debug=True)
                        
                        # Prevent repeated plates in the Recent Plates tab
                        for plate in plates:
                            formatted_plate = format_indian_plate(plate)
                            if formatted_plate:  # Only proceed if it's a valid Indian format
                                current_time = time.strftime("%H:%M:%S")
                                
                                # Check if this plate text is already in detected_plates
                                plate_exists = False
                                for existing_plate, _ in detected_plates:
                                    if existing_plate == formatted_plate:
                                        plate_exists = True
                                        break
                                        
                                # Only add if it's a new unique plate
                                if not plate_exists:
                                    plate_with_time = (formatted_plate, current_time)
                                    detected_plates.append(plate_with_time)
                                    print(f"Detected license plate: {formatted_plate} at {current_time}")
                        
                        # Update the persistent plates container with current plates
                        if detected_plates:
                            # Create a nicer display with confidence and timestamp
                            plates_text = "\n".join([f"{i+1}. {plate} ({timestamp})" 
                                                   for i, (plate, timestamp) in enumerate(detected_plates[-10:])])
                            
                            # Use markdown for better formatting
                            plates_container.markdown(f"### Recent Plates:\n```\n{plates_text}\n```")
                            
                            # Keep track for non-processing frames
                            last_plates_displayed = detected_plates[-10:]
                        else:
                            plates_container.markdown("*No license plates detected yet*")
                        
                        # Show debug images if enabled
                        if show_debug and debug_imgs:
                            # Display in a grid for better organization
                            num_debug = min(len(debug_imgs), 2)
                            
                            for i in range(num_debug):
                                orig, processed = debug_imgs[i]
                                
                                # Add more columns if needed
                                if i == 0:
                                    debug_col1.image(orig, caption=f"Original #{i+1}", width=200)
                                    debug_col1.image(processed, caption=f"Enhanced #{i+1}", width=200)
                                else:
                                    debug_col2.image(orig, caption=f"Original #{i+1}", width=200)
                                    debug_col2.image(processed, caption=f"Enhanced #{i+1}", width=200)
            else:
                # For non-processing frames, still show the most recent vehicle count
                if "Vehicle Count" in options:
                    # Draw a semi-transparent background for the count text
                    overlay = display_frame.copy()
                    cv2.rectangle(overlay, (5, 25), (200, 80), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0, display_frame)
                    
                    # Draw the count with larger font
                    cv2.putText(display_frame, f"Vehicles: {vehicle_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
                # Also always show the license plates in the sidebar
                if "License Plate Recognition" in options and last_plates_displayed:
                    plates_text = "\n".join([f"{i+1}. {plate} ({timestamp})" 
                                           for i, (plate, timestamp) in enumerate(last_plates_displayed)])
                    # Use markdown instead of text_area to avoid duplicate widget ID issues
                    plates_container.markdown(f"### Recent Plates:\n```\n{plates_text}\n```")

            # Face recognition processing remains the same
            if "Face Recognition" in options and face_embedder:
                # Only do detection every face_detection_interval seconds
                if current_time - last_face_detection_time >= face_detection_interval:
                    # Run face detection and recognition
                    boxes, names, confidences = face_embedder.detect_and_recognize(frame)
                    
                    # Update stored results
                    if boxes:  # Only update if faces were detected
                        face_results["boxes"] = boxes
                        face_results["names"] = names
                        face_results["confidences"] = confidences
                        
                        # Update the faces container with recognized faces
                        frame_height, frame_width = frame.shape[:2]
                        faces_text_lines = []
                        for (x1, y1, x2, y2), name, conf in zip(boxes, names, confidences):
                            face_width = x2 - x1
                            # Use the same estimate_distance as in your inference.py
                            focal_length = (frame_width * 0.5) / math.tan(70 * 0.5 * math.pi/180)
                            distance = (15 * focal_length) / face_width / 100  # 15cm average face width, convert to metres
                            faces_text_lines.append(f"{name} ({conf:.2f}), {distance:.1f}m")
                        faces_text = "\n".join(faces_text_lines)
                        faces_text_container.text_area(
                            "Recognized Faces:",
                            faces_text,
                            height=150,
                            key=f"face_recognition_text_{uuid.uuid4()}"
)
                        
                    last_face_detection_time = current_time
                
                # Always draw the face boxes and labels using the most recent results
                display_frame = run_face_recognition(face_embedder, display_frame, 
                                                    face_results["boxes"],
                                                    face_results["names"],
                                                    face_results["confidences"])

            # Convert BGR to RGB
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            stframe.image(display_frame, channels="RGB", use_column_width=True)
            
            # Increment frame counter
            frame_count += 1
            
            # Calculate wait time to maintain original video speed
            elapsed = time.time() - last_frame_time
            wait_time = max(0, frame_time - elapsed)
            
            if wait_time > 0:
                time.sleep(wait_time)
                
            last_frame_time = time.time()

        cap.release()
        st.success("Video processing completed.")
    else:
        st.error("Please select a valid video source.")
