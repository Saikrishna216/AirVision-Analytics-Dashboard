import numpy as np
import os
import cv2
import torch
from sklearn.metrics.pairwise import cosine_similarity
from facenet_pytorch import InceptionResnetV1

class FaceEmbedder:
    def __init__(self, detection_model_path, embedding_model_path, known_faces_dir, threshold=0.4):
        # Import YOLO here to avoid circular imports
        from ultralytics import YOLO
        self.detector = YOLO(detection_model_path)
        self.recognizer = Recognizer(embedding_model_path, known_faces_dir, threshold)

    def detect_and_recognize(self, frame):
        """
        Detect faces in the frame and recognize them
        Returns:
            boxes: List of face bounding boxes (x1, y1, x2, y2)
            names: List of recognized names
            confidences: List of confidence scores
        """
        if frame is None or frame.size == 0:
            return [], [], []  # Return empty lists if frame is empty
            
        try:
            result = self.detector(frame, verbose=False)
            boxes, names, confidences = [], [], []

            if len(result) > 0 and hasattr(result[0], 'boxes') and hasattr(result[0].boxes, 'data'):
                for det in result[0].boxes.data:
                    if len(det) >= 5:
                        x1, y1, x2, y2, conf = det[:5].cpu().numpy()
                        if conf < 0.5 or (x2 - x1) < 20 or (y2 - y1) < 20:
                            continue

                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        if y1 >= 0 and y2 <= frame.shape[0] and x1 >= 0 and x2 <= frame.shape[1]:
                            face_crop = frame[y1:y2, x1:x2]
                            if face_crop.size > 0:
                                name, confidence = self.recognizer.identify(face_crop)
                                boxes.append((x1, y1, x2, y2))
                                names.append(name)
                                confidences.append(confidence)
            
            return boxes, names, confidences
        except Exception as e:
            print(f"Error in face detection: {e}")
            return [], [], []  # Return empty lists on error

class Recognizer:
    def __init__(self, embedding_model_path, known_faces_dir, threshold=0.4):
        self.threshold = threshold
        
        # Load the model
        if os.path.exists(embedding_model_path):
            state_dict = torch.load(embedding_model_path)
            if "logits.weight" in state_dict and state_dict["logits.weight"].shape[0] == 8631:
                # VGGFace2 pretrained model with 8631 identities
                self.model = InceptionResnetV1(pretrained=None, classify=True, num_classes=8631)
                self.model.load_state_dict(state_dict, strict=False)
                self.get_features = True
            else:
                # Model without classifier or with matching classifier size
                self.model = InceptionResnetV1(pretrained=None)
                self.model.load_state_dict(state_dict, strict=False)
                self.get_features = False
        else:
            # If model file not found, use pretrained model
            print(f"Model file {embedding_model_path} not found. Using pretrained model.")
            self.model = InceptionResnetV1(pretrained='vggface2')
            self.get_features = False
        
        self.model.eval()
        
        # Check if CUDA is available and move model to GPU if possible
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.embeddings = []
        self.labels = []

        self._load_known_faces(known_faces_dir)

    def preprocess(self, face):
        # OpenCV loads images in BGR, but PyTorch expects RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (160, 160))
        face = face.astype(np.float32)
        face = (face - 127.5) / 128.0
        # Convert to PyTorch tensor and adjust dimensions
        face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0)
        return face_tensor.to(self.device)

    def get_embedding(self, face_img):
        with torch.no_grad():
            input_tensor = self.preprocess(face_img)
            
            if self.get_features:
                # Get the 512-dimensional embedding before the classifier
                x = self.model.conv2d_1a(input_tensor)
                x = self.model.conv2d_2a(x)
                x = self.model.conv2d_2b(x)
                x = self.model.maxpool_3a(x)
                x = self.model.conv2d_3b(x)
                x = self.model.conv2d_4a(x)
                x = self.model.conv2d_4b(x)
                x = self.model.repeat_1(x)
                x = self.model.mixed_6a(x)
                x = self.model.repeat_2(x)
                x = self.model.mixed_7a(x)
                x = self.model.repeat_3(x)
                x = self.model.block8(x)
                x = self.model.avgpool_1a(x)
                x = self.model.dropout(x)
                x = self.model.last_linear(x.view(x.shape[0], -1))
                embedding = x
            else:
                # Standard forward pass
                embedding = self.model(input_tensor)
            
            return embedding.cpu().numpy()[0]

    def _load_known_faces(self, directory):
        self.embeddings = []
        self.labels = []
        
        if not os.path.exists(directory):
            print(f"Warning: Known faces directory '{directory}' does not exist.")
            return
            
        # Loop through each person's directory
        for person_name in os.listdir(directory):
            person_dir = os.path.join(directory, person_name)
            
            # Skip if not a directory
            if not os.path.isdir(person_dir):
                continue
                
            # Process each image
            for image_file in os.listdir(person_dir):
                if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                image_path = os.path.join(person_dir, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Could not read {image_path}")
                        continue
                        
                    embedding = self.get_embedding(image)
                    
                    self.embeddings.append(embedding)
                    self.labels.append(person_name)
                    
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    
        print(f"Total faces loaded: {len(self.embeddings)} from {len(set(self.labels))} people")
        self.embeddings = np.array(self.embeddings)

    def identify(self, face_img):
        try:
            if not self.embeddings.size:
                return "Unknown", 0.0
                
            embedding = self.get_embedding(face_img)
            sims = cosine_similarity([embedding], self.embeddings)[0]
            max_idx = np.argmax(sims)
            max_sim = sims[max_idx]
            if max_sim >= self.threshold:
                return self.labels[max_idx], float(max_sim)
            else:
                return "Unknown", float(max_sim)
        except Exception as e:
            print(f"Error in embedding: {e}")
            return "Unknown", 0.0