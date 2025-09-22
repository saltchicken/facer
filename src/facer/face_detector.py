import cv2
import numpy as np
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

# Global variable to cache the model
_model = None

def _load_model():
    """Loads and caches the YOLO model for face detection."""
    global _model
    if _model is None:
        try:
            model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
            _model = YOLO(model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    return _model

def crop_and_center_face(image, bbox):
    """Crops a 1:1 square from the image centered on the detected face."""
    h, w, _ = image.shape
    x1, y1, x2, y2 = map(int, bbox)
    
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    
    size = max(x2 - x1, y2 - y1)
    padding = size * 0.2
    size = int(size + padding)
    
    start_x = max(0, center_x - size // 2)
    start_y = max(0, center_y - size // 2)
    end_x = min(w, center_x + size // 2)
    end_y = min(h, center_y + size // 2)
    
    new_size = min(end_x - start_x, end_y - start_y)
    start_x = max(0, center_x - new_size // 2)
    start_y = max(0, center_y - new_size // 2)
    end_x = start_x + new_size
    end_y = start_y + new_size
    
    cropped_image = image[start_y:end_y, start_x:end_x]
    
    return cropped_image

def detect_and_crop_faces(image_array: np.ndarray) -> list:
    """Detects faces in an image and returns a list of cropped face images."""
    try:
        model = _load_model()
        results = model(image_array)
        
        cropped_faces = []
        if results and len(results[0].boxes.xyxy) > 0:
            for bbox in results[0].boxes.xyxy:
                cropped_face = crop_and_center_face(image_array, bbox)
                cropped_faces.append(cropped_face)
        
        return cropped_faces
    
    except Exception as e:
        print(f"An error occurred during face detection: {e}")
        return []
