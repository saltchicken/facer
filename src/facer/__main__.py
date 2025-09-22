from facer.face_direction import FaceDirection
from facer.face_detector import FaceDetector
from facer.face_embedder import FaceEmbedder
import cv2

def main():
    image = cv2.imread("/home/saltchicken/Pictures/mpv-shot0001.jpg")
    if image is None:
        print("Error: Could not load image.")
        return

    face_orienter = FaceDirection()
    face_detector = FaceDetector()
    face_embedder = FaceEmbedder()
    
    cropped_faces = face_detector.detect_and_crop(image)
    
    for cropped_face in cropped_faces:
        face_direction = face_orienter.direction(cropped_face)
        if face_direction:
            print(f"Face direction: {face_direction}")
        else:
            print("No direction detected for this face.")
            
        face_embedding = face_embedder.get_embedding(cropped_face)
        if face_embedding.size > 0:
            print(f"Face embedding generated (size: {face_embedding.shape[0]})")
            # You can now use the embedding for further tasks like face recognition.
        else:
            print("Failed to generate face embedding.")

if __name__ == "__main__":
    main()
