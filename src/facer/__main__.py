from facer.face_direction import FaceDirection
from facer.face_detector import FaceDetector
import cv2

def main():
    image = cv2.imread("/home/saltchicken/Pictures/mpv-shot0001.jpg")
    face_orienter = FaceDirection()
    face_detector = FaceDetector()
    cropped_faces = face_detector.detect_and_crop(image)
    for cropped_face in cropped_faces:
        face_direction = face_orienter.direction(cropped_face)
        if face_direction:
            print(face_direction)
        else:
            print("No direction detected for this face.")

if __name__ == "__main__":
    main()
