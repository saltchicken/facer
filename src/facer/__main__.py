import argparse
import cv2
from facer.face_detector import FaceDetector
from facer.face_direction import FaceDirection
from facer.face_embedder import FaceEmbedder
from facer.face_identifier import FaceIdentifier

# Define acceptable pose thresholds (in degrees) that you can tune
YAW_THRESHOLD = 25.0
PITCH_THRESHOLD = 25.0

def main():
    parser = argparse.ArgumentParser(description="Facer: Analyze, enroll, or identify faces in an image.")
    parser.add_argument("image_path", type=str, help="Path to the input image.")
    parser.add_argument("--enroll", type=str, help="Enroll the first valid face found with the given name.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Cosine similarity threshold for face identification (default: 0.5).")
    args = parser.parse_args()

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: Could not load image from {args.image_path}")
        return

    # Instantiate all necessary classes
    face_detector = FaceDetector()
    face_direction_finder = FaceDirection()
    face_embedder = FaceEmbedder()
    face_identifier = FaceIdentifier(threshold=args.threshold)

    cropped_faces = face_detector.detect_and_crop(image)
    if not cropped_faces:
        print("No faces detected in the image.")
        return
        
    print(f"Detected {len(cropped_faces)} face(s). Analyzing each one...")
    
    # Flag to ensure we only enroll one face per command execution
    has_enrolled = False

    for i, face in enumerate(cropped_faces):
        print(f"\n--- Processing Face #{i+1} ---")
        
        # Step 1: Determine the face's orientation
        direction_info = face_direction_finder.direction(face)

        # Step 2: Check if the pose is acceptable
        if not (direction_info and abs(direction_info.yaw) < YAW_THRESHOLD and abs(direction_info.pitch) < PITCH_THRESHOLD):
            if direction_info:
                print(f"Skipping face: Pose is too extreme (Yaw: {direction_info.yaw:.2f}, Pitch: {direction_info.pitch:.2f}).")
            else:
                print("Skipping face: Could not determine direction.")
            continue # Move to the next face in the image

        print(f"Face pose is acceptable (Yaw: {direction_info.yaw:.2f}, Pitch: {direction_info.pitch:.2f}).")
        
        # Step 3: If pose is good, generate the embedding
        embedding = face_embedder.get_embedding(face)
        if embedding.size == 0:
            print("Skipping face: Failed to generate embedding.")
            continue

        # Step 4: Perform enrollment or identification
        if args.enroll and not has_enrolled:
            face_identifier.enroll(args.enroll, embedding)
            has_enrolled = True # Set flag to prevent enrolling other faces in the same image
        else:
            name = face_identifier.identify(embedding)
            print(f"Result: Face identified as '{name}'.")

if __name__ == "__main__":
    main()
