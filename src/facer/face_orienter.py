import cv2
import mediapipe as mp
import numpy as np

# Adjust imports for the new structure
from .utils import get_yaw_pitch_roll, visualize_rotation_vector, plot_selected_landmarks
from .dataclasses import Direction

class FaceOrienter:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5, 
            min_tracking_confidence=0.5
        )

    def _get_image_points(self, landmarks, frame_dimensions):
        """Extracts key landmark points and converts them to image coordinates."""
        landmark_indices = [4, 152, 263, 33, 291, 61]
        image_points = np.array([
            (landmarks.landmark[idx].x * frame_dimensions[1], 
             landmarks.landmark[idx].y * frame_dimensions[0])
            for idx in landmark_indices
        ], dtype="double")
        return image_points

    def _get_camera_matrix(self, frame_dimensions):
        """Creates a simple camera matrix based on image size."""
        focal_length = frame_dimensions[1]
        center = (frame_dimensions[1] / 2, frame_dimensions[0] / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype="double")
        return camera_matrix

    def orient(self, frame: np.ndarray, show=False):
        """Analyzes a face from a NumPy array and returns yaw, pitch, and roll angles."""
        if frame is None:
            print("Error: Input frame is None.")
            return None, None, None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            print("No face detected.")
            return None, None, None

        landmarks = results.multi_face_landmarks[0]
        
        # Plot landmarks for visualization
        if show:
            plot_selected_landmarks(frame, landmarks)

        frame_dimensions = frame.shape
        image_points = self._get_image_points(landmarks, frame_dimensions)

        # 3D model points are based on a generic head model
        model_points = np.array([
            (0.0, 0.0, 0.0),      # Nose tip
            (0.0, -330.0, -65.0), # Chin
            (-225.0, 170.0, -135.0), # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),# Left mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        camera_matrix = self._get_camera_matrix(frame_dimensions)
        dist_coeffs = np.zeros((4, 1))

        (success, rotation_vector, translation_vector) = cv2.solvePnP(
            model_points, image_points, camera_matrix, dist_coeffs
        )

        if not success:
            print("Failed to solve PnP problem.")
            return None, None, None

        yaw, pitch, roll = get_yaw_pitch_roll(rotation_vector)
        
        if show:
            visualize_rotation_vector(frame, rotation_vector, translation_vector, 
                                     camera_matrix, dist_coeffs, image_points)
            cv2.imshow("Face Direction", frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return yaw, pitch, roll

    def direction(self, frame: np.ndarray, show=False):
        yaw, pitch, roll = self.orient(frame, show)
        if yaw is None:
            return None
        return Direction(yaw, pitch)
