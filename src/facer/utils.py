import cv2
import numpy as np
import mediapipe as mp

def draw_line(frame, a, b, color=(255, 255, 0)):
    cv2.line(frame, a, b, color, 10)

def get_yaw_pitch_roll(rotation_vector):
    """Converts a rotation vector to yaw, pitch, and roll angles."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    
    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = np.arctan2(-rotation_matrix[2, 0], np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2))
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)

def visualize_rotation_vector(frame, rotation_vector, translation_vector, 
                              camera_matrix, dist_coeffs, image_points):
    """Draws a 3D coordinate system on the face."""
    axis_length = 100.0
    axis_points_3D = np.array([
        (axis_length, 0, 0),    # X-axis (Red)
        (0, axis_length, 0),    # Y-axis (Green)
        (0, 0, axis_length)     # Z-axis (Blue)
    ], dtype="double")

    projected_points, _ = cv2.projectPoints(
        axis_points_3D, rotation_vector, translation_vector, 
        camera_matrix, dist_coeffs
    )

    nose_tip = tuple(map(int, image_points[0]))
    x_axis = tuple(map(int, projected_points[0].ravel()))
    y_axis = tuple(map(int, projected_points[1].ravel()))
    z_axis = tuple(map(int, projected_points[2].ravel()))

    cv2.line(frame, nose_tip, x_axis, (0, 0, 255), 3) # X-axis
    cv2.line(frame, nose_tip, y_axis, (0, 255, 0), 3) # Y-axis
    cv2.line(frame, nose_tip, z_axis, (255, 0, 0), 3) # Z-axis

def plot_selected_landmarks(frame, landmarks):
    """Plots selected facial landmarks as circles on the frame."""
    landmark_indices = [4, 152, 263, 33, 291, 61]
    for index in landmark_indices:
        x = int(landmarks.landmark[index].x * frame.shape[1])
        y = int(landmarks.landmark[index].y * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

