import cv2
import numpy as np
import mediapipe as mp


def draw_line(frame, a, b, color=(255, 255, 0)):
    cv2.line(frame, a, b, color, 10)


def get_yaw_pitch_roll(rotation_vector):
    """Converts a rotation vector to yaw, pitch, and roll angles."""
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)

    pitch = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
    yaw = np.arctan2(
        -rotation_matrix[2, 0],
        np.sqrt(rotation_matrix[2, 1] ** 2 + rotation_matrix[2, 2] ** 2),
    )
    roll = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    return np.degrees(yaw), np.degrees(pitch), np.degrees(roll)


def visualize_rotation_vector(
    frame, rotation_vector, translation_vector, camera_matrix, dist_coeffs, image_points
):
    """Draws a 3D coordinate system on the face."""
    axis_length = 100.0
    axis_points_3D = np.array(
        [
            (axis_length, 0, 0),  # X-axis (Red)
            (0, axis_length, 0),  # Y-axis (Green)
            (0, 0, axis_length),  # Z-axis (Blue)
        ],
        dtype="double",
    )

    projected_points, _ = cv2.projectPoints(
        axis_points_3D, rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    nose_tip = tuple(map(int, image_points[0]))
    x_axis = tuple(map(int, projected_points[0].ravel()))
    y_axis = tuple(map(int, projected_points[1].ravel()))
    z_axis = tuple(map(int, projected_points[2].ravel()))

    cv2.line(frame, nose_tip, x_axis, (0, 0, 255), 3)  # X-axis
    cv2.line(frame, nose_tip, y_axis, (0, 255, 0), 3)  # Y-axis
    cv2.line(frame, nose_tip, z_axis, (255, 0, 0), 3)  # Z-axis


def plot_selected_landmarks(frame, landmarks):
    """Plots selected facial landmarks as circles on the frame."""
    landmark_indices = [4, 152, 263, 33, 291, 61]
    for index in landmark_indices:
        x = int(landmarks.landmark[index].x * frame.shape[1])
        y = int(landmarks.landmark[index].y * frame.shape[0])
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)



def apply_image_filter(image: np.ndarray, script_name: str) -> np.ndarray:
    """
    Applies a specific CV2 operation or script to the image.
    """
    if script_name == "grayscale":
        # Convert to grayscale and back to BGR so it remains a 3-channel image for consistency
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    elif script_name == "edges":
        # Canny edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        # Invert edges (black on white) for better visibility
        edges = cv2.bitwise_not(edges)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    elif script_name == "invert":
        return cv2.bitwise_not(image)

    elif script_name == "remove_bg":

        try:
            from rembg import remove

            # rembg expects RGB/BGR but usually returns RGBA
            # We need to handle the alpha channel for it to look right
            result = remove(image)

            # If result has alpha channel, composite it over white or black
            if result.shape[2] == 4:
                # Create white background
                bg = np.ones_like(result[:, :, :3]) * 255
                alpha = result[:, :, 3] / 255.0

                for c in range(3):
                    bg[:, :, c] = result[:, :, c] * alpha + bg[:, :, c] * (1 - alpha)

                return bg.astype(np.uint8)
            return result
        except ImportError:
            print("⚠️ 'rembg' not installed. Returning original image.")
            # Fallback for demo purposes if library is missing
            return image

    return image