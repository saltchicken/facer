import numpy as np
import cv2
import os
from onnxruntime import InferenceSession

class FaceEmbedder:
    """A class to generate face embeddings using a pre-trained ONNX model."""

    _session = None

    def __init__(self):
        """Initializes the FaceEmbedder and loads the ONNX model."""
        if FaceEmbedder._session is None:
            self._load_model()

    def _load_model(self):
        """Loads and caches the ONNX model for face embedding from a local path."""
        try:
            # Construct the local path to the ONNX model
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "w600k_r50.onnx")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"ONNX model not found at {model_path}. Please download it manually.")

            self._session = InferenceSession(model_path)
        except Exception as e:
            print(f"Error loading face embedding model: {e}")
            raise

    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Calculates the embedding of a single face.

        Args:
            face_image: A cropped face image as a NumPy array (H, W, 3).

        Returns:
            A NumPy array representing the face embedding.
            Returns an empty NumPy array if an error occurs.
        """
        try:
            # Preprocess the image for the model
            input_size = self._session.get_inputs()[0].shape[2:]
            resized_image = cv2.resize(face_image, input_size)
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
            normalized_image = (rgb_image - 127.5) / 128.0
            input_tensor = np.expand_dims(normalized_image.transpose(2, 0, 1), axis=0).astype(np.float32)

            # Get the embedding from the ONNX session
            output = self._session.run(None, {self._session.get_inputs()[0].name: input_tensor})
            embedding = output[0].flatten()

            return embedding
        except Exception as e:
            print(f"Error generating face embedding: {e}")
            return np.array([])
