import numpy as np
import json
from pathlib import Path

class FaceIdentifier:
    """Manages a database of face embeddings for recognition."""
    def __init__(self, db_path: str = "face_db.json", threshold: float = 0.5):
        self.db_path = Path(db_path)
        self.threshold = threshold
        self.known_faces = self._load_database()

    def _load_database(self) -> dict:
        """Loads the face database from a JSON file."""
        if self.db_path.exists():
            with open(self.db_path, 'r') as f:
                db = json.load(f)
                # Convert embedding lists back to numpy arrays
                for name, embeddings in db.items():
                    db[name] = [np.array(e) for e in embeddings]
                return db
        return {}

    def _save_database(self):
        """Saves the face database to a JSON file."""
        db_to_save = {}
        # Convert numpy arrays to lists for JSON serialization
        for name, embeddings in self.known_faces.items():
            db_to_save[name] = [e.tolist() for e in embeddings]
            
        with open(self.db_path, 'w') as f:
            json.dump(db_to_save, f, indent=4)

    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculates the cosine similarity between two embeddings."""
        dot_product = np.dot(emb1, emb2)
        norm_emb1 = np.linalg.norm(emb1)
        norm_emb2 = np.linalg.norm(emb2)
        return dot_product / (norm_emb1 * norm_emb2)

    def enroll(self, name: str, new_embedding: np.ndarray):
        """Adds a new face embedding to the database."""
        if name in self.known_faces:
            self.known_faces[name].append(new_embedding)
        else:
            self.known_faces[name] = [new_embedding]
        self._save_database()
        print(f"Enrolled new embedding for '{name}'.")

    def identify(self, unknown_embedding: np.ndarray) -> str:
        """
        Identifies a face by comparing its embedding to the known faces.
        Returns the name of the person or 'Unknown'.
        """
        best_match_name = "Unknown"
        highest_similarity = self.threshold

        for name, embeddings in self.known_faces.items():
            for known_embedding in embeddings:
                similarity = self._cosine_similarity(unknown_embedding, known_embedding)
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_name = name
        
        return best_match_name
