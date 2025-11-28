from fastapi import FastAPI, File, UploadFile, HTTPException
import cv2
import numpy as np
import uvicorn
from facer.face_detector import FaceDetector
from facer.face_direction import FaceDirection
from facer.face_embedder import FaceEmbedder
from facer.schemas import AnalysisResponse, FaceData, FacePose

app = FastAPI(title="Facer Service", description="Face Analysis API for PostgreSQL storage")

# Global instances (Lazy loaded on startup)
detector = None
direction_finder = None
embedder = None

# Configuration Thresholds
YAW_THRESHOLD = 25.0
PITCH_THRESHOLD = 25.0

@app.on_event("startup")
async def load_models():
    """Load models once when server starts to save time per request."""
    global detector, direction_finder, embedder
    print("Loading models...")
    detector = FaceDetector()
    direction_finder = FaceDirection()
    embedder = FaceEmbedder()
    print("Models loaded successfully.")

# ‼️ New shutdown event to help release resources gracefully
@app.on_event("shutdown")
async def shutdown_event():
    """Clear global models on shutdown to prevent resource leaks."""
    global detector, direction_finder, embedder
    print("Shutting down and cleaning up models...")
    
    # Clear references
    detector = None
    direction_finder = None
    embedder = None
    
    # Force garbage collection to release PyTorch/OpenCV locks
    import gc
    gc.collect()
    print("Cleanup complete.")

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(file: UploadFile = File(...)):
    """
    Receives an image, detects faces, checks pose, and generates embeddings.
    Returns JSON structure ready for database insertion.
    """
    
    # 1. Read Image File
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")

    # 2. Detect Faces
    # ‼️ Now unpacks (crop, bbox) instead of just crop
    detections = detector.detect_and_crop(image)
    
    results = []

    for i, (face_crop, bbox) in enumerate(detections):
        
        # 3. Analyze Direction
        direction_info = direction_finder.direction(face_crop)
        
        # Default values if direction fails
        yaw, pitch, roll = 0.0, 0.0, 0.0
        label = "unknown"
        
        if direction_info:
            yaw = direction_info.yaw
            pitch = direction_info.pitch
            # We recover roll if available, otherwise 0
            # Note: your Direction class stores yaw/pitch, but orient() returns roll too.
            # Ideally FaceDirection.direction() should return roll too, but we work with what we have.
            label = str(direction_info)

        # 4. Check Validity
        is_valid = (
            direction_info is not None 
            and abs(yaw) < YAW_THRESHOLD 
            and abs(pitch) < PITCH_THRESHOLD
        )

        # 5. Generate Embedding (Only if valid, or force generation if you prefer)
        embedding_vector = []
        if is_valid:
            emb_array = embedder.get_embedding(face_crop)
            if emb_array.size > 0:
                embedding_vector = emb_array.tolist() # Convert numpy to list for JSON

        # 6. Build Result Object
        face_data = FaceData(
            bbox=bbox,
            pose=FacePose(
                yaw=yaw, 
                pitch=pitch, 
                roll=roll, 
                direction_label=label
            ),
            is_valid_pose=is_valid,
            embedding=embedding_vector if embedding_vector else None
        )
        
        results.append(face_data)

    return AnalysisResponse(
        filename=file.filename,
        face_count=len(results),
        results=results
    )

if __name__ == "__main__":
    uvicorn.run("facer.server:app", host="0.0.0.0", port=8000, reload=True)
