from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from contextlib import asynccontextmanager
import cv2
import numpy as np
import uvicorn
import psycopg2
import hashlib
import os
from dotenv import load_dotenv
from psycopg2 import Binary
from facer.face_detector import FaceDetector
from facer.face_direction import FaceDirection
from facer.face_embedder import FaceEmbedder
from facer.schemas import AnalysisResponse, FaceData, FacePose

load_dotenv()

# TODO: Fix this for production
DB_URL = os.getenv("DB_URL")
if not DB_URL:
    raise ValueError("Missing DB_URL environment variable. Please set it in the .env file.")

# Global instances
detector = None
direction_finder = None
embedder = None

# Configuration Thresholds
YAW_THRESHOLD = 25.0
PITCH_THRESHOLD = 25.0

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    global detector, direction_finder, embedder
    print("Loading models...")
    detector = FaceDetector()
    direction_finder = FaceDirection()
    embedder = FaceEmbedder()
    print("Models loaded successfully.")
    
    yield # Server runs here
    
    # --- Shutdown Logic ---
    print("Shutting down...")
    detector = None
    direction_finder = None
    embedder = None
    import gc
    gc.collect()

app = FastAPI(
    title="Facer Service", 
    description="Face Analysis API and Static File Server",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def check_image_exists(file_hash: str):
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT image_name FROM faces WHERE source_image_hash = %s LIMIT 1", (file_hash,))
            row = cur.fetchone()
            existing_name = row[0] if row else None
        conn.close()
        return existing_name
    except Exception as e:
        print(f"DB Check Error: {e}")
        return None


def save_to_db(filename: str, description: str, keywords: str, classification: str, faces_with_images: list, file_hash: str):
    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                for face_data, face_img_bytes in faces_with_images:
                    embedding_val = str(face_data.embedding) if face_data.embedding else None
                    

                    cur.execute("""
                        INSERT INTO faces (
                            image_name, description, keywords, classification, bbox, yaw, pitch, roll, 
                            embedding, is_valid_pose, face_image, direction, source_image_hash
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        filename, 
                        description,
                        keywords,
                        classification,
                        face_data.bbox, 
                        face_data.pose.yaw, 
                        face_data.pose.pitch, 
                        face_data.pose.roll, 
                        embedding_val, 
                        face_data.is_valid_pose,
                        Binary(face_img_bytes) if face_img_bytes else None,
                        face_data.pose.direction_label,
                        file_hash
                    ))
        print(f"✅ Saved {len(faces_with_images)} faces to DB.")
        conn.close()
    except Exception as e:
        print(f"❌ Database Error: {e}")

@app.get("/faces")
def get_faces(limit: int = 100, offset: int = 0):
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:

            cur.execute("""
                SELECT id, image_name, is_valid_pose, yaw, pitch, roll, created_at, description, direction, keywords, classification
                FROM faces
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """, (limit, offset))
            
            rows = cur.fetchall()
            faces = []
            for row in rows:
                faces.append({
                    "id": row[0],
                    "image_name": row[1],
                    "is_valid_pose": row[2],
                    "yaw": row[3],
                    "pitch": row[4],
                    "roll": row[5],
                    "created_at": row[6],
                    "description": row[7],
                    "direction": row[8],
                    "keywords": row[9],
                    "classification": row[10]
                })
        conn.close()
        return faces
    except Exception as e:
        print(f"DB Error: {e}")
        return []

@app.get("/faces/{face_id}/image")
def get_face_image(face_id: int):
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT face_image FROM faces WHERE id = %s", (face_id,))
            row = cur.fetchone()
            
            if row and row[0]:
                return Response(content=row[0], media_type="image/jpeg")
            else:
                return Response(status_code=404)
    except Exception as e:
        print(f"DB Error: {e}")
        return Response(status_code=500)

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    description: str = Form(None),
    keywords: str = Form(None),
    classification: str = Form(None),
    save: bool = Form(False)
):
    # 1. Read Image
    try:
        contents = await file.read()
        
        file_hash = hashlib.sha256(contents).hexdigest()

        if save:
            existing_name = check_image_exists(file_hash)
            if existing_name:
                raise HTTPException(
                    status_code=409, 
                    detail=f"Duplicate Image: This image is already in the database as '{existing_name}'."
                )

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None: raise ValueError("Could not decode")
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. Detect
    detections = detector.detect_and_crop(image)
    results = []
    db_payload = []

    for i, (face_crop, bbox) in enumerate(detections):
        # 3. Direction
        direction_info = direction_finder.direction(face_crop)
        yaw, pitch, roll = 0.0, 0.0, 0.0
        label = "unknown"
        
        if direction_info:
            yaw = direction_info.yaw
            pitch = direction_info.pitch
            label = str(direction_info)

        # 4. Validity
        is_valid = (
            direction_info is not None 
            and abs(yaw) < YAW_THRESHOLD 
            and abs(pitch) < PITCH_THRESHOLD
        )

        # 5. Embedding
        embedding_vector = []
        if is_valid:
            emb_array = embedder.get_embedding(face_crop)
            if emb_array.size > 0:
                embedding_vector = emb_array.tolist()

        # 6. Build Object
        face_data = FaceData(
            bbox=bbox,
            pose=FacePose(yaw=yaw, pitch=pitch, roll=roll, direction_label=label),
            is_valid_pose=is_valid,
            embedding=embedding_vector if embedding_vector else None
        )
        
        success, buffer = cv2.imencode('.jpg', face_crop)
        face_bytes = buffer.tobytes() if success else None
        
        results.append(face_data)
        db_payload.append((face_data, face_bytes))

    # 7. Save (Only if requested)
    if save and db_payload:

        save_to_db(file.filename, description, keywords, classification, db_payload, file_hash)

    return AnalysisResponse(
        filename=file.filename,
        face_count=len(results),
        results=results
    )

FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "facer-ui" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("facer.server:app", host="0.0.0.0", port=8000, reload=True)
