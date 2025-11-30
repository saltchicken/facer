from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import (
    run_in_threadpool,
)
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

DB_URL = os.getenv("DB_URL")
if not DB_URL:
    print("WARNING: DB_URL is not set. Database features will fail.")

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

    yield  # Server runs here

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
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def check_image_exists(file_hash: str):
    if not DB_URL:
        return None
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT image_name FROM faces WHERE source_image_hash = %s LIMIT 1",
                (file_hash,),
            )
            row = cur.fetchone()
            existing_name = row[0] if row else None
        conn.close()
        return existing_name
    except Exception as e:
        print(f"DB Check Error: {e}")
        return None


def save_to_db(
    filename: str,
    description: str,
    keywords: str,
    classification: str,
    faces_data: list,
    file_hash: str,
    original_image_bytes: bytes,
    width: int,
    height: int,
):
    if not DB_URL:
        return
    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                for face_data in faces_data:
                    embedding_val = (
                        str(face_data.embedding) if face_data.embedding else None
                    )

                    cur.execute(
                        """
                        INSERT INTO faces (
                            image_name, description, keywords, classification, bbox, yaw, pitch, roll, 
                            embedding, is_valid_pose, direction, source_image_hash,
                            original_image, width, height
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                        (
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
                            face_data.pose.direction_label,
                            file_hash,
                            Binary(original_image_bytes),
                            width,
                            height,
                        ),
                    )
        print(f"✅ Saved {len(faces_data)} faces to DB.")
        conn.close()
    except Exception as e:
        print(f"❌ Database Error: {e}")


@app.get("/faces")
def get_faces(limit: int = 100, offset: int = 0):
    if not DB_URL:
        return []
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, image_name, is_valid_pose, yaw, pitch, roll, created_at, description, direction, keywords, classification
                FROM faces
                ORDER BY created_at DESC
                LIMIT %s OFFSET %s
            """,
                (limit, offset),
            )

            rows = cur.fetchall()
            faces = []
            for row in rows:
                faces.append(
                    {
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
                        "classification": row[10],
                    }
                )
        conn.close()
        return faces
    except Exception as e:
        print(f"DB Error: {e}")
        return []


@app.get("/faces/{face_id}/image")
def get_face_image(face_id: int):
    if not DB_URL:
        return Response(status_code=500, content="DB not connected")
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT original_image, bbox FROM faces WHERE id = %s", (face_id,)
            )
            row = cur.fetchone()

            if row and row[0] and row[1]:
                original_bytes = row[0]
                bbox = row[1]  # Expected [x1, y1, x2, y2]

                nparr = np.frombuffer(original_bytes, np.uint8)
                full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if full_image is None:
                    return Response(
                        status_code=500, content="Failed to decode stored image"
                    )

                # We use the internal method _crop_and_center_face from the global detector instance
                if detector:
                    face_crop = detector._crop_and_center_face(full_image, bbox)
                else:
                    # Fallback if detector isn't loaded (unlikely) -> Simple crop
                    x1, y1, x2, y2 = map(int, bbox)
                    face_crop = full_image[y1:y2, x1:x2]

                # Encode to JPEG
                success, buffer = cv2.imencode(".jpg", face_crop)
                if success:
                    return Response(content=buffer.tobytes(), media_type="image/jpeg")
                else:
                    return Response(status_code=500)
            else:
                return Response(status_code=404)
    except Exception as e:
        print(f"DB Error: {e}")
        return Response(status_code=500)


def process_analysis_sync(
    image, save_flag, filename, desc, keys, classif, file_hash, contents, width, height
):
    # 2. Detect
    detections = detector.detect_and_crop(image)


    if len(detections) > 1:
        if save_flag:
            print(
                f"⚠️  Multiple faces detected ({len(detections)}). Saving disabled for '{filename}'."
            )
        save_flag = False

    results = []
    faces_to_save = []

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
            embedding=embedding_vector if embedding_vector else None,
        )

        results.append(face_data)
        faces_to_save.append(face_data)

    # 7. Save (Only if requested)
    if save_flag and faces_to_save:
        save_to_db(
            filename,
            desc,
            keys,
            classif,
            faces_to_save,
            file_hash,
            contents,
            width,
            height,
        )

    return results


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    description: str = Form(None),
    keywords: str = Form(None),
    classification: str = Form(None),
    save: bool = Form(False),
):
    # 1. Read Image
    try:
        contents = await file.read()

        file_hash = hashlib.sha256(contents).hexdigest()

        if save:
            # but optimally could be awaited if converted to async.
            # For now, it's fast enough to leave or wrap.
            existing_name = check_image_exists(file_hash)
            if existing_name:
                raise HTTPException(
                    status_code=409,
                    detail=f"Duplicate Image: This image is already in the database as '{existing_name}'.",
                )

        nparr = np.frombuffer(contents, np.uint8)

        # but usually negligible compared to ML models.
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode")

        height, width = image.shape[:2]

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # This prevents the async event loop from blocking while YOLO/MediaPipe run.
    results = await run_in_threadpool(
        process_analysis_sync,
        image,
        save,
        file.filename,
        description,
        keywords,
        classification,
        file_hash,
        contents,
        width,
        height,
    )

    return AnalysisResponse(
        filename=file.filename, face_count=len(results), results=results
    )


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "facer-ui" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("facer.server:app", host="0.0.0.0", port=8000, reload=True)
