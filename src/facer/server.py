from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
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
import zipfile
import io
from typing import List
from dotenv import load_dotenv
from psycopg2 import Binary
from facer.face_detector import FaceDetector
from facer.face_direction import FaceDirection
from facer.face_embedder import FaceEmbedder
from facer.schemas import (
    AnalysisResponse,
    FaceData,
    FacePose,
    FaceUpdate,
)

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


@app.patch("/faces/{face_id}")
def update_face_record(face_id: int, update: FaceUpdate):
    if not DB_URL:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                # Build dynamic query based on what fields were sent
                fields = []
                values = []

                if update.description is not None:
                    fields.append("description = %s")
                    values.append(update.description)

                if update.classification is not None:
                    fields.append("classification = %s")
                    values.append(update.classification)

                if update.keywords is not None:
                    fields.append("keywords = %s")
                    values.append(update.keywords)

                if not fields:
                    return {"message": "No fields to update"}

                # Add ID for WHERE clause
                values.append(face_id)

                query = f"UPDATE faces SET {', '.join(fields)} WHERE id = %s"

                cur.execute(query, tuple(values))

                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Face record not found")

        conn.close()
        return {
            "status": "success",
            "id": face_id,
            "updated": update.dict(exclude_unset=True),
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Update Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/faces/{face_id}")
def delete_face(face_id: int):
    if not DB_URL:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        conn = psycopg2.connect(DB_URL)
        with conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM faces WHERE id = %s", (face_id,))

                if cur.rowcount == 0:
                    raise HTTPException(status_code=404, detail="Face record not found")

        conn.close()
        return {"status": "success", "message": f"Face {face_id} deleted successfully"}

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Delete Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filters")
def get_filters():
    if not DB_URL:
        return {"classifications": [], "keywords": []}
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            # 1. Get distinct classifications
            cur.execute(
                "SELECT DISTINCT classification FROM faces WHERE classification IS NOT NULL AND classification != ''"
            )
            class_rows = cur.fetchall()
            classifications = sorted([r[0] for r in class_rows])

            # 2. Get all keywords to parse distinct ones (assuming CSV storage)
            cur.execute("SELECT keywords FROM faces WHERE keywords IS NOT NULL")
            keyword_rows = cur.fetchall()

            unique_keywords = set()
            for row in keyword_rows:
                # row[0] is like "front, daylight"
                if row[0]:
                    parts = [p.strip() for p in row[0].split(",")]
                    unique_keywords.update(p for p in parts if p)

            keywords = sorted(list(unique_keywords))

        conn.close()
        return {"classifications": classifications, "keywords": keywords}
    except Exception as e:
        print(f"Filter Fetch Error: {e}")
        return {"classifications": [], "keywords": []}


@app.get("/faces")
def get_faces(
    limit: int = 100,
    offset: int = 0,
    keyword: list[str] = Query(None),
    classification: list[str] = Query(None),
):
    if not DB_URL:
        return []
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            query = """
                SELECT id, image_name, is_valid_pose, yaw, pitch, roll, created_at, description, direction, keywords, classification, width, height
                FROM faces
            """
            conditions = []
            params = []

            # Updated logic for multiple Classifications (OR logic)
            if classification:
                # Check if we need to filter for NULLs separately from strings
                has_none = "__NONE__" in classification
                real_classes = [c for c in classification if c != "__NONE__"]

                class_sub_conditions = []

                # Handle standard string matches
                if real_classes:
                    # Postgres ANY syntax for arrays
                    class_sub_conditions.append("classification = ANY(%s)")
                    params.append(real_classes)

                # Handle Unclassified/NULL matches
                if has_none:
                    class_sub_conditions.append(
                        "(classification IS NULL OR classification = '')"
                    )

                if class_sub_conditions:
                    conditions.append(f"({' OR '.join(class_sub_conditions)})")

            # Updated logic for multiple Keywords (OR logic)
            if keyword:
                has_none = "__NONE__" in keyword
                real_keywords = [k for k in keyword if k != "__NONE__"]

                keyword_sub_conditions = []

                if real_keywords:
                    # Construct multiple ILIKE statements OR'd together
                    # We can't use ANY with ILIKE easily without UNNEST, so explicit OR is safer/simpler here
                    likes = []
                    for k in real_keywords:
                        likes.append("keywords ILIKE %s")
                        params.append(f"%{k}%")
                    keyword_sub_conditions.append(f"({' OR '.join(likes)})")

                if has_none:
                    keyword_sub_conditions.append("(keywords IS NULL OR keywords = '')")

                if keyword_sub_conditions:
                    conditions.append(f"({' OR '.join(keyword_sub_conditions)})")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            cur.execute(query, tuple(params))

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
                        "width": row[11],
                        "height": row[12],
                    }
                )
        conn.close()
        return faces
    except Exception as e:
        print(f"DB Error: {e}")
        return []


@app.get("/export")
def export_faces(
    keyword: list[str] = Query(None),
    classification: list[str] = Query(None),
):
    if not DB_URL:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        conn = psycopg2.connect(DB_URL)
        # Use a dictionary cursor if possible, but standard is fine since we know column order
        with conn.cursor() as cur:
            query = """
                SELECT id, image_name, original_image
                FROM faces
            """
            conditions = []
            params = []

            # --- Replicate Filter Logic (Same as get_faces) ---
            if classification:
                has_none = "__NONE__" in classification
                real_classes = [c for c in classification if c != "__NONE__"]
                class_sub_conditions = []
                if real_classes:
                    class_sub_conditions.append("classification = ANY(%s)")
                    params.append(real_classes)
                if has_none:
                    class_sub_conditions.append(
                        "(classification IS NULL OR classification = '')"
                    )
                if class_sub_conditions:
                    conditions.append(f"({' OR '.join(class_sub_conditions)})")

            if keyword:
                has_none = "__NONE__" in keyword
                real_keywords = [k for k in keyword if k != "__NONE__"]
                keyword_sub_conditions = []
                if real_keywords:
                    likes = []
                    for k in real_keywords:
                        likes.append("keywords ILIKE %s")
                        params.append(f"%{k}%")
                    keyword_sub_conditions.append(f"({' OR '.join(likes)})")
                if has_none:
                    keyword_sub_conditions.append("(keywords IS NULL OR keywords = '')")
                if keyword_sub_conditions:
                    conditions.append(f"({' OR '.join(keyword_sub_conditions)})")

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            # Order by ID to keep it deterministic
            query += " ORDER BY id ASC"

            cur.execute(query, tuple(params))

            # Use BytesIO to build zip in memory
            zip_buffer = io.BytesIO()

            with zipfile.ZipFile(
                zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zip_file:
                for row in cur:
                    face_id = row[0]
                    image_name = row[1]
                    original_bytes = row[2]

                    if not original_bytes:
                        continue

                    # Create a unique filename: originalname_id.ext
                    path = Path(image_name)
                    zip_filename = f"{path.stem}_{face_id}{path.suffix}"

                    zip_file.writestr(zip_filename, original_bytes)

        conn.close()

        # Reset buffer position
        zip_buffer.seek(0)

        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=exported_originals.zip"
            },
        )

    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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

            # This allows images with empty bbox (no faces) to be processed.
            if row and row[0]:
                original_bytes = row[0]
                bbox = row[1]  # Expected [x1, y1, x2, y2]

                nparr = np.frombuffer(original_bytes, np.uint8)
                full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if full_image is None:
                    return Response(
                        status_code=500, content="Failed to decode stored image"
                    )

                # We use the internal method _crop_and_center_face from the global detector instance
                # If bbox is empty (from a no-face save), return full image or handle gracefully

                if not bbox:
                    face_crop = full_image
                elif detector:
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


@app.get("/faces/{face_id}/full_image")
def get_face_full_image(face_id: int):
    if not DB_URL:
        return Response(status_code=500, content="DB not connected")
    try:
        conn = psycopg2.connect(DB_URL)
        with conn.cursor() as cur:
            cur.execute("SELECT original_image FROM faces WHERE id = %s", (face_id,))
            row = cur.fetchone()

            if row and row[0]:
                original_bytes = row[0]
                # Decode to ensure validity and allow re-encoding to standard JPEG
                nparr = np.frombuffer(original_bytes, np.uint8)
                full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if full_image is None:
                    return Response(status_code=500, content="Failed to decode")

                # Encode to JPEG to ensure consistent content type
                success, buffer = cv2.imencode(".jpg", full_image)
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

    # Logic is now handled below to save a fallback record instead.

    results = []

    # Process detections as normal for API response
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

    # 7. Save (Only if requested)
    if save_flag:
        faces_to_save = []

        if (
            len(results) >= 1
        ):
            faces_to_save = results
        else:
            # Contains no embedding, empty bbox, null pose, invalid status.

            class FallbackPose:
                yaw = None
                pitch = None
                roll = None
                direction_label = None

            class FallbackData:
                bbox = []  # Empty array for Postgres
                pose = FallbackPose()
                is_valid_pose = False
                embedding = None

            faces_to_save = [FallbackData()]

        if faces_to_save:
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


@app.post("/analyze", response_model=List[AnalysisResponse])
async def analyze_image(
    files: List[UploadFile] = File(...),
    description: str = Form(None),
    keywords: str = Form(None),
    classification: str = Form(None),
    save: bool = Form(False),
):
    all_responses = []

    for file in files:
        # 1. Read Image
        try:
            contents = await file.read()

            file_hash = hashlib.sha256(contents).hexdigest()
            error_msg = None

            if save:
                # Check duplicate per file
                existing_name = check_image_exists(file_hash)
                if existing_name:

                    error_msg = f"Duplicate: Already exists as '{existing_name}'."

            if error_msg:
                all_responses.append(
                    AnalysisResponse(
                        filename=file.filename,
                        face_count=0,
                        results=[],
                        error=error_msg,
                    )
                )
                continue

            nparr = np.frombuffer(contents, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                all_responses.append(
                    AnalysisResponse(
                        filename=file.filename,
                        face_count=0,
                        results=[],
                        error="Could not decode image",
                    )
                )
                continue

            height, width = image.shape[:2]

            # Run analysis
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

            all_responses.append(
                AnalysisResponse(
                    filename=file.filename, face_count=len(results), results=results
                )
            )

        except Exception as e:

            print(f"Error processing {file.filename}: {e}")
            all_responses.append(
                AnalysisResponse(
                    filename=file.filename, face_count=0, results=[], error=str(e)
                )
            )

    return all_responses


FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "facer-ui" / "dist"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("facer.server:app", host="0.0.0.0", port=8000, reload=False)