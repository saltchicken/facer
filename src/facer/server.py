from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from contextlib import asynccontextmanager
import cv2
import numpy as np
import uvicorn
import hashlib
import zipfile
import io
import ast
from typing import List, Dict, Tuple
from psycopg2 import Binary
import time


from facer.db import Database
from facer.face_detector import FaceDetector
from facer.face_direction import FaceDirection
from facer.face_embedder import FaceEmbedder
from facer.schemas import (
    AnalysisResponse,
    FaceData,
    FacePose,
    FaceUpdate,
)

# Global instances
detector = None
direction_finder = None
embedder = None
db = None

# Configuration Thresholds
YAW_THRESHOLD = 40.0
PITCH_THRESHOLD = 25.0
MATCH_THRESHOLD = 0.4



class EmbeddingCache:
    def __init__(self):
        self.known_embeddings: List[np.ndarray] = []
        self.known_names: List[str] = []
        self.last_updated = 0.0

    def load_from_db(self, db_instance):
        """Reloads the cache from Postgres."""
        if not db_instance:
            return
        try:
            print("🔄 Refreshing Embedding Cache...")
            start_t = time.time()
            with db_instance.get_cursor() as cur:

                cur.execute(
                    "SELECT classification, embedding FROM faces WHERE classification IS NOT NULL AND classification != '' AND embedding IS NOT NULL"
                )
                rows = cur.fetchall()

            new_embeddings = []
            new_names = []

            for classification, embedding_str in rows:
                try:

                    emb_list = ast.literal_eval(embedding_str)
                    new_embeddings.append(np.array(emb_list, dtype=np.float32))
                    new_names.append(classification)
                except Exception:
                    continue

            self.known_embeddings = new_embeddings
            self.known_names = new_names
            self.last_updated = time.time()
            print(
                f"✅ Cache refreshed: {len(self.known_names)} identities loaded in {time.time() - start_t:.3f}s"
            )
        except Exception as e:
            print(f"❌ Cache refresh failed: {e}")

    def add_identity(self, name: str, embedding: List[float]):
        """Incrementally update cache without full reload."""
        self.known_names.append(name)
        self.known_embeddings.append(np.array(embedding, dtype=np.float32))



face_cache = EmbeddingCache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    global detector, direction_finder, embedder, db
    print("Loading models...")
    detector = FaceDetector()
    direction_finder = FaceDirection()
    embedder = FaceEmbedder()

    try:
        db = Database.get_instance()
        print("Database connection pool initialized.")

        face_cache.load_from_db(db)
    except Exception as e:
        print(f"‼️ WARNING: Failed to connect to DB: {e}")

    print("Models loaded successfully.")

    yield  # Server runs here

    # --- Shutdown Logic ---
    print("Shutting down...")
    detector = None
    direction_finder = None
    embedder = None

    Database.close_pool()

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
    if not db:
        return None
    try:
        with db.get_cursor() as cur:
            # Check if we have any face records linked to a stored image with this hash
            cur.execute(
                """
                SELECT f.image_name 
                FROM faces f
                JOIN stored_images si ON f.stored_image_id = si.id
                WHERE si.hash = %s 
                LIMIT 1
                """,
                (file_hash,),
            )
            row = cur.fetchone()
            return row[0] if row else None
    except Exception as e:
        print(f"DB Check Error: {e}")
        return None


def identify_face_from_db(target_embedding: list) -> str:
    """
    Identifies a face using optimized Matrix Multiplication (Vectorization).
    ‼️ NOW USES IN-MEMORY CACHE
    """

    if not face_cache.known_embeddings or not target_embedding:
        return None

    try:
        # Create NumPy array from target and normalize immediately
        target_arr = np.array(target_embedding, dtype=np.float32)
        norm_target = np.linalg.norm(target_arr)

        if norm_target == 0:
            return None

        target_arr = target_arr / norm_target


        known_matrix = np.array(face_cache.known_embeddings)  # (N, 512)

        # Calculate L2 Norm for every row in the matrix
        norms = np.linalg.norm(known_matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Safety

        normalized_matrix = known_matrix / norms

        # Calculate Cosine Similarity via Dot Product
        similarities = np.dot(normalized_matrix, target_arr)

        # Find the index of the maximum score
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score > MATCH_THRESHOLD:

            return face_cache.known_names[best_idx]

        return None
    except Exception as e:
        print(f"Identification Error: {e}")
        return None


def save_to_db(
    filename,
    description,
    keywords,
    classification,
    faces_data,
    file_hash,
    original_image_bytes,
    width,
    height,
):
    if not db:
        return
    try:
        with db.get_cursor() as cur:
            # 1. Insert or Retrieve Image Blob ID
            cur.execute("SELECT id FROM stored_images WHERE hash = %s", (file_hash,))
            row = cur.fetchone()

            stored_image_id = None
            if row:
                stored_image_id = row[0]
            else:
                cur.execute(
                    "INSERT INTO stored_images (image_data, hash) VALUES (%s, %s) RETURNING id",
                    (Binary(original_image_bytes), file_hash),
                )
                stored_image_id = cur.fetchone()[0]

            # 2. Insert Faces linked to Blob ID
            faces_to_process = []
            if faces_data:
                # Logic to find best face remains same
                def get_face_score(f):
                    is_valid = 1 if f.is_valid_pose else 0
                    area = 0.0
                    if f.bbox and len(f.bbox) == 4:
                        w = f.bbox[2] - f.bbox[0]
                        h = f.bbox[3] - f.bbox[1]
                        area = w * h
                    return (is_valid, area)

                best_face = max(faces_data, key=get_face_score)
                faces_to_process = [best_face]

            for face_data in faces_to_process:
                embedding_val = (
                    str(face_data.embedding) if face_data.embedding else None
                )
                final_classification = classification or face_data.classification

                cur.execute(
                    """
                    INSERT INTO faces (
                        image_name, description, keywords, classification, bbox, yaw, pitch, roll, 
                        embedding, is_valid_pose, direction, stored_image_id,
                        width, height
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        filename,
                        description,
                        keywords,
                        final_classification,
                        face_data.bbox,
                        face_data.pose.yaw,
                        face_data.pose.pitch,
                        face_data.pose.roll,
                        embedding_val,
                        face_data.is_valid_pose,
                        face_data.pose.direction_label,
                        stored_image_id,
                        width,
                        height,
                    ),
                )


                if final_classification and face_data.embedding:
                    face_cache.add_identity(final_classification, face_data.embedding)

        print(f"✅ Saved faces to DB.")
    except Exception as e:
        print(f"❌ Database Error: {e}")


@app.patch("/faces/{face_id}")
def update_face_record(face_id: int, update: FaceUpdate):
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with db.get_cursor() as cur:
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

            values.append(face_id)
            query = f"UPDATE faces SET {', '.join(fields)} WHERE id = %s"

            cur.execute(query, tuple(values))

            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Face record not found")


        if update.classification is not None:
            face_cache.load_from_db(db)

        return {
            "status": "success",
            "id": face_id,
            "updated": update.dict(exclude_unset=True),
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/faces/{face_id}/reanalyze")
async def reanalyze_face(face_id: int):
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT si.image_data, f.bbox, f.classification 
                FROM faces f
                JOIN stored_images si ON f.stored_image_id = si.id
                WHERE f.id = %s
                """,
                (face_id,),
            )
            row = cur.fetchone()
            if not row or not row[0]:
                raise HTTPException(status_code=404, detail="Image not found")

            original_bytes = row[0]
            current_bbox = row[1]
            current_classification = row[2]

        nparr = np.frombuffer(original_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=500, detail="Failed to decode image")

        # 2. Run Detection (CPU Bound - run in thread)
        detections = await run_in_threadpool(detector.detect_and_crop, image)
        if not detections:
            return {"error": "No faces detected in original image"}

        # 3. Find matching face
        target_face = None
        target_bbox = None

        if current_bbox and len(current_bbox) == 4:
            cx_old = (current_bbox[0] + current_bbox[2]) / 2
            cy_old = (current_bbox[1] + current_bbox[3]) / 2
            min_dist = float("inf")
            for face_crop, bbox in detections:
                cx_new = (bbox[0] + bbox[2]) / 2
                cy_new = (bbox[1] + bbox[3]) / 2
                dist = ((cx_new - cx_old) ** 2 + (cy_new - cy_old) ** 2) ** 0.5
                if dist < min_dist:
                    min_dist = dist
                    target_face = face_crop
                    target_bbox = bbox
        else:
            target_face, target_bbox = detections[0]

        # 4. Analyze Pose
        direction_info = await run_in_threadpool(
            direction_finder.direction, target_face
        )
        yaw, pitch, roll = 0.0, 0.0, 0.0
        label = "unknown"
        if direction_info:
            yaw = float(direction_info.yaw)
            pitch = float(direction_info.pitch)
            label = str(direction_info)

        is_valid = bool(
            direction_info is not None
            and abs(yaw) < YAW_THRESHOLD
            and abs(pitch) < PITCH_THRESHOLD
        )

        # 5. Embedding
        embedding_val = None
        identified_classification = None
        if is_valid:
            emb_array = await run_in_threadpool(embedder.get_embedding, target_face)
            if emb_array.size > 0:
                embedding_val = str(emb_array.tolist())

                if not current_classification:
                    identified_classification = identify_face_from_db(
                        emb_array.tolist()
                    )

        # 6. Update Database
        with db.get_cursor() as cur:
            update_query = """
                UPDATE faces 
                SET bbox = %s, yaw = %s, pitch = %s, roll = %s, 
                    is_valid_pose = %s, direction = %s, embedding = %s
                WHERE id = %s
            """
            params = [
                target_bbox,
                yaw,
                pitch,
                roll,
                is_valid,
                label,
                embedding_val,
                face_id,
            ]

            if identified_classification:
                update_query = update_query.replace(
                    "WHERE", ", classification = %s WHERE"
                )
                params.insert(-1, identified_classification)

            cur.execute(update_query, tuple(params))


        if identified_classification and emb_array.size > 0:
            face_cache.add_identity(identified_classification, emb_array.tolist())

        return {
            "status": "success",
            "data": {
                "id": face_id,
                "bbox": target_bbox,
                "yaw": yaw,
                "pitch": pitch,
                "is_valid_pose": is_valid,
                "direction": label,
                "classification": identified_classification or current_classification,
            },
        }

    except Exception as e:
        print(f"Reanalyze Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/faces/{face_id}")
def delete_face(face_id: int):
    if not db:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        with db.get_cursor() as cur:
            cur.execute("DELETE FROM faces WHERE id = %s", (face_id,))
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Face record not found")


        face_cache.load_from_db(db)

        return {"status": "success", "message": f"Face {face_id} deleted successfully"}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/filters")
def get_filters():
    if not db:
        return {"classifications": [], "keywords": []}
    try:
        with db.get_cursor() as cur:
            cur.execute(
                "SELECT DISTINCT classification FROM faces WHERE classification IS NOT NULL AND classification != ''"
            )
            classifications = sorted([r[0] for r in cur.fetchall()])

            cur.execute("SELECT keywords FROM faces WHERE keywords IS NOT NULL")
            keyword_rows = cur.fetchall()
            unique_keywords = set()
            for row in keyword_rows:
                if row[0]:
                    parts = [p.strip() for p in row[0].split(",")]
                    unique_keywords.update(p for p in parts if p)

        return {
            "classifications": classifications,
            "keywords": sorted(list(unique_keywords)),
        }
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
    if not db:
        return []
    try:
        query = """
            SELECT id, image_name, is_valid_pose, yaw, pitch, roll, created_at, description, direction, keywords, classification, width, height
            FROM faces
        """
        params = []

        query, params = db.build_filter_query(query, params, keyword, classification)

        query += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
        params.extend([limit, offset])

        with db.get_cursor() as cur:
            cur.execute(query, tuple(params))
            rows = cur.fetchall()

            # Map columns to dict
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
        return faces
    except Exception as e:
        print(f"DB Error: {e}")
        return []


@app.get("/export")
def export_faces(
    keyword: list[str] = Query(None),
    classification: list[str] = Query(None),
):
    if not db:
        raise HTTPException(status_code=503, detail="Database not connected")
    try:
        query = """
            SELECT f.id, f.image_name, si.image_data 
            FROM faces f
            JOIN stored_images si ON f.stored_image_id = si.id
        """
        params = []

        query, params = db.build_filter_query(query, params, keyword, classification)
        query += " ORDER BY f.id ASC"

        zip_buffer = io.BytesIO()

        with db.get_cursor() as cur:
            cur.execute(query, tuple(params))

            with zipfile.ZipFile(
                zip_buffer, "a", zipfile.ZIP_DEFLATED, False
            ) as zip_file:
                for row in cur:
                    face_id, image_name, original_bytes = row
                    if not original_bytes:
                        continue
                    path = Path(image_name)
                    zip_filename = f"{path.stem}_{face_id}{path.suffix}"
                    zip_file.writestr(zip_filename, original_bytes)

        zip_buffer.seek(0)
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": "attachment; filename=exported_originals.zip"
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/faces/{face_id}/image")
def get_face_image(face_id: int):
    if not db:
        return Response(status_code=500, content="DB not connected")
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT si.image_data, f.bbox 
                FROM faces f
                JOIN stored_images si ON f.stored_image_id = si.id
                WHERE f.id = %s
                """,
                (face_id,),
            )
            row = cur.fetchone()

            if row and row[0]:
                original_bytes = row[0]
                bbox = row[1]
                nparr = np.frombuffer(original_bytes, np.uint8)
                full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if full_image is None:
                    return Response(
                        status_code=500, content="Failed to decode stored image"
                    )

                if not bbox:
                    face_crop = full_image
                elif detector:
                    face_crop = detector._crop_and_center_face(full_image, bbox)
                else:
                    x1, y1, x2, y2 = map(int, bbox)
                    face_crop = full_image[y1:y2, x1:x2]

                success, buffer = cv2.imencode(".jpg", face_crop)
                return (
                    Response(content=buffer.tobytes(), media_type="image/jpeg")
                    if success
                    else Response(status_code=500)
                )
            return Response(status_code=404)
    except Exception as e:
        print(f"DB Error: {e}")
        return Response(status_code=500)


@app.get("/faces/{face_id}/full_image")
def get_face_full_image(face_id: int):
    if not db:
        return Response(status_code=500, content="DB not connected")
    try:
        with db.get_cursor() as cur:
            cur.execute(
                """
                SELECT si.image_data 
                FROM faces f
                JOIN stored_images si ON f.stored_image_id = si.id
                WHERE f.id = %s
                """,
                (face_id,),
            )
            row = cur.fetchone()
            if row and row[0]:
                nparr = np.frombuffer(row[0], np.uint8)
                full_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                success, buffer = cv2.imencode(".jpg", full_image)
                return (
                    Response(content=buffer.tobytes(), media_type="image/jpeg")
                    if success
                    else Response(status_code=500)
                )
            return Response(status_code=404)
    except Exception:
        return Response(status_code=500)


# Sync worker for analysis
def process_analysis_sync(
    image, save_flag, filename, desc, keys, classif, file_hash, contents, width, height
):
    detections = detector.detect_and_crop(image)
    results = []

    for i, (face_crop, bbox) in enumerate(detections):
        direction_info = direction_finder.direction(face_crop)
        yaw, pitch, roll = 0.0, 0.0, 0.0
        label = "unknown"

        if direction_info:
            yaw = direction_info.yaw
            pitch = direction_info.pitch
            label = str(direction_info)

        is_valid = (
            direction_info is not None
            and abs(yaw) < YAW_THRESHOLD
            and abs(pitch) < PITCH_THRESHOLD
        )
        embedding_vector = []
        identified_classification = None

        if is_valid:
            emb_array = embedder.get_embedding(face_crop)
            if emb_array.size > 0:
                embedding_vector = emb_array.tolist()
                if not classif:
                    identified_classification = identify_face_from_db(embedding_vector)

        face_data = FaceData(
            bbox=bbox,
            pose=FacePose(yaw=yaw, pitch=pitch, roll=roll, direction_label=label),
            is_valid_pose=is_valid,
            embedding=embedding_vector if embedding_vector else None,
            classification=identified_classification,
        )
        results.append(face_data)

    if save_flag:
        save_to_db(
            filename, desc, keys, classif, results, file_hash, contents, width, height
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
        try:
            contents = await file.read()
            file_hash = hashlib.sha256(contents).hexdigest()

            # Check for existing image via hash
            existing_name = check_image_exists(file_hash)
            if save and existing_name:
                all_responses.append(
                    AnalysisResponse(
                        filename=file.filename,
                        face_count=0,
                        results=[],
                        error=f"Duplicate: {existing_name}",
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