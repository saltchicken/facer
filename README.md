# Facer

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-green)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-14%2B-blue)

**Facer** is a full-stack computer vision application designed to detect, analyze, enroll, and identify faces. It combines modern deep learning models for detection and recognition with a responsive React UI and a PostgreSQL backend for data persistence.

## 🚀 Features

- **High-Performance Detection**: Uses `YOLOv11` (via Ultralytics) for fast and accurate face detection.
- **Pose Estimation**: leveraging `MediaPipe Face Mesh` to calculate 3D head orientation (Yaw, Pitch, Roll).
  - Automatically filters out faces with extreme angles to ensure high-quality enrollment.
- **Face Recognition**: Generates 512-dimensional embeddings using `InsightFace` (ONNX Runtime) for identity verification.
- **Vector Identification**: Identifies individuals using Cosine Similarity matching against a known database.
- **Dual Interface**:
  - **CLI**: A command-line tool for local image analysis and quick enrollment.
  - **Web Dashboard**: A React-based UI for uploading images, visualizing bounding boxes/pose validity, and browsing the face gallery.
- **Database Integration**: Stores face metadata, embeddings, and image blobs in **PostgreSQL**.

## 🛠 Tech Stack

**Backend**
- Python 3.8+
- **FastAPI** & **Uvicorn**: REST API and static file serving.
- **OpenCV** & **NumPy**: Image processing.
- **AI/ML**: Ultralytics (YOLO), MediaPipe, ONNX Runtime, Hugging Face Hub.
- **Database**: `psycopg2-binary` for PostgreSQL interactions.

**Frontend**
- **React**: UI framework.
- **Vite**: Build tool.
- **Tailwind CSS**: Styling.
- **Lucide React**: Icons.

## 📋 Prerequisites

Before running Facer, ensure you have the following installed:

- Python >= 3.8
- Node.js & npm (for building the UI)
- PostgreSQL Database

## 📦 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/facer.git](https://github.com/yourusername/facer.git)
cd facer
```

### 2. Backend Setup
It is recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e .
pip install "mediapipe==0.10.14"
```

### 3. Frontend Setup
```bash
cd facer-ui
npm install
npm run build
cd ..
```

### 4. Database Configuration
You must have a PostgreSQL instance running.

1. Create a database (e.g., `facer_db`).
2. Update the connection string in `src/facer/server.py`:
   ```python
   # TODO: Update credentials
   DB_DSN = "postgresql://<user>:<password>@<host>:5432/<dbname>"
   ```
3. Create the required table. You can execute the following SQL:
   ```sql
   CREATE TABLE faces (
       id SERIAL PRIMARY KEY,
       image_name TEXT,
       description TEXT,
       keywords TEXT,
       classification TEXT,
       bbox FLOAT[],       -- Array of floats [x1, y1, x2, y2]
       yaw FLOAT,
       pitch FLOAT,
       roll FLOAT,
       embedding TEXT,     -- Stored as stringified list
       is_valid_pose BOOLEAN,
       face_image BYTEA,   -- Binary image data
       direction TEXT,
       source_image_hash TEXT,
       created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

## 💻 Usage

### 1. Running the API Server
The server handles analysis requests and serves the frontend.

```bash
uvicorn facer.server:app --host 0.0.0.0 --port 8000 --reload
```
*The API will attempt to download necessary models from Hugging Face on the first run.*

### 2. Running the Frontend
For development mode with hot-reload:

```bash
cd facer-ui
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

### 3. Using the CLI
You can use the python module directly for command-line operations.

**Analyze an image:**
```bash
python -m facer ./images/group_photo.jpg
```

**Enroll a person:**
Finds the first valid face in the image and saves the embedding to `face_db.json` (local JSON fallback used by CLI).
```bash
python -m facer ./images/employee_id.jpg --enroll "Jane Doe"
```

**Identify a person:**
Compares faces in the target image against enrolled embeddings.
```bash
python -m facer ./images/security_cam.jpg --threshold 0.6
```

## ⚙️ Configuration

- **Pose Thresholds**: By default, faces with Yaw or Pitch > **25 degrees** are considered "Invalid" for enrollment. This can be adjusted in `src/facer/server.py` or `src/facer/__main__.py`.
- **Model Caching**: Models are downloaded to the standard Hugging Face cache directory.

## 📄 License

[MIT](LICENSE)
