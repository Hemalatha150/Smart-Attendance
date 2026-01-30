# Attendance System - AI Coding Agent Instructions

## Project Overview

**Type**: Face Recognition-Based Attendance Management System  
**Tech Stack**: Flask (web backend), PyTorch + FaceNet (biometric), SQLite (data), OpenCV (video processing)  
**Architecture**: Web application with dual-mode face recognition (IN/OUT attendance) + faculty management portal

---

## Core System Architecture

### 1. Face Recognition Pipeline
**Files**: `server.py`, `student_register.py`, `register_face.py`, `student_db.py`

- **Model Stack**: MTCNN (face detection) → InceptionResnetV1 (face embedding generation)
- **Database Schema**: SQLite `faces` table stores:
  - `reg_no` (registration number, unique, case-insensitive)
  - `name`, `phone`, `branch`, `section`, `year`
  - `embedding` (BLOB binary format of 512-dim numpy float32 array)
- **Matching Logic**: Cosine similarity threshold = **0.65** (critical constant in `main.py` line 44)
- **Device Strategy**: Defaults to CPU (not CUDA) - see `torch.set_num_threads(1)` optimization in `server.py` line 24

### 2. Web Application Routes (Flask)
**File**: `server.py` (981 lines)

**Key Entry Points**:
- `POST /api/register_face` - Face capture and embedding generation for new students
- `POST /api/verify_face` - Real-time face verification during IN/OUT
- `/secure_in`, `/secure_out` - Faculty dashboard for attendance mode toggle
- `/faculty_dashboard` - Faculty portal with student management
- `@app.route('/api/toggle_attendance/<mode>/<action>')` - Start/stop attendance (requires `role == 'faculty'`)
- `/attendance_reports` - Generate CSV exports

**Session Management**: Uses Flask sessions with role-based access (`role: 'faculty'` or `'student'`)

### 3. Dual Attendance Modes
**File**: `main.py` (171 lines - desktop client for IN/OUT)

- Runs as command-line tool: `python main.py IN|OUT`
- Uses **Eye Aspect Ratio (EAR) liveness detection**: threshold = **0.24** (line 45)
- Plays system beeps + text-to-speech feedback (winsound, pyttsx3)
- Outputs: `attendance` table with fields: `date`, `time`, `status` (IN/OUT)

---

## Critical Data Flows

### Student Enrollment (Registration)
```
student_register.py (GUI with Tkinter)
  ↓
Capture 5 poses (straight, left, right, up, down)
  ↓
Extract embeddings via model()
  ↓
Store in faces table (reg_no PRIMARY KEY)
  ↓
/faculty_dashboard allows bulk student uploads
```

### Attendance Marking
```
server.py /api/verify_face (from web)
  OR
main.py IN|OUT (desktop executable)
  ↓
Face detection (MTCNN) + embedding
  ↓
Cosine similarity search against all stored embeddings
  ↓
Insert into attendance table if match ≥ 0.65
```

---

## Key Design Patterns & Conventions

### 1. Database Initialization
- Always call `init_db()` before Flask app starts (auto-creates tables if missing)
- **Schema Evolution**: Use `upgrade_faces_table()` pattern to add columns without breaking existing data
- Example: `face_area`, `geo_sig` columns added via ALTER TABLE

### 2. Numpy Binary Serialization
```python
# Storing: embedding.cpu().numpy() → np.frombuffer() → BLOB
embedding_blob = model_output.cpu().numpy().tobytes()

# Retrieving: BLOB → numpy array
retrieved = np.frombuffer(blob, dtype=np.float32)
```

### 3. Case-Insensitive Registration Numbers
- Enforce UPPERCASE-ONLY input validation: `student_db.py` lines 37-47
- Database queries use `COLLATE NOCASE` and `UPPER()` function
- Example: `WHERE UPPER(reg_no) = ?`

### 4. Thread/Process Optimization
- **Voice feedback**: Separate daemon thread with queue (`server.py` style in main.py)
- **Torch inference**: `torch.set_num_threads(1)` + `cv2.setNumThreads(0)` to avoid CPU contention
- **MTCNN tuning**: Custom thresholds `[0.6, 0.7, 0.7]` + `min_face_size=60` for "Look Down" pose

### 5. Warmup Route
Flask requires model warmup before production:
```python
@app.route("/warmup")
def warmup():
    # Dummy forward pass to initialize CUDA/CPU pipeline
```

---

## Environment & Deployment

### Python & Dependencies
- **Python**: 3.8+ (uses f-strings, type hints optional)
- **Key Packages**:
  - `facenet-pytorch==2.5.2` (MTCNN + InceptionResnet)
  - `flask==2.2.5` + `flask-cors` (web framework)
  - `torch==2.0.1` CPU variant (for memory efficiency)
  - `opencv-python-headless==4.8.0.76` (server deployment)
  - `mediapipe==0.10.9` (eye mesh landmarks for liveness)

### Deployment
- **Server**: Gunicorn (config: `gunicorn.conf.py`) - 1 worker, 2 threads, 180s timeout
- **Database Path**: `data/attendance1.db` (created on first run)
- **Static Assets**: `static/` → CSS, JS (responsive design in `static/css/responsive.css`)

---

## Common Developer Tasks

### Modifying Recognition Threshold
- Edit `RECOGNITION_THRESHOLD = 0.65` in `main.py` line 44
- Also check `server.py` line 900+ for verification logic
- Lower value = more false positives, higher = stricter matching

### Adding New Student Attributes
1. Alter `faces` table schema: `student_register.py` `init_db()`
2. Update registration form: `templates/register.html`
3. Update embedding storage logic: `student_register.py` database insert
4. Update attendance export: `export_attendance.py`

### Debugging Attendance Mismatches
- Use `embedding_accuracy.py` - loads all embeddings, calculates pairwise cosine similarity
- Use `check_db.py` - inspects database schema and record counts
- Use `check_col.py` - validates column names across tables

### Testing Face Recognition Pipeline
- `register_face.py` - Manual webcam registration (press 'S' to capture)
- Run with: `python register_face.py` or `python register_face.py REG_NO "Student Name"`

---

## Common Pitfalls & Gotchas

1. **Embedding Format**: Always use `float32` when serializing/deserializing from BLOB - mismatched dtypes cause silent failures
2. **Session Expiry**: Faculty-only routes check `session.get('role')` - login clears session
3. **Face Detection Sensitivity**: MTCNN thresholds heavily impact "Look Down" detection - tuned at `server.py` line 130
4. **GPU vs CPU**: Codebase defaults to CPU - CUDA code exists but not recommended due to memory constraints on deployment servers
5. **File Paths**: Use `os.path.join()` and `BASE_DIR` (not hardcoded paths) for cross-platform compatibility

---

## File Organization Quick Reference

| Purpose | Primary File | Utility Scripts |
|---------|--------------|-----------------|
| Web API & Flask app | `server.py` | `gunicorn.conf.py` |
| Face registration | `student_register.py` | `register_face.py`, `student_db.py` |
| Attendance marking | `main.py` | — |
| Data management | `export_attendance.py` | `check_db.py`, `reset_db.py`, `delete_face.py`, `update_db.py` |
| Accuracy validation | `embedding_accuracy.py` | `check_col.py` |
| Frontend | `templates/` | `static/` (CSS, JS) |

---

## Next Steps for AI Agents

- Before modifying database schema: check `upgrade_faces_table()` pattern in `server.py`
- Before adding routes: verify role-based access requirements and session handling
- Before tuning ML params: run `embedding_accuracy.py` to establish baseline matching accuracy
