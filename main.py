import os
import cv2
import torch
import numpy as np
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import pyttsx3
from sklearn.metrics.pairwise import cosine_similarity
import mediapipe as mp
import sys

if len(sys.argv) < 2:
    print("Usage: python main.py IN|OUT")
    sys.exit()

ATTENDANCE_MODE = sys.argv[1].upper()

if ATTENDANCE_MODE not in ["IN", "OUT"]:
    print("Invalid attendance mode. Use IN or OUT.")
    sys.exit()


# =============================
# CONFIG
# =============================


# =============================
# Device & Models
# =============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    keep_all=False,
    device=device
)

model = InceptionResnetV1(
    pretrained="vggface2"
).eval().to(device)

# =============================
# Text To Speech
# =============================
engine = pyttsx3.init()

# =============================
# Database
# =============================
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/attendance1.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reg_no TEXT,
    name TEXT,
    status TEXT,
    date TEXT,
    time TEXT
)
""")
conn.commit()

# =============================
# Load Registered Faces
# =============================
c.execute("SELECT name, reg_no, embedding FROM faces")
rows = c.fetchall()

if not rows:
    print("No registered students found")
    engine.say("No registered students found")
    engine.runAndWait()
    conn.close()
    exit()

known_names = []
known_regs = []
known_embeddings = []

for n, r, e in rows:
    known_names.append(n)
    known_regs.append(r)
    known_embeddings.append(np.frombuffer(e, dtype=np.float32))

known_embeddings = np.array(known_embeddings)

# =============================
# MediaPipe FaceMesh (Blink)
# =============================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

eye_closed = False
blink_verified = False
blink_start_time = None

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

# =============================
# Start Camera
# =============================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not accessible")
    engine.say("Camera not accessible")
    engine.runAndWait()
    conn.close()
    exit()

print(f"Camera started. Mode: {ATTENDANCE_MODE}. Blink after recognition.")
start_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    # ---------------- LIVENESS CHECK ----------------
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark

        left_eye = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in LEFT_EYE])
        right_eye = np.array([(int(lm[i].x * w), int(lm[i].y * h)) for i in RIGHT_EYE])

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < 0.15 and not eye_closed:
            eye_closed = True

        if ear > 0.18 and eye_closed and blink_start_time is not None:
            blink_verified = True
            eye_closed = False

        cv2.putText(
            frame,
            f"EAR: {ear:.2f}",
            (30, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

    # Blink window timeout
    if blink_start_time is not None:
        if (datetime.now() - blink_start_time).seconds > 3:
            blink_start_time = None
            blink_verified = False

    if not blink_verified:
        cv2.putText(
            frame,
            "Blink after face recognition",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

    # ---------------- FACE RECOGNITION ----------------
    face_tensor = mtcnn(rgb)

    if face_tensor is not None:
        with torch.no_grad():
            face_embedding = model(
                face_tensor.unsqueeze(0).to(device)
            ).cpu().numpy()

        sims = cosine_similarity(face_embedding, known_embeddings)[0]
        idx = int(np.argmax(sims))
        score = float(sims[idx])

        if score > 0.65:
            name = known_names[idx]
            reg_no = known_regs[idx]

            if blink_start_time is None:
                blink_start_time = datetime.now()
                blink_verified = False

            cv2.putText(
                frame,
                f"Recognized: {name}",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            if blink_verified:
                now = datetime.now()
                d = now.strftime("%Y-%m-%d")
                t = now.strftime("%H:%M:%S")

                c.execute(
                    "SELECT * FROM attendance WHERE reg_no=? AND date=? AND status=?",
                    (reg_no, d, ATTENDANCE_MODE)
                )

                if c.fetchone():
                    engine.say(f"{ATTENDANCE_MODE} attendance already marked for {name}")
                    engine.runAndWait()
                    break

                c.execute(
                    "INSERT INTO attendance (name, reg_no, status, date, time) VALUES (?,?,?,?,?)",
                    (name, reg_no, ATTENDANCE_MODE, d, t)
                )
                conn.commit()

                engine.say(f"Hello {name}, your {ATTENDANCE_MODE} attendance has been marked")
                engine.runAndWait()

                cv2.putText(
                    frame,
                    f"{name} - {ATTENDANCE_MODE} Marked",
                    (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

                cv2.imshow("Smart Attendance", frame)
                cv2.waitKey(1000)
                break

        else:
            cv2.putText(
                frame,
                "Unknown face",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("Smart Attendance", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

    if (datetime.now() - start_time).seconds > 120:
        print("Timeout")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
