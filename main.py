import os
import cv2
import torch
import numpy as np
import sqlite3
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import pyttsx3
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1) Device & Models
# -----------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# -----------------------------
# 2) TTS
# -----------------------------
engine = pyttsx3.init()

# -----------------------------
# 3) Database setup
# -----------------------------
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/attendance1.db")
c = conn.cursor()

# ensure table exists
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

# -----------------------------
# 4) Load registered faces
# -----------------------------
c.execute("SELECT name, reg_no, embedding FROM faces")
data = c.fetchall()

if len(data) == 0:
    print("❌ No registered students found in database.")
    engine.say("No registered students found.")
    engine.runAndWait()
    conn.close()
    exit()

known_embeddings = []
known_names = []
known_regs = []

for name, reg_no, embedding_blob in data:
    emb = np.frombuffer(embedding_blob, dtype=np.float32)
    known_embeddings.append(emb)
    known_names.append(name)
    known_regs.append(reg_no)

known_embeddings = np.array(known_embeddings)

# -----------------------------
# 5) Start camera
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not accessible.")
    engine.say("Camera not accessible")
    engine.runAndWait()
    conn.close()
    exit()

print("[INFO] Showing webcam. It will auto-exit after marking attendance.")

start_time = datetime.now()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img)

    # if face detected
    if face_tensor is not None:
        with torch.no_grad():
            face_embedding = model(face_tensor.unsqueeze(0).to(device)).cpu().numpy()

        # compute cosine similarity
        similarities = cosine_similarity(face_embedding, known_embeddings)[0]
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        # recognition threshold
        if best_score > 0.65:
            name = known_names[best_idx]
            reg_no = known_regs[best_idx]

            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")
            time_str = now.strftime("%H:%M:%S")

            # ---------- ONCE PER DAY CHECK ----------
            c.execute("SELECT * FROM attendance WHERE reg_no=? AND date=?", (reg_no, date_str))
            already = c.fetchone()

            if already:
                text = f"Attendance already marked for {name} today."
                print("[INFO]", text)
                engine.say(text)
                engine.runAndWait()
                break

            # ---------- MARK ATTENDANCE ----------
            c.execute(
                "INSERT INTO attendance (name, reg_no, status, date, time) VALUES (?, ?, ?, ?, ?)",
                (name, reg_no, "Present", date_str, time_str),
            )
            conn.commit()

            text = f"Hello {name}, your attendance has been marked."
            print("[INFO]", text)
            engine.say(text)
            engine.runAndWait()

            cv2.putText(frame, f"{name} - Marked", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Smart Attendance", frame)
            cv2.waitKey(1000)
            break

        else:
            cv2.putText(frame, "Unknown face", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    else:
        cv2.putText(frame, "Show your face to the camera", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)

    cv2.imshow("Smart Attendance", frame)

    # ESC to quit manually
    if cv2.waitKey(1) & 0xFF == 27:
        break

    # auto-timeout: 60 seconds
    if (datetime.now() - start_time).seconds > 60:
        print("[INFO] Auto exit due to timeout.")
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
