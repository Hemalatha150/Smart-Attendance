import cv2
import torch
import numpy as np
import sqlite3
import os
import sys
from facenet_pytorch import MTCNN, InceptionResnetV1

# -----------------------------
# Get student details
# -----------------------------
if len(sys.argv) >= 3:
    reg_no = sys.argv[1]
    name = sys.argv[2]
else:
    reg_no = input("Enter Registration Number: ")
    name = input("Enter Student Name: ")

# -----------------------------
# Models
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, keep_all=False, device=device)
model = InceptionResnetV1(pretrained="vggface2").eval().to(device)

# -----------------------------
# Database
# -----------------------------
os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/attendance1.db")
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    reg_no TEXT UNIQUE,
    embedding BLOB
)
""")
conn.commit()

# -----------------------------
# Camera
# -----------------------------
cap = cv2.VideoCapture(0)
print("📸 Press 'S' to capture face")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = mtcnn(rgb)

    cv2.imshow("Register Face", frame)

    if face is not None:
        if cv2.waitKey(1) & 0xFF == ord('s'):
            with torch.no_grad():
                embedding = model(face.unsqueeze(0).to(device)).cpu().numpy()

            emb_blob = embedding.astype(np.float32).tobytes()

            c.execute(
                "INSERT OR REPLACE INTO faces (name, reg_no, embedding) VALUES (?, ?, ?)",
                (name, reg_no, emb_blob)
            )
            conn.commit()

            print(f"✅ {name} registered successfully")
            break

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
conn.close()
