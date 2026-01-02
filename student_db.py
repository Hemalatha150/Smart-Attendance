import os
import cv2
import torch
import numpy as np
import sqlite3
import re
from facenet_pytorch import MTCNN, InceptionResnetV1

# ==========================
# 1) MODEL INITIALIZATION
# ==========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ==========================
# 2) DATABASE SETUP
# ==========================
os.makedirs("data", exist_ok=True)
db_path = "data/attendance1.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS faces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reg_no TEXT UNIQUE COLLATE NOCASE,
    name TEXT,
    embedding BLOB
)
""")
conn.commit()

# ==========================
# 3) INPUT STUDENT DETAILS
# ==========================

pattern = r'^[A-Z0-9]+$'   # ONLY BLOCK LETTERS + NUMBERS

while True:
    reg_no = input("Enter Registration Number (BLOCK letters & numbers only): ").strip()

    if not reg_no:
        print("❌ Registration number cannot be empty.")
        continue

    # strictly reject lowercase
    if any(c.islower() for c in reg_no):
        print("❌ Lowercase letters are NOT allowed. Please use BLOCK CAPITAL LETTERS only.")
        continue

    # only A–Z and digits allowed
    if not re.fullmatch(pattern, reg_no):
        print("❌ Only BLOCK letters (A–Z) and digits (0–9) allowed — no spaces or symbols.")
        continue

    break  # valid value

name = input("Enter Student Name: ").strip().title()

# case-insensitive duplicate check
cursor.execute("SELECT * FROM faces WHERE UPPER(reg_no) = ?", (reg_no.upper(),))
if cursor.fetchone():
    print("\n❌ ERROR: Registration number already exists in database.\n")
    conn.close()
    exit()

# ==========================
# 4) LOAD EXISTING EMBEDDINGS
# ==========================
cursor.execute("SELECT reg_no, name, embedding FROM faces")
existing_faces = cursor.fetchall()

existing_embeddings = []
existing_details = []

for reg, nm, emb in existing_faces:
    existing_embeddings.append(np.frombuffer(emb, dtype=np.float32))
    existing_details.append((reg, nm))

# ==========================
# 5) DUPLICATE FACE CHECK FUNCTION
# ==========================
def is_duplicate_face(new_emb, threshold=0.7):
    for idx, emb in enumerate(existing_embeddings):
        dist = np.linalg.norm(new_emb - emb)
        if dist < threshold:
            return True, existing_details[idx], dist
    return False, None, None

# ==========================
# 6) POSE DEFINITIONS
# ==========================
POSES = [
    "Look straight",
    "Turn face LEFT",
    "Turn face RIGHT",
    "Look UP",
    "Look DOWN"
]

def detect_pose(landmarks):
    left_eye, right_eye, nose, mouth_left, mouth_right = landmarks

    lx, ly = left_eye
    rx, ry = right_eye
    nx, ny = nose
    mx1, my1 = mouth_left
    mx2, my2 = mouth_right

    eye_center_x = (lx + rx) / 2

    if nx < eye_center_x - 5:
        return "LEFT"
    if nx > eye_center_x + 5:
        return "RIGHT"

    mouth_center_y = (my1 + my2) / 2
    eye_center_y = (ly + ry) / 2
    mid_face_y = (mouth_center_y + eye_center_y) / 2

    if ny < mid_face_y - 5:
        return "UP"
    if ny > mid_face_y + 5:
        return "DOWN"

    return "CENTER"

# ==========================
# 7) CAMERA SETUP
# ==========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not accessible.")
    conn.close()
    exit()

print("\n📸 Registration started")
print("Follow on-screen instructions.")
print("Press 'q' anytime to cancel.\n")

captured_embeddings = []

# ==========================
# 8) POSE-GUIDED CAPTURE LOOP
# ==========================
for pose in POSES:
    print(f"\n👉 Instruction: {pose}")
    print("✔️ Hold still when ready…")

    collected = False

    while not collected:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, probs, landmarks = mtcnn.detect(rgb, landmarks=True)

        cv2.putText(frame, pose, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        if boxes is None:
            cv2.putText(frame, "Face not detected", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        else:
            if len(boxes) > 1:
                cv2.putText(frame, "Only ONE face allowed!", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            else:
                detected_pose = detect_pose(landmarks[0])

                expected = {
                    "Look straight": "CENTER",
                    "Turn face LEFT": "LEFT",
                    "Turn face RIGHT": "RIGHT",
                    "Look UP": "UP",
                    "Look DOWN": "DOWN"
                }[pose]

                if detected_pose == expected:
                    cv2.putText(frame, "Pose Correct ✔️", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    face_tensor = mtcnn(rgb)

                    with torch.no_grad():
                        emb = model(face_tensor.unsqueeze(0).to(device)).cpu().numpy().flatten()

                    duplicate, info, dist = is_duplicate_face(emb)

                    if duplicate:
                        print("\n⚠️ ALERT: Face already registered!")
                        print(f"Existing Student: {info[1]}  (Reg: {info[0]})")
                        print(f"Similarity score: {dist:.4f}")
                        print("❌ Registration stopped.\n")

                        cap.release()
                        cv2.destroyAllWindows()
                        conn.close()
                        exit()

                    captured_embeddings.append(emb)
                    collected = True
                    print(f"[OK] Captured: {expected}")

                    cv2.imshow("Registration Camera", frame)
                    cv2.waitKey(600)

                else:
                    cv2.putText(frame, "Follow instruction correctly", (30, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Registration Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n❌ Registration cancelled by user.")
            cap.release()
            cv2.destroyAllWindows()
            conn.close()
            exit()

cap.release()
cv2.destroyAllWindows()

# ==========================
# 9) SAVE FINAL EMBEDDING
# ==========================
avg_embedding = np.mean(captured_embeddings, axis=0).astype(np.float32).tobytes()

cursor.execute(
    "INSERT INTO faces (reg_no, name, embedding) VALUES (?, ?, ?)",
    (reg_no, name, avg_embedding)
)

conn.commit()
conn.close()

print(f"\n🎉 SUCCESS — Registered {name} ({reg_no}) with pose-verified, duplicate-checked samples.\n")
