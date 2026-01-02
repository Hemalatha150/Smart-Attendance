import cv2
import torch
import numpy as np
import sqlite3
from flask import Flask, request, jsonify, render_template
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime
import os

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mtcnn = MTCNN(image_size=160, margin=0, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

os.makedirs("data", exist_ok=True)
conn = sqlite3.connect("data/attendance1.db", check_same_thread=False)
c = conn.cursor()

c.execute("""
CREATE TABLE IF NOT EXISTS attendance(
 id INTEGER PRIMARY KEY AUTOINCREMENT,
 reg_no TEXT,
 name TEXT,
 status TEXT,
 date TEXT,
 time TEXT
)
""")
conn.commit()

def load_faces():
    c.execute("SELECT name, reg_no, embedding FROM faces")
    rows = c.fetchall()

    names, regs, embs = [], [], []

    for n, r, e in rows:
        names.append(n)
        regs.append(r)
        embs.append(np.frombuffer(e, dtype=np.float32))

    return names, regs, np.array(embs) if embs else np.array([])

known_names, known_regs, known_embs = load_faces()

def process(mark_type):

    global known_names, known_regs, known_embs
    known_names, known_regs, known_embs = load_faces()

    if len(known_embs) == 0:
        return {"status":"error","message":"No registered faces found"}

    if "image" not in request.files:
        return {"status":"error","message":"No image received"}

    challenge = request.form.get("challenge","")

    if challenge == "":
        return {"status":"error","message":"Get challenge first"}

    # read image
    npimg = np.frombuffer(request.files["image"].read(), np.uint8)
    frame = cv2.imdecode(npimg, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # detect
    face = mtcnn(rgb)
    if face is None:
        return {"status":"no_face","message":"Face not detected"}

    with torch.no_grad():
        emb = model(face.unsqueeze(0).to(device)).cpu().numpy()

    emb = emb / np.linalg.norm(emb)
    db = known_embs / np.linalg.norm(known_embs, axis=1, keepdims=True)

    sims = np.dot(db, emb.squeeze())
    idx = int(np.argmax(sims))
    score = float(sims[idx])

    if score < 0.55:
        return {"status":"unknown","message":"Face not recognized"}

    name = known_names[idx]
    reg = known_regs[idx]

    now = datetime.now()
    d = now.strftime("%Y-%m-%d")
    t = now.strftime("%H:%M:%S")

    c.execute("SELECT * FROM attendance WHERE reg_no=? AND date=? AND status=?", (reg, d, mark_type))

    if c.fetchone():
        return {"status":"already","message":f"{mark_type} already done today"}

    # accept challenge (simple version)
    print("Challenge completed:", challenge)

    c.execute("INSERT INTO attendance (reg_no,name,status,date,time) VALUES (?,?,?,?,?)",
              (reg,name,mark_type,d,t))
    conn.commit()

    return {"status":"marked","message":f"{mark_type} marked for {name}"}

@app.route("/")
def in_page():
    return render_template("in_cam.html")

@app.route("/out")
def out_page():
    return render_template("out_cam.html")

@app.route("/mark_in", methods=["POST"])
def mark_in():
    return jsonify(process("IN"))

@app.route("/mark_out", methods=["POST"])
def mark_out():
    return jsonify(process("OUT"))

if __name__ == "__main__":
    app.run(debug=True)
