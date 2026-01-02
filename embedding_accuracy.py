import sqlite3
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

THRESHOLD = 0.6
DB_PATH = "data/attendance1.db"

# -------------------------------
# Step 1: Connect to database
# -------------------------------
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# -------------------------------
# Step 2: Find table containing embeddings
# -------------------------------
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in cursor.fetchall()]

face_table = None
name_col = None
embed_col = None

for table in tables:
    cursor.execute(f"PRAGMA table_info({table})")
    cols = cursor.fetchall()
    col_names = [c[1].lower() for c in cols]

    if "embedding" in col_names and ("name" in col_names or "username" in col_names):
        face_table = table
        name_col = cols[col_names.index("name")][1]
        embed_col = cols[col_names.index("embedding")][1]
        break

if face_table is None:
    print("❌ No face embedding table found in database.")
    conn.close()
    exit()

print(f"✅ Using table: {face_table}")
print(f"   Name column: {name_col}")
print(f"   Embedding column: {embed_col}")

# -------------------------------
# Step 3: Load embeddings
# -------------------------------
cursor.execute(f"SELECT {name_col}, {embed_col} FROM {face_table}")
rows = cursor.fetchall()
conn.close()

names = []
embeddings = []

for name, emb_blob in rows:
    emb = np.frombuffer(emb_blob, dtype=np.float32)
    embeddings.append(emb)
    names.append(name)

if len(embeddings) < 2:
    print("❌ Not enough embeddings to calculate accuracy.")
    exit()

# -------------------------------
# Step 4: Accuracy calculation
# -------------------------------
correct = 0
total = 0

for i in range(len(embeddings)):
    for j in range(i + 1, len(embeddings)):
        sim = cosine_similarity(
            embeddings[i].reshape(1, -1),
            embeddings[j].reshape(1, -1)
        )[0][0]

        total += 1

        if names[i] == names[j] and sim >= THRESHOLD:
            correct += 1
        elif names[i] != names[j] and sim < THRESHOLD:
            correct += 1

accuracy = (correct / total) * 100
print(f"\n🎯 Accuracy (Embedding-based): {accuracy:.2f}%")
