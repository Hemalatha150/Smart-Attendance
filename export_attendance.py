import sqlite3
import pandas as pd
import os

DB_PATH = "data/attendance1.db"
EXPORT_PATH = "data/attendance_export.csv"

# ---------- Safety checks ----------
if not os.path.exists(DB_PATH):
    print("❌ Database file not found at:", DB_PATH)
    exit()

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ---------- Check attendance table exists ----------
cursor.execute("""
SELECT name FROM sqlite_master WHERE type='table' AND name='attendance';
""")
table_exists = cursor.fetchone()

if not table_exists:
    print("❌ attendance table does not exist in database.")
    conn.close()
    exit()

# ---------- Check if table has rows ----------
cursor.execute("SELECT COUNT(*) FROM attendance")
count = cursor.fetchone()[0]

if count == 0:
    print("⚠️ No attendance records found. CSV not created.")
    conn.close()
    exit()

# ---------- Export ----------
df = pd.read_sql_query("SELECT * FROM attendance", conn)

df.to_csv(EXPORT_PATH, index=False)

conn.close()

print("✅ Attendance exported successfully")
print("📁 File saved at:", EXPORT_PATH)
