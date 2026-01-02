import sqlite3

db_path = "data/attendance1.db"
print("Using DB:", db_path)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 1) list tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("\nTables in database:")
for t in tables:
    print(" -", t[0])

# 2) check attendance count if exists
try:
    cursor.execute("SELECT COUNT(*) FROM attendance")
    print("\nattendance rows:", cursor.fetchone()[0])
except:
    print("\n❌ No table named 'attendance'")

conn.close()
