import sqlite3
import os

# 1. ఇక్కడ మీ server.py లో ఉన్న ఖచ్చితమైన పేరు ఇవ్వండి
DB_NAME = 'database.db' 

def super_fix():
    if not os.path.exists(DB_NAME):
        print(f"Warning: {DB_NAME} not found, creating new one.")
    
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        # పాత టేబుల్ ని తీసేస్తున్నాం (Error రాకుండా ఉండటానికి)
        cursor.execute("DROP TABLE IF EXISTS attendance")
        
        # కొత్తగా అన్ని కాలమ్స్ తో టేబుల్ క్రియేట్ చేస్తున్నాం
        cursor.execute('''
            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reg_no TEXT,
                name TEXT,
                year TEXT,
                branch TEXT,
                section TEXT,
                subject TEXT,
                date TEXT,
                time TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        print(f"✅ Success! Table 'attendance' re-created in {DB_NAME}")
        
        # ఏయే కాలమ్స్ ఉన్నాయో ప్రింట్ చేసి చూపిస్తుంది
        cursor.execute("PRAGMA table_info(attendance)")
        cols = [c[1] for c in cursor.fetchall()]
        print(f"Current columns: {cols}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    super_fix()