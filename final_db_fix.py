import sqlite3
import os

# మీ server.py లో ఉన్న ఖచ్చితమైన పాత్
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "attendance1.db")

def super_fix():
    # ఒకవేళ data ఫోల్డర్ లేకపోతే క్రియేట్ చేస్తుంది
    if not os.path.exists(os.path.join(BASE_DIR, "data")):
        os.makedirs(os.path.join(BASE_DIR, "data"))
        print("Created 'data' folder.")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    try:
        # పాత టేబుల్ ని డిలీట్ చేసి కొత్తగా క్రియేట్ చేస్తున్నాం
        cursor.execute("DROP TABLE IF EXISTS attendance")
        
        cursor.execute('''
            CREATE TABLE attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                reg_no TEXT,
                name TEXT,
                branch TEXT,
                section TEXT,
                year TEXT,
                subject TEXT,
                date TEXT,
                time TEXT,
                status TEXT
            )
        ''')
        conn.commit()
        print(f"✅ Success! Table 'attendance' created with 'subject' column in: {DB_PATH}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    super_fix()