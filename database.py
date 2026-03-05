import sqlite3
import datetime

DB_NAME = "attendance.db"

def initialize_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def log_attendance(user_id, name):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO attendance (user_id, name) VALUES (?, ?)", (user_id, name))
    conn.commit()
    conn.close()

def has_attended_today(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    today = datetime.date.today().isoformat()
    cursor.execute("""
        SELECT 1 FROM attendance 
        WHERE user_id = ? AND date(timestamp) = ?
    """, (user_id, today))
    result = cursor.fetchone()
    conn.close()
    return result is not None
