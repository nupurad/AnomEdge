from app.db import init_db, get_connection

init_db()  # <-- this creates the tables

conn = get_connection()
cur = conn.cursor()

cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
tables = cur.fetchall()

print("Tables in DB:")
for row in tables:
    print(row[0])

conn.close()