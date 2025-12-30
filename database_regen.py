import sqlite3
import numpy as np
from pathlib import Path

# Generate fresh database
db_path = Path('moonshine.db')
if db_path.exists():
    db_path.unlink()

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE routing_table (
        triangle_id INTEGER PRIMARY KEY,
        sigma REAL,
        j_real REAL,
        j_imag REAL,
        theta REAL,
        pq_addr INTEGER,
        v_addr INTEGER,
        iv_addr INTEGER
    )
''')

print("Generating 196,883 routes...")
MOONSHINE_DIMENSION = 196883

for triangle_id in range(MOONSHINE_DIMENSION):
    theta = 2 * np.pi * triangle_id / MOONSHINE_DIMENSION
    sigma = (triangle_id % 8000) / 1000.0
    q = np.exp(2j * np.pi * theta / MOONSHINE_DIMENSION)
    j_real = float(np.real(1/q + 744 + 196884*q))
    j_imag = float(np.imag(1/q + 744 + 196884*q))
    
    cursor.execute('''
        INSERT INTO routing_table 
        (triangle_id, sigma, j_real, j_imag, theta, pq_addr, v_addr, iv_addr)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        triangle_id,
        sigma,
        j_real,
        j_imag,
        theta,
        0x100000000 + triangle_id * 512,
        0x200000000 + triangle_id * 256,
        0x300000000 + triangle_id * 256
    ))
    
    if triangle_id % 50000 == 0:
        print(f"  {triangle_id:,} routes...")

conn.commit()

# Verify
cursor.execute('SELECT COUNT(*) FROM routing_table')
count = cursor.fetchone()[0]
print(f"✓ Database complete: {count:,} routes")

cursor.execute('SELECT * FROM routing_table WHERE triangle_id = 0')
row = cursor.fetchone()
print(f"✓ Sample: sigma={row[1]:.6f}, j={row[2]:.2f}+{row[3]:.2f}i")

conn.close()
print(f"✓ Saved to moonshine.db")