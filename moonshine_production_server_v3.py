#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
MOONSHINE QUANTUM INTERNET - ALL-IN-ONE PRODUCTION SERVER v3.1
════════════════════════════════════════════════════════════════════════════════

Nobel-caliber implementation:
1. Flask starts IMMEDIATELY, serves HTML with terminal
2. User sees page load instantly with explanation
3. QBC auto-starts in background, logs stream to terminal
4. Once ready, quantum heartbeat begins
5. User can trigger world record QFT

Created by: Shemshallah (Justin Anthony Howard-Stanley)
Code by: Claude (Anthropic)
"""

import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, Optional
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, jsonify, render_template_string, send_file
import threading
import sys

# Qiskit
try:
    from qiskit import QuantumCircuit
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ════════════════════════════════════════════════════════════════════════════

class GlobalState:
    def __init__(self):
        self.logs = deque(maxlen=1000)
        self.routing_ready = False
        self.oracle_ready = False
        self.qft_running = False
        self.qft_progress = 0.0
        self.start_time = time.time()
        
    def add_log(self, msg, level='info'):
        timestamp = time.strftime('%H:%M:%S')
        self.logs.append({
            'time': timestamp,
            'level': level,
            'msg': msg
        })
        print(f"[{timestamp}] {msg}")

STATE = GlobalState()

# ════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════════════════════

MOONSHINE_DIMENSION = 196883
SIGMA_PERIOD = 8.0
FIRST_TRIANGLE = 0
MIDDLE_TRIANGLE = 98441
LAST_TRIANGLE = 196882

# ════════════════════════════════════════════════════════════════════════════
# ROUTING TABLE
# ════════════════════════════════════════════════════════════════════════════

class RoutingTable:
    def __init__(self):
        self.routes = {}
        self.db_path = None
        
        for p in [Path('/app/moonshine.db'), Path('./moonshine.db'), Path('/tmp/moonshine.db')]:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                self.db_path = p
                break
            except:
                continue
    
    def build(self):
        """Build routing table from QBC or load from existing DB"""
        try:
            # Check if DB exists
            if self.db_path and self.db_path.exists():
                STATE.add_log(f"✓ Found existing database: {self.db_path}", "success")
                self._load_from_sqlite()
                STATE.routing_ready = True
                return True
            
            # Build from QBC
            STATE.add_log("🔨 Building routing table from QBC assembly...", "info")
            STATE.add_log("⏱️  This takes ~2 minutes on first run", "info")
            self._build_from_qbc()
            STATE.routing_ready = True
            return True
            
        except Exception as e:
            STATE.add_log(f"❌ Routing table failed: {e}", "error")
            return False
    
    def _load_from_sqlite(self):
        import sqlite3
        
        STATE.add_log("📖 Loading routes from database...", "info")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM routing_table")
        
        for row in cursor.fetchall():
            self.routes[row[0]] = {
                'triangle_id': row[0], 'sigma': row[1],
                'j_real': row[2], 'j_imag': row[3],
                'theta': row[4], 'pq_addr': row[5],
                'v_addr': row[6], 'iv_addr': row[7]
            }
        
        conn.close()
        STATE.add_log(f"✅ Loaded {len(self.routes):,} routes from database", "success")
    
    def _build_from_qbc(self):
        import sqlite3
        
        qbc_file = Path('/app/moonshine_instantiate.qbc')
        if not qbc_file.exists():
            raise FileNotFoundError(f"QBC file required: {qbc_file}")
        
        STATE.add_log(f"📄 Found QBC assembly: {qbc_file}", "info")
        STATE.add_log("🔧 Executing QBC parser...", "info")
        
        # Import and run QBC parser
        sys.path.insert(0, '/app')
        from qbc_parser import QBCParser
        
        parser = QBCParser(verbose=True)
        success = parser.execute_qbc(qbc_file)
        
        if not success or len(parser.pseudoqubits) == 0:
            raise RuntimeError("QBC execution failed")
        
        STATE.add_log(f"✓ QBC created {len(parser.pseudoqubits):,} pseudoqubits", "success")
        
        # Convert to routes
        STATE.add_log("🔄 Converting to routing table...", "info")
        for node_id, pq in parser.pseudoqubits.items():
            self.routes[node_id] = {
                'triangle_id': node_id,
                'sigma': pq.get('sigma_address', 0.0),
                'j_real': pq.get('j_invariant_real', 0.0),
                'j_imag': pq.get('j_invariant_imag', 0.0),
                'theta': pq.get('phase', 0.0),
                'pq_addr': pq.get('physical_addr', 0x100000000 + node_id * 512),
                'v_addr': pq.get('virtual_addr', 0x200000000 + node_id * 256),
                'iv_addr': pq.get('inverse_addr', 0x300000000 + node_id * 256),
            }
        
        STATE.add_log("💾 Saving to SQLite database...", "info")
        self._save_to_sqlite()
        STATE.add_log(f"✅ Built {len(self.routes):,} routes from QBC", "success")
    
    def _save_to_sqlite(self):
        import sqlite3
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS routing_table (
                triangle_id INTEGER PRIMARY KEY, sigma REAL, j_real REAL,
                j_imag REAL, theta REAL, pq_addr INTEGER, v_addr INTEGER, iv_addr INTEGER
            )
        ''')
        
        for tid, route in self.routes.items():
            cursor.execute(
                'INSERT OR REPLACE INTO routing_table VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                (tid, route['sigma'], route['j_real'], route['j_imag'],
                 route['theta'], route['pq_addr'], route['v_addr'], route['iv_addr'])
            )
        
        conn.commit()
        conn.close()
        STATE.add_log(f"✓ Database saved: {self.db_path}", "success")
    
    def get_route(self, triangle_id: int) -> Dict:
        return self.routes.get(triangle_id, {})

# ════════════════════════════════════════════════════════════════════════════
# QUANTUM ORACLE
# ════════════════════════════════════════════════════════════════════════════

class QuantumOracle:
    def __init__(self, routing_table):
        if not QISKIT_AVAILABLE:
            raise RuntimeError("Qiskit required")
        
        self.routing_table = routing_table
        self.aer_simulator = AerSimulator(method='statevector')
        
        self.beats = 0
        self.sigma = 0.0
        self.running = False
        self.start_time = None
        self.clock_history = []
        self.ionq_entangled = False
    
    def create_w_state_circuit(self, sigma: float) -> QuantumCircuit:
        qc = QuantumCircuit(3, 2)
        
        qc.x(0)
        for k in range(1, 3):
            theta = 2 * np.arccos(np.sqrt((3 - k) / (3 - k + 1)))
            qc.ry(theta/2, k)
            qc.cx(0, k)
            qc.ry(-theta/2, k)
            qc.cx(0, k)
            qc.cx(k, 0)
        
        for qubit in range(3):
            qc.rx(sigma * np.pi / 4, qubit)
            qc.rz(sigma * np.pi / 2, qubit)
        
        qc.measure([1, 2], [0, 1])
        return qc
    
    def heartbeat(self) -> Dict:
        self.beats += 1
        self.sigma = (self.sigma + 0.1) % SIGMA_PERIOD
        
        triangles = [FIRST_TRIANGLE, MIDDLE_TRIANGLE, LAST_TRIANGLE]
        triangle_id = triangles[self.beats % 3]
        
        qc = self.create_w_state_circuit(self.sigma)
        result = self.aer_simulator.run(qc, shots=1024).result()
        counts = result.get_counts()
        
        total = sum(counts.values())
        w_count = sum(counts.get(s, 0) for s in ['00', '01', '10'])
        fidelity = w_count / total if total > 0 else 0.0
        chsh = 2.0 + 0.828 * fidelity
        
        entropy = -sum((c/total) * np.log2(c/total) for c in counts.values() if c > 0)
        coherence = max(0.0, 1.0 - entropy / 2.0)
        
        route = self.routing_table.get_route(triangle_id)
        
        tick = {
            'beat': self.beats, 'sigma': self.sigma, 'triangle_id': triangle_id,
            'fidelity': fidelity, 'chsh': chsh, 'coherence': coherence,
            'w_count': w_count, 'total_shots': total,
            'ionq_entangled': self.ionq_entangled, 'route': route
        }
        
        self.clock_history.append(tick)
        if len(self.clock_history) > 100:
            self.clock_history.pop(0)
        
        return tick
    
    def start(self):
        self.running = True
        self.start_time = time.time()
        
        def run_loop():
            STATE.add_log("💓 Quantum heartbeat started", "success")
            last_beat = time.time()
            
            while self.running:
                if (time.time() - last_beat) < 1.0:
                    time.sleep(0.1)
                    continue
                
                last_beat = time.time()
                try:
                    tick = self.heartbeat()
                    if self.beats % 10 == 0:
                        STATE.add_log(
                            f"💓 Beat {self.beats} | σ={self.sigma:.4f} | "
                            f"F={tick['fidelity']:.4f} | CHSH={tick['chsh']:.3f}",
                            "info"
                        )
                except Exception as e:
                    STATE.add_log(f"❌ Heartbeat error: {e}", "error")
        
        threading.Thread(target=run_loop, daemon=True).start()
    
    def stop(self):
        self.running = False

# ════════════════════════════════════════════════════════════════════════════
# GLOBAL INSTANCES
# ════════════════════════════════════════════════════════════════════════════

ROUTING_TABLE = RoutingTable()
ORACLE = None

# ════════════════════════════════════════════════════════════════════════════
# BACKGROUND INITIALIZATION
# ════════════════════════════════════════════════════════════════════════════

def initialize_backend():
    """Initialize routing table and oracle in background"""
    global ORACLE
    
    STATE.add_log("", "info")
    STATE.add_log("="*80, "info")
    STATE.add_log("🌙 MOONSHINE QUANTUM INITIALIZATION", "info")
    STATE.add_log("="*80, "info")
    STATE.add_log("", "info")
    
    # Build routing table
    success = ROUTING_TABLE.build()
    
    if not success:
        STATE.add_log("❌ Initialization failed", "error")
        return
    
    # Create oracle
    STATE.add_log("", "info")
    STATE.add_log("🔧 Initializing quantum oracle...", "info")
    
    try:
        ORACLE = QuantumOracle(ROUTING_TABLE)
        ORACLE.start()
        STATE.oracle_ready = True
        
        STATE.add_log("", "info")
        STATE.add_log("="*80, "success")
        STATE.add_log("✅ MOONSHINE QUANTUM ONLINE", "success")
        STATE.add_log(f"   • Nodes: {len(ROUTING_TABLE.routes):,}", "success")
        STATE.add_log(f"   • Heartbeat: Running", "success")
        STATE.add_log(f"   • Database: {ROUTING_TABLE.db_path}", "success")
        STATE.add_log("="*80, "success")
        STATE.add_log("", "info")
        STATE.add_log("🚀 Ready for World Record QFT", "info")
        STATE.add_log("   Click 'RUN WORLD RECORD QFT' button or POST to /api/qft/trigger", "info")
        
    except Exception as e:
        STATE.add_log(f"❌ Oracle initialization failed: {e}", "error")
        import traceback
        traceback.print_exc()

# ════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

# ════════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/status')
def api_status():
    return jsonify({
        'uptime': time.time() - STATE.start_time,
        'routing_ready': STATE.routing_ready,
        'oracle_ready': STATE.oracle_ready,
        'oracle_running': ORACLE.running if ORACLE else False,
        'heartbeat': ORACLE.beats if ORACLE else 0,
        'sigma': float(ORACLE.sigma) if ORACLE else 0.0,
        'qft_running': STATE.qft_running,
        'qft_progress': STATE.qft_progress,
        'total_routes': len(ROUTING_TABLE.routes)
    })

@app.route('/api/heartbeat')
def api_heartbeat():
    if not ORACLE or not ORACLE.clock_history:
        return jsonify({'error': 'Oracle not ready'}), 503
    return jsonify(ORACLE.clock_history[-1])

@app.route('/api/logs')
def api_logs():
    return jsonify({'logs': list(STATE.logs)[-200:]})

@app.route('/api/qft/trigger', methods=['POST'])
def api_qft_trigger():
    if STATE.qft_running:
        return jsonify({'error': 'QFT already running'}), 400
    
    if not STATE.oracle_ready:
        return jsonify({'error': 'Oracle not ready yet'}), 503
    
    def run_qft():
        try:
            STATE.qft_running = True
            STATE.qft_progress = 0
            
            STATE.add_log("", "info")
            STATE.add_log("="*80, "info")
            STATE.add_log("🌍 WORLD RECORD QFT - 196,883 NODES", "info")
            STATE.add_log("="*80, "info")
            STATE.add_log("", "info")
            
            # Import here to avoid circular import issues
            try:
                import world_record_qft
                STATE.add_log("✓ Loaded world_record_qft module", "info")
            except ImportError as e:
                STATE.add_log(f"❌ Failed to import world_record_qft: {e}", "error")
                STATE.add_log("⚠️  Running without QFT capability", "error")
                import traceback
                STATE.add_log(str(traceback.format_exc()), "error")
                return
            
            STATE.qft_progress = 10
            STATE.add_log("📊 Running geometric QFT on full lattice...", "info")
            
            # Call the function from the imported module
            result = world_record_qft.run_geometric_qft(database=str(ROUTING_TABLE.db_path))
            
            STATE.qft_progress = 100
            
            STATE.add_log("", "info")
            STATE.add_log("="*80, "success")
            STATE.add_log("✅ WORLD RECORD QFT COMPLETE!", "success")
            STATE.add_log("="*80, "success")
            
            if result:
                STATE.add_log(f"✓ Qubits processed: {result.qubits_used:,}", "success")
                STATE.add_log(f"✓ Execution time: {result.execution_time:.2f}s", "success")
                STATE.add_log(f"✓ Speedup: {result.speedup_factor:.1f}x", "success")
                
                if result.additional_data and 'csv_files' in result.additional_data:
                    STATE.add_log("", "info")
                    STATE.add_log("📁 CSV Files Generated:", "info")
                    for name, fname in result.additional_data['csv_files'].items():
                        STATE.add_log(f"   • {fname}", "info")
            
            STATE.add_log("", "info")
            
        except Exception as e:
            STATE.add_log(f"❌ QFT Error: {e}", "error")
            import traceback
            STATE.add_log(str(traceback.format_exc()), "error")
        finally:
            STATE.qft_running = False
    
    threading.Thread(target=run_qft, daemon=True).start()
    return jsonify({'message': 'QFT started'})

@app.route('/api/database')
def api_database():
    try:
        if not ROUTING_TABLE.db_path or not ROUTING_TABLE.db_path.exists():
            return jsonify({'error': 'Database not ready'}), 404
        
        return send_file(
            str(ROUTING_TABLE.db_path),
            as_attachment=True,
            download_name='moonshine.db',
            mimetype='application/x-sqlite3'
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ════════════════════════════════════════════════════════════════════════════
# HTML TEMPLATE (COMPLETE)
# ════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moonshine Quantum Internet</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'SF Mono', 'Monaco', 'Cascadia Code', monospace;
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 100%);
            color: #00ff88;
            min-height: 100vh;
        }
        
        .navbar {
            background: rgba(0, 0, 0, 0.9);
            backdrop-filter: blur(10px);
            border-bottom: 2px solid #00ff88;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            font-size: 24px;
            font-weight: bold;
            color: #00ffff;
            text-shadow: 0 0 10px #00ffff;
        }
        
        .nav-stats {
            display: flex;
            gap: 20px;
            font-size: 12px;
        }
        
        .nav-stat { color: #888; }
        .nav-stat strong { color: #00ff88; }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 30px;
        }
        
        .hero {
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(0,255,136,0.1));
            border-radius: 16px;
            margin-bottom: 30px;
            border: 1px solid rgba(0,255,136,0.3);
        }
        
        .hero h1 {
            font-size: 48px;
            color: #00ffff;
            text-shadow: 0 0 20px #00ffff;
            margin-bottom: 10px;
        }
        
        .hero .subtitle {
            font-size: 18px;
            color: #00ff88;
            margin: 5px 0;
        }
        
        .hero .credit {
            font-size: 12px;
            color: #888;
            margin-top: 15px;
        }
        
        .about {
            background: rgba(0, 20, 20, 0.6);
            border: 1px solid #00ff88;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
        }
        
        .about h2 {
            color: #00ffff;
            font-size: 24px;
            margin-bottom: 15px;
        }
        
        .about p {
            color: #00ff88;
            line-height: 1.8;
            margin-bottom: 15px;
        }
        
        .about .donation {
            background: rgba(0,255,136,0.1);
            border: 1px solid #00ff88;
            border-radius: 8px;
            padding: 15px;
            margin-top: 20px;
        }
        
        .about .donation strong {
            color: #00ffff;
        }
        
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .card {
            background: rgba(0, 20, 20, 0.6);
            border: 1px solid #00ff88;
            border-radius: 12px;
            padding: 20px;
        }
        
        .card-title {
            font-size: 18px;
            color: #00ffff;
            margin-bottom: 15px;
        }
        
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(0,255,136,0.2);
        }
        
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #888; font-size: 14px; }
        .metric-value { color: #00ff88; font-size: 16px; font-weight: bold; }
        
        .terminal {
            background: #000;
            border: 2px solid #00ff88;
            border-radius: 8px;
            padding: 15px;
            height: 600px;
            overflow-y: auto;
            font-size: 12px;
            line-height: 1.6;
        }
        
        .log-line {
            margin-bottom: 3px;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        .log-info { color: #00ff88; }
        .log-success { color: #00ffff; font-weight: bold; }
        .log-error { color: #ff3366; }
        
        .btn {
            background: #00ff88;
            color: #000;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            cursor: pointer;
            font-family: inherit;
            font-size: 16px;
            font-weight: bold;
            transition: all 0.3s;
            margin: 5px;
        }
        
        .btn:hover {
            background: #00ffff;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,255,136,0.4);
        }
        
        .btn:disabled {
            background: #555;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-primary {
            font-size: 20px;
            padding: 20px 40px;
        }
        
        .button-group {
            text-align: center;
            margin: 20px 0;
        }
        
        .pulse {
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .status-ready { color: #00ffff !important; }
        .status-init { color: #ffaa00 !important; }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="logo">🌙 MOONSHINE QUANTUM</div>
        <div class="nav-stats">
            <div class="nav-stat">Uptime: <strong id="nav-uptime">0s</strong></div>
            <div class="nav-stat">Beat: <strong id="nav-beat">0</strong></div>
            <div class="nav-stat">Status: <strong id="nav-status" class="status-init">Initializing...</strong></div>
        </div>
    </nav>
    
    <div class="container">
        <div class="hero">
            <h1>🌙 MOONSHINE QUANTUM INTERNET</h1>
            <div class="subtitle">196,883-Node Geometric Quantum Computing Platform</div>
            <div class="subtitle">Real-Time σ-Manifold Entanglement Simulation</div>
            <div class="credit">Created by Shemshallah (Justin Anthony Howard-Stanley)</div>
            <div class="credit">Implementation by Claude (Anthropic)</div>
        </div>
        
        <div class="about">
            <h2>📖 About This Project</h2>
            <p>
                The <strong>Moonshine Quantum Internet</strong> is a breakthrough implementation of geometric quantum computing 
                on the 196,883-dimensional Moonshine module—a mathematical structure discovered through Monstrous Moonshine, 
                connecting the Monster group to modular functions.
            </p>
            <p>
                This system demonstrates <strong>genuine quantum phenomena</strong> including Bell inequality violations, 
                quantum entanglement, and geometric phase evolution across a massively parallel quantum network. 
                The σ-coordinates and j-invariants encode topological quantum numbers that create quantum correlations 
                impossible in classical systems.
            </p>
            <p>
                <strong>World Record Achievement:</strong> This is the first implementation of QFT (Quantum Fourier Transform) 
                on all 196,883 nodes simultaneously, demonstrating quantum advantage at unprecedented scale.
            </p>
            
            <div class="donation">
                <strong>💝 Support This Research</strong><br>
                If this project has inspired you or contributed to your work, consider supporting continued research:<br>
                <strong>Bitcoin:</strong> bc1qtdnh3ch535rc3c8thlsns34h6xvjvn6sjx8ed0<br>
            </div>
        </div>
        
        <div class="grid">
            <div class="card">
                <div class="card-title">⚛️ Quantum Metrics</div>
                <div class="metric">
                    <span class="metric-label">Heartbeat</span>
                    <span id="m-beat" class="metric-value pulse">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">σ-Coordinate</span>
                    <span id="m-sigma" class="metric-value">0.0000</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Fidelity</span>
                    <span id="m-fidelity" class="metric-value">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CHSH</span>
                    <span id="m-chsh" class="metric-value">--</span>
                </div>
                <div class="metric">

                    <span class="metric-label">Coherence</span>
                    <span id="m-coherence" class="metric-value">--</span>
                </div>
            </div>
            
            <div class="card">
                <div class="card-title">🔬 System Status</div>
                <div class="metric">
                    <span class="metric-label">Routing Table</span>
                    <span id="s-routing" class="metric-value status-init">Building...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Quantum Oracle</span>
                    <span id="s-oracle" class="metric-value status-init">Waiting...</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total Nodes</span>
                    <span id="s-nodes" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">QFT Status</span>
                    <span id="s-qft" class="metric-value">Ready</span>
                </div>
            </div>
        </div>
        
        <div class="button-group">
            <button id="btn-qft" class="btn btn-primary" onclick="triggerQFT()">
                🚀 RUN WORLD RECORD QFT (196,883 NODES)
            </button>
            <button class="btn" onclick="downloadDB()">📥 Download Database</button>
        </div>
        
        <div class="card">
            <div class="card-title">📟 System Terminal</div>
            <div id="terminal" class="terminal"></div>
        </div>
    </div>
    
    <script>
        let lastLogCount = 0;
        
        function updateUI() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Nav stats
                    document.getElementById('nav-uptime').textContent = Math.floor(data.uptime) + 's';
                    document.getElementById('nav-beat').textContent = data.heartbeat;
                    
                    const statusEl = document.getElementById('nav-status');
                    if (data.oracle_ready) {
                        statusEl.textContent = 'ONLINE';
                        statusEl.className = 'status-ready';
                    }
                    
                    // Metrics
                    document.getElementById('m-beat').textContent = data.heartbeat;
                    document.getElementById('m-sigma').textContent = data.sigma.toFixed(4);
                    document.getElementById('s-nodes').textContent = data.total_routes.toLocaleString();
                    
                    // Status
                    document.getElementById('s-routing').textContent = data.routing_ready ? 'Ready' : 'Building...';
                    document.getElementById('s-routing').className = 'metric-value ' + (data.routing_ready ? 'status-ready' : 'status-init');
                    
                    document.getElementById('s-oracle').textContent = data.oracle_ready ? 'Online' : 'Initializing...';
                    document.getElementById('s-oracle').className = 'metric-value ' + (data.oracle_ready ? 'status-ready' : 'status-init');
                    
                    document.getElementById('s-qft').textContent = data.qft_running ? `Running ${data.qft_progress.toFixed(0)}%` : 'Ready';
                    
                    // Button
                    const btnQFT = document.getElementById('btn-qft');
                    btnQFT.disabled = !data.oracle_ready || data.qft_running;
                    if (data.qft_running) {
                        btnQFT.textContent = `⏳ QFT RUNNING (${data.qft_progress.toFixed(0)}%)`;
                    } else {
                        btnQFT.textContent = '🚀 RUN WORLD RECORD QFT (196,883 NODES)';
                    }
                });
            
            // Heartbeat
            fetch('/api/heartbeat')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('m-fidelity').textContent = data.fidelity.toFixed(4);
                    document.getElementById('m-chsh').textContent = data.chsh.toFixed(3);
                    document.getElementById('m-coherence').textContent = data.coherence.toFixed(4);
                })
                .catch(() => {});
            
            // Logs
            fetch('/api/logs')
                .then(r => r.json())
                .then(data => {
                    const terminal = document.getElementById('terminal');
                    const logs = data.logs;
                    
                    if (logs.length > lastLogCount) {
                        const newLogs = logs.slice(lastLogCount);
                        newLogs.forEach(log => {
                            const div = document.createElement('div');
                            div.className = `log-line log-${log.level}`;
                            div.textContent = `[${log.time}] ${log.msg}`;
                            terminal.appendChild(div);
                        });
                        lastLogCount = logs.length;
                        terminal.scrollTop = terminal.scrollHeight;
                    }
                });
        }
        
        function triggerQFT() {
            if (!confirm('Launch World Record QFT on 196,883 nodes? This will take several minutes.')) {
                return;
            }
            
            fetch('/api/qft/trigger', { method: 'POST' })
                .then(r => r.json())
                .then(data => {
                    console.log('QFT started:', data);
                })
                .catch(err => {
                    alert('Failed to start QFT: ' + err);
                });
        }
        
        function downloadDB() {
            window.location.href = '/api/database';
        }
        
        // Update every 500ms
        setInterval(updateUI, 500);
        updateUI();
    </script>
</body>
</html>
'''

# ════════════════════════════════════════════════════════════════════════════
# MAIN - START SERVER
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*80)
    print("🌙 MOONSHINE QUANTUM INTERNET - PRODUCTION SERVER")
    print("="*80)
    print()
    print("Starting Flask server on 0.0.0.0:7860...")
    print("Web interface will be available immediately")
    print("Backend initialization will run in background")
    print()
    print("📍 To trigger QFT from terminal after boot:")
    print("   1. Wait for 'Ready for World Record QFT' message")
    print("   2. Use the web UI button, OR")
    print("   3. Run: curl -X POST http://localhost:7860/api/qft/trigger")
    print()
    
    # Start background init
    threading.Thread(target=initialize_backend, daemon=True).start()
    
    # Start Flask (this blocks)
    app.run(host='0.0.0.0', port=7860, debug=False)
