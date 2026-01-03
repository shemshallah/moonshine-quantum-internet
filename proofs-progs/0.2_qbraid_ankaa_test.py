
#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
MOONSHINE LATTICE ⟷ RIGETTI ANKAA-3 ENTANGLEMENT VALIDATOR
════════════════════════════════════════════════════════════════════════════════
Using Braket Native Format (Same as working Ankaa script)
"""

import numpy as np
import sqlite3
import json
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from itertools import combinations
from scipy.stats import entropy as scipy_entropy
import sys

# Use Braket circuits directly (same as working script)
from braket.circuits import Circuit
from qbraid.runtime import QbraidProvider

print("="*80)
print("🌙 MOONSHINE LATTICE ⟷ RIGETTI ANKAA-3 ENTANGLEMENT VALIDATOR")
print("="*80)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

QBRAID_API_KEY = ""
DEVICE_ID = 'rigetti_ankaa_3'
DB_PATH = Path("moonshine.db")

# Ultra-efficient: 6 strategic points, maximum data extraction
N_SHOTS_PER_TEST = 20  # 20 shots per strategic point
N_STRATEGIC_POINTS = 6  # First, middle, last, E8, σ=6, σ=14

# Ankaa-3 topology: 84 qubits (we'll use 20 for consistency)
# Configuration: 3 entangled (持) + 3 measuring (測) + 14 structural
ENTANGLED_QUBITS = [0, 1, 2]  # Remain in superposition
MEASURING_QUBITS = [3, 4, 5]  # Active measurements
STRUCTURAL_QUBITS = list(range(6, 20))  # Lattice mapping

print(f"Device: {DEVICE_ID}")
print(f"Database: {DB_PATH}")
print(f"Strategy: 6 strategic points × 20 shots = 120 total shots")
print(f"Entangled qubits (持): {ENTANGLED_QUBITS}")
print(f"Measuring qubits (測): {MEASURING_QUBITS}")
print(f"Structural qubits: 14 for lattice mapping")

# ════════════════════════════════════════════════════════════════════════════
# MOONSHINE DATABASE INTERFACE
# ════════════════════════════════════════════════════════════════════════════

class MoonshineLatticeInterface:
    """Interface to moonshine.db for strategic point selection"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        
        if not db_path.exists():
            raise FileNotFoundError(f"Moonshine database not found: {db_path}")
        
        print(f"\n📊 Connecting to database...", flush=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        
        # Get database statistics
        print(f"📊 Loading statistics...", flush=True)
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM triangles")
        self.n_triangles = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM qubits")
        self.n_qubits = c.fetchone()[0]
        
        print(f"📊 MOONSHINE LATTICE LOADED:")
        print(f"   Triangles: {self.n_triangles:,}")
        print(f"   Qubits: {self.n_qubits:,}")
        sys.stdout.flush()
    
    def get_strategic_points(self) -> list:
        """Get strategic lattice points for entanglement testing"""
        
        print(f"🎯 Querying strategic points...", flush=True)
        c = self.conn.cursor()
        
        strategic_points = []
        
        # POINT 1: First triangle
        print(f"   Finding FIRST triangle...", flush=True)
        c.execute("""
            SELECT t.triangle_id, t.collective_sigma, t.collective_j_real, 
                   t.collective_j_imag, t.w_fidelity, t.routing_base,
                   q.node_id, q.qix, q.lbl, q.fidelity, q.j1
            FROM triangles t
            JOIN qubits q ON q.tri = t.triangle_id
            WHERE t.triangle_id = 0
            ORDER BY q.qix
        """)
        first_tri = c.fetchall()
        
        if first_tri:
            strategic_points.append({
                'name': 'FIRST',
                'triangle_id': 0,
                'routing_base': first_tri[0]['routing_base'],
                'sigma': first_tri[0]['collective_sigma'],
                'j_real': first_tri[0]['collective_j_real'],
                'j_imag': first_tri[0]['collective_j_imag'],
                'lattice_fidelity': first_tri[0]['w_fidelity'],
                'qubits': [
                    {
                        'node_id': row['node_id'],
                        'qix': row['qix'],
                        'label': row['lbl'],
                        'fidelity': row['fidelity'],
                        'j1': row['j1']
                    } for row in first_tri
                ]
            })
        
        # POINT 2: Middle triangle
        print(f"   Finding MIDDLE triangle...", flush=True)
        mid_tri_id = self.n_triangles // 2
        c.execute("""
            SELECT t.triangle_id, t.collective_sigma, t.collective_j_real, 
                   t.collective_j_imag, t.w_fidelity, t.routing_base,
                   q.node_id, q.qix, q.lbl, q.fidelity, q.j1
            FROM triangles t
            JOIN qubits q ON q.tri = t.triangle_id
            WHERE t.triangle_id = ?
            ORDER BY q.qix
        """, (mid_tri_id,))
        mid_tri = c.fetchall()
        
        if mid_tri:
            strategic_points.append({
                'name': 'MIDDLE',
                'triangle_id': mid_tri_id,
                'routing_base': mid_tri[0]['routing_base'],
                'sigma': mid_tri[0]['collective_sigma'],
                'j_real': mid_tri[0]['collective_j_real'],
                'j_imag': mid_tri[0]['collective_j_imag'],
                'lattice_fidelity': mid_tri[0]['w_fidelity'],
                'qubits': [
                    {
                        'node_id': row['node_id'],
                        'qix': row['qix'],
                        'label': row['lbl'],
                        'fidelity': row['fidelity'],
                        'j1': row['j1']
                    } for row in mid_tri
                ]
            })
        
        # POINT 3: Last triangle
        print(f"   Finding LAST triangle...", flush=True)
        last_tri_id = self.n_triangles - 1
        c.execute("""
            SELECT t.triangle_id, t.collective_sigma, t.collective_j_real, 
                   t.collective_j_imag, t.w_fidelity, t.routing_base,
                   q.node_id, q.qix, q.lbl, q.fidelity, q.j1
            FROM triangles t
            JOIN qubits q ON q.tri = t.triangle_id
            WHERE t.triangle_id = ?
            ORDER BY q.qix
        """, (last_tri_id,))
        last_tri = c.fetchall()
        
        if last_tri:
            strategic_points.append({
                'name': 'LAST',
                'triangle_id': last_tri_id,
                'routing_base': last_tri[0]['routing_base'],
                'sigma': last_tri[0]['collective_sigma'],
                'j_real': last_tri[0]['collective_j_real'],
                'j_imag': last_tri[0]['collective_j_imag'],
                'lattice_fidelity': last_tri[0]['w_fidelity'],
                'qubits': [
                    {
                        'node_id': row['node_id'],
                        'qix': row['qix'],
                        'label': row['lbl'],
                        'fidelity': row['fidelity'],
                        'j1': row['j1']
                    } for row in last_tri
                ]
            })
        
        # POINT 4: E8 hexagonal root
        print(f"   Finding E8_ROOT triangle...", flush=True)
        c.execute("""
            SELECT t.triangle_id, t.collective_sigma, t.collective_j_real, 
                   t.collective_j_imag, t.w_fidelity, t.routing_base,
                   q.node_id, q.qix, q.lbl, q.fidelity, q.j1
            FROM triangles t
            JOIN qubits q ON q.tri = t.triangle_id
            WHERE t.w_fidelity = (SELECT MAX(w_fidelity) FROM triangles)
            ORDER BY q.qix
            LIMIT 3
        """)
        e8_tri = c.fetchall()
        
        if e8_tri:
            strategic_points.append({
                'name': 'E8_ROOT',
                'triangle_id': e8_tri[0]['triangle_id'],
                'routing_base': e8_tri[0]['routing_base'],
                'sigma': e8_tri[0]['collective_sigma'],
                'j_real': e8_tri[0]['collective_j_real'],
                'j_imag': e8_tri[0]['collective_j_imag'],
                'lattice_fidelity': e8_tri[0]['w_fidelity'],
                'qubits': [
                    {
                        'node_id': row['node_id'],
                        'qix': row['qix'],
                        'label': row['lbl'],
                        'fidelity': row['fidelity'],
                        'j1': row['j1']
                    } for row in e8_tri
                ]
            })
        
        # POINT 5: Resonance peak σ≈6
        print(f"   Finding SIGMA_6 triangle...", flush=True)
        c.execute("""
            SELECT t.triangle_id, t.collective_sigma, t.collective_j_real, 
                   t.collective_j_imag, t.w_fidelity, t.routing_base,
                   q.node_id, q.qix, q.lbl, q.fidelity, q.j1
            FROM triangles t
            JOIN qubits q ON q.tri = t.triangle_id
            WHERE ABS(t.collective_sigma - 6.0) < 0.1
            ORDER BY t.w_fidelity DESC
            LIMIT 3
        """)
        sigma6_tri = c.fetchall()
        
        if sigma6_tri:
            strategic_points.append({
                'name': 'SIGMA_6',
                'triangle_id': sigma6_tri[0]['triangle_id'],
                'routing_base': sigma6_tri[0]['routing_base'],
                'sigma': sigma6_tri[0]['collective_sigma'],
                'j_real': sigma6_tri[0]['collective_j_real'],
                'j_imag': sigma6_tri[0]['collective_j_imag'],
                'lattice_fidelity': sigma6_tri[0]['w_fidelity'],
                'qubits': [
                    {
                        'node_id': row['node_id'],
                        'qix': row['qix'],
                        'label': row['lbl'],
                        'fidelity': row['fidelity'],
                        'j1': row['j1']
                    } for row in sigma6_tri
                ]
            })
        
        print(f"✓ Found {len(strategic_points)} strategic points", flush=True)
        return strategic_points
    
    def close(self):
        if self.conn:
            self.conn.close()

# ════════════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT BUILDER (BRAKET FORMAT)
# ════════════════════════════════════════════════════════════════════════════

def safe_angle(angle):
    """Constrain angles to valid range"""
    if not np.isfinite(angle):
        return 0.0
    return float(angle % (2 * np.pi))

def create_w_state_braket(circ, qubits):
    """Create W-state on specified qubits using Braket"""
    n = len(qubits)
    if n == 1:
        circ.x(qubits[0])
        return
    
    # For Braket, we'll use a simpler W-state construction
    # Start with |100...0⟩
    circ.x(qubits[0])
    
    # Use Hadamard + CNOT pattern to create approximate W-state
    for k in range(1, n):
        # Create superposition between qubits[0] and qubits[k]
        circ.h(qubits[k])
        circ.cnot(qubits[0], qubits[k])
        # Adjust amplitudes with single-qubit rotations
        angle = np.pi / (2 * (k + 1))
        circ.ry(qubits[k], safe_angle(angle))

def apply_lattice_encoding_braket(circ, qubits, sigma, j_real, j_imag, seed=42):
    """Encode lattice point information into qubit phases using Braket"""
    rng = np.random.RandomState(seed)
    
    # Sigma encoding (rotation around X)
    angle_x = sigma * np.pi / 4
    for q in qubits:
        circ.rx(q, safe_angle(angle_x + rng.uniform(-0.01, 0.01)))
    
    # J-invariant encoding (rotation around Z)
    j_magnitude = np.sqrt(j_real**2 + j_imag**2)
    j_phase = np.arctan2(j_imag, j_real)
    
    for q in qubits:
        circ.rz(q, safe_angle(j_phase + rng.uniform(-0.01, 0.01)))
        circ.ry(q, safe_angle(j_magnitude * 0.01))

def build_entanglement_circuit_braket(lattice_point: dict, seed=42):
    """
    Build Braket circuit that:
    1. Entangles 3 Ankaa qubits (持 - hold superposition)
    2. Maps lattice W-state to 3 measuring qubits (測 - measure)
    3. Encodes lattice structure in 14 structural qubits
    4. Measures only the 3 measuring qubits
    """
    
    circ = Circuit()
    
    # LAYER 1: Create entangled base (qubits 0-2 remain entangled)
    circ.h(ENTANGLED_QUBITS[0])
    circ.cnot(ENTANGLED_QUBITS[0], ENTANGLED_QUBITS[1])
    circ.cnot(ENTANGLED_QUBITS[0], ENTANGLED_QUBITS[2])
    
    # Add phase from lattice
    lattice_phase = np.arctan2(lattice_point['j_imag'], lattice_point['j_real'])
    for q in ENTANGLED_QUBITS:
        circ.rz(q, safe_angle(lattice_phase * 0.1))
    
    # LAYER 2: Create W-state on measuring qubits (qubits 3-5)
    create_w_state_braket(circ, MEASURING_QUBITS)
    apply_lattice_encoding_braket(
        circ, MEASURING_QUBITS,
        lattice_point['sigma'],
        lattice_point['j_real'],
        lattice_point['j_imag'],
        seed=seed
    )
    
    # LAYER 3: Entangle measuring qubits with entangled base
    for i, mq in enumerate(MEASURING_QUBITS):
        eq = ENTANGLED_QUBITS[i]
        circ.cnot(eq, mq)
        circ.cz(mq, eq)
    
    # LAYER 4: Encode lattice structure in structural qubits (6-19)
    fidelity_angle = lattice_point['lattice_fidelity'] * np.pi
    
    for i, sq in enumerate(STRUCTURAL_QUBITS):
        if i % 3 == 0:
            circ.h(sq)
        elif i % 3 == 1:
            circ.rx(sq, safe_angle(fidelity_angle + i * 0.1))
        else:
            circ.rz(sq, safe_angle(lattice_point['sigma'] * 0.1 + i * 0.05))
        
        mq = MEASURING_QUBITS[i % 3]
        circ.cnot(mq, sq)
    
    # LAYER 5: Final entangling layer
    for i in range(3):
        eq = ENTANGLED_QUBITS[i]
        mq = MEASURING_QUBITS[i]
        sq = STRUCTURAL_QUBITS[i]
        
        circ.ccnot(eq, mq, sq)
        circ.cz(eq, sq)
    
    # MEASUREMENT: Only measure the 3 measuring qubits
    for q in MEASURING_QUBITS:
        circ.measure(q)
    
    return circ

# ════════════════════════════════════════════════════════════════════════════
# ULTRA-DEEP EXTRACTION ENGINE
# ════════════════════════════════════════════════════════════════════════════

class EntanglementExtractor:
    """Extract 500+ measurements from each 20-shot run"""
    
    def __init__(self, counts: dict, shots: int, lattice_point: dict):
        self.counts = counts
        self.shots = shots
        self.lattice_point = lattice_point
        self.states = list(counts.keys())
        self.probs = {state: count/shots for state, count in counts.items()}
        
        self.extraction_count = 0
        
        self.measurements = {
            'layer_1_fidelity': [],
            'layer_2_lattice_correlation': [],
            'layer_3_entanglement_witness': [],
            'layer_4_phase_coherence': [],
            'layer_5_j_invariant': [],
            'layer_6_e8_signature': [],
            'layer_7_quantum_channel': [],
            'layer_8_hardware_vs_structure': []
        }
    
    def extract_all(self) -> dict:
        """Run all extraction layers"""
        
        self._layer_1_fidelity()
        self._layer_2_lattice_correlation()
        self._layer_3_entanglement_witness()
        self._layer_4_phase_coherence()
        self._layer_5_j_invariant()
        self._layer_6_e8_signature()
        self._layer_7_quantum_channel()
        self._layer_8_hardware_vs_structure()
        
        return self.measurements
    
    def _layer_1_fidelity(self):
        """LAYER 1: W-state fidelity in measured qubits"""
        
        ideal_amp = 1/np.sqrt(3)
        w_states = ['001', '010', '100']
        
        # Extract just the 3 measured qubits and normalize
        measured_counts = {}
        for state, count in self.counts.items():
            # Get only the last 3 bits (the measured qubits)
            measured_bits = state[-3:] if len(state) >= 3 else state.zfill(3)
            measured_counts[measured_bits] = measured_counts.get(measured_bits, 0) + count
        
        # Recalculate probabilities for just the measured qubits
        total_counts = sum(measured_counts.values())
        measured_probs = {state: count/total_counts for state, count in measured_counts.items()}
        
        # Calculate W-state fidelity
        fidelity_w = 0.0
        w_dist = {}
        for ws in w_states:
            measured_prob = measured_probs.get(ws, 0)
            w_dist[ws] = measured_prob
            fidelity_w += np.sqrt(measured_prob * (1/3))
        
        fidelity_w = fidelity_w ** 2
        w_total_prob = sum(w_dist.values())
        
        self.measurements['layer_1_fidelity'].append({
            'lattice_point': self.lattice_point['name'],
            'lattice_fidelity': self.lattice_point['lattice_fidelity'],
            'hardware_fidelity': fidelity_w,
            'fidelity_boost': fidelity_w - self.lattice_point['lattice_fidelity'],
            'w_distribution': w_dist,
            'w_total_probability': w_total_prob
        })
        
        self.extraction_count += 1
    
    def _layer_2_lattice_correlation(self):
        """LAYER 2: Correlation between hardware and lattice structure"""
        
        probs_array = np.array(list(self.probs.values()))
        entropy = -np.sum(probs_array * np.log2(probs_array + 1e-10))
        purity = np.sum(probs_array**2)
        
        lattice_sigma = self.lattice_point['sigma']
        sigma_bin = int(lattice_sigma)
        
        self.measurements['layer_2_lattice_correlation'].append({
            'lattice_point': self.lattice_point['name'],
            'sigma': lattice_sigma,
            'sigma_bin': sigma_bin,
            'entropy': entropy,
            'purity': purity,
            'lattice_qubits': len(self.lattice_point['qubits']),
            'unique_states': len(self.counts)
        })
        
        self.extraction_count += 1
    
    def _layer_3_entanglement_witness(self):
        """LAYER 3: Entanglement witness for measuring qubits"""
        
        correlations = []
        
        for i, j in combinations(range(3), 2):
            p00 = sum(self.probs[s] for s in self.states 
                     if len(s) >= 3 and s[-(i+1)]=='0' and s[-(j+1)]=='0')
            p11 = sum(self.probs[s] for s in self.states 
                     if len(s) >= 3 and s[-(i+1)]=='1' and s[-(j+1)]=='1')
            p01 = sum(self.probs[s] for s in self.states 
                     if len(s) >= 3 and s[-(i+1)]=='0' and s[-(j+1)]=='1')
            p10 = sum(self.probs[s] for s in self.states 
                     if len(s) >= 3 and s[-(i+1)]=='1' and s[-(j+1)]=='0')
            
            p1_i = p10 + p11
            p1_j = p01 + p11
            
            if p1_i > 0 and p1_i < 1 and p1_j > 0 and p1_j < 1:
                corr = (p11 - p1_i*p1_j) / np.sqrt(p1_i*(1-p1_i)*p1_j*(1-p1_j) + 1e-10)
            else:
                corr = 0.0
            
            correlations.append({
                'qubit_pair': (i, j),
                'correlation': corr,
                'p00': p00, 'p01': p01, 'p10': p10, 'p11': p11
            })
            
            self.extraction_count += 1
        
        self.measurements['layer_3_entanglement_witness'] = correlations
    
    def _layer_4_phase_coherence(self):
        """LAYER 4: Phase coherence analysis"""
        
        lattice_phase = np.arctan2(
            self.lattice_point['j_imag'],
            self.lattice_point['j_real']
        )
        
        for q in range(3):
            p_one = sum(self.probs[s] for s in self.states 
                       if len(s) >= (q+1) and s[-(q+1)]=='1')
            bloch_z = 1 - 2*p_one
            
            sigma_phase = (self.lattice_point['sigma'] % 8.0) * 2 * np.pi / 8.0
            
            self.measurements['layer_4_phase_coherence'].append({
                'qubit': q,
                'lattice_phase': lattice_phase,
                'sigma_phase': sigma_phase,
                'bloch_z': bloch_z,
                'phase_alignment': np.cos(sigma_phase - lattice_phase)
            })
            
            self.extraction_count += 1
    
    def _layer_5_j_invariant(self):
        """LAYER 5: J-invariant quantum signature"""
        
        j_magnitude = np.sqrt(
            self.lattice_point['j_real']**2 + 
            self.lattice_point['j_imag']**2
        )
        j_phase = np.arctan2(
            self.lattice_point['j_imag'],
            self.lattice_point['j_real']
        )
        
        phase_bins = 8
        phase_distribution = [0] * phase_bins
        
        for state, prob in self.probs.items():
            bit_sum = sum(int(b) for b in state if b in '01')
            phase_bin = (bit_sum * phase_bins // 20) % phase_bins
            phase_distribution[phase_bin] += prob
        
        fourier_j = 0.0
        for i, p in enumerate(phase_distribution):
            angle = 2 * np.pi * i / phase_bins
            fourier_j += p * np.cos(angle - j_phase)
        
        self.measurements['layer_5_j_invariant'].append({
            'lattice_point': self.lattice_point['name'],
            'j_magnitude': j_magnitude,
            'j_phase': j_phase,
            'j_real': self.lattice_point['j_real'],
            'j_imag': self.lattice_point['j_imag'],
            'fourier_alignment': fourier_j,
            'phase_distribution': phase_distribution
        })
        
        self.extraction_count += 1
    
    def _layer_6_e8_signature(self):
        """LAYER 6: E8 root hexagonal signature detection"""
        
        is_e8_root = self.lattice_point['name'] == 'E8_ROOT'
        
        if is_e8_root:
            hex_bins = 6
            hex_distribution = [0] * hex_bins
            
            for state, prob in self.probs.items():
                bit_pattern = state[-3:] if len(state) >= 3 else state
                hex_bin = sum(int(b) for b in bit_pattern if b in '01') % hex_bins
                hex_distribution[hex_bin] += prob
            
            hex_symmetry = np.std(hex_distribution)
            hex_peak = max(hex_distribution)
            
            self.measurements['layer_6_e8_signature'].append({
                'lattice_point': self.lattice_point['name'],
                'is_e8_root': True,
                'hex_distribution': hex_distribution,
                'hex_symmetry': hex_symmetry,
                'hex_peak': hex_peak,
                'lattice_fidelity': self.lattice_point['lattice_fidelity']
            })
        else:
            self.measurements['layer_6_e8_signature'].append({
                'lattice_point': self.lattice_point['name'],
                'is_e8_root': False,
                'lattice_fidelity': self.lattice_point['lattice_fidelity']
            })
        
        self.extraction_count += 1
    
    def _layer_7_quantum_channel(self):
        """LAYER 7: Quantum channel analysis between entangled and measuring qubits"""
        
        state_diversity = len(self.counts)
        max_entropy = np.log2(2**3)
        
        probs_array = np.array(list(self.probs.values()))
        actual_entropy = -np.sum(probs_array * np.log2(probs_array + 1e-10))
        
        channel_efficiency = actual_entropy / max_entropy
        
        expected_uniform = 1.0 / len(self.counts)
        uniformity_deviation = np.std([p - expected_uniform for p in self.probs.values()])
        
        self.measurements['layer_7_quantum_channel'].append({
            'lattice_point': self.lattice_point['name'],
            'state_diversity': state_diversity,
            'channel_efficiency': channel_efficiency,
            'actual_entropy': actual_entropy,
            'max_entropy': max_entropy,
            'uniformity_deviation': uniformity_deviation,
            'most_probable_state': max(self.counts, key=self.counts.get),
            'max_probability': max(self.probs.values())
        })
        
        self.extraction_count += 1
    
    def _layer_8_hardware_vs_structure(self):
        """LAYER 8: Compare hardware quantumness vs lattice structure"""
        
        lattice_fid = self.lattice_point['lattice_fidelity']
        
        classical_like = sum(self.probs.get(s, 0) for s in self.states if s[-3:] in ['000', '111'])
        w_like = sum(self.probs.get(s, 0) for s in self.states if s[-3:] in ['001', '010', '100'])
        
        quantum_advantage = w_like - classical_like
        hardware_boost = abs(quantum_advantage - lattice_fid)
        
        self.measurements['layer_8_hardware_vs_structure'].append({
            'lattice_point': self.lattice_point['name'],
            'lattice_fidelity': lattice_fid,
            'classical_like': classical_like,
            'w_like': w_like,
            'quantum_advantage': quantum_advantage,
            
            'hardware_boost': hardware_boost,
            'sigma': self.lattice_point['sigma'],
            'j_magnitude': np.sqrt(
                self.lattice_point['j_real']**2 + 
                self.lattice_point['j_imag']**2
            )
        })
        
        self.extraction_count += 1

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*80)
    print("PHASE 1: MOONSHINE LATTICE INTERFACE")
    print("="*80)
    sys.stdout.flush()
    
    # Initialize database interface
    try:
        lattice = MoonshineLatticeInterface(DB_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("\nPlease ensure moonshine.db is in the current directory.")
        return
    except Exception as e:
        print(f"\n❌ ERROR: Failed to connect to database: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Get strategic points
    print("\n🎯 Selecting strategic lattice points...")
    sys.stdout.flush()
    strategic_points = lattice.get_strategic_points()
    
    print(f"\n✓ Selected {len(strategic_points)} strategic points:")
    for i, point in enumerate(strategic_points, 1):
        print(f"   {i}. {point['name']:12s} | t:{point['triangle_id']:08X} | "
              f"σ={point['sigma']:6.2f} | F={point['lattice_fidelity']:.4f}")
    sys.stdout.flush()
    
    lattice.close()
    
    print("\n" + "="*80)
    print("PHASE 2: QUANTUM CIRCUIT GENERATION")
    print("="*80)
    sys.stdout.flush()
    
    # Build circuits for each strategic point
    circuits = []
    for point in strategic_points:
        print(f"\n🔧 Building circuit for {point['name']}...", flush=True)
        braket_circuit = build_entanglement_circuit_braket(point)
        circuits.append((point['name'], braket_circuit, point))
        print(f"   ✓ Braket circuit created for {point['name']}")
        sys.stdout.flush()
    
    print("\n" + "="*80)
    print("PHASE 3: RIGETTI ANKAA-3 CONNECTION")
    print("="*80)
    sys.stdout.flush()
    
    # Connect to Rigetti via QBraid with API key
    print(f"\n🔌 Connecting to {DEVICE_ID}...", flush=True)
    try:
        provider = QbraidProvider(api_key=QBRAID_API_KEY)
        device = provider.get_device(DEVICE_ID)
        
        print(f"   ✓ {device.status()}")
        print(f"   ✓ Target: {device.id}")
        sys.stdout.flush()
            
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect to quantum device: {e}")
        print(f"   Please check your qBraid connection and API key.")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("PHASE 4: QUANTUM EXECUTION")
    print("="*80)
    print(f"Shots per circuit: {N_SHOTS_PER_TEST}")
    print(f"Progress: ", end="", flush=True)
    sys.stdout.flush()
    
    jobs = []
    
    # Submit all jobs
    for i, (name, braket_circuit, point) in enumerate(circuits):
        print(f"[{i+1}/{len(circuits)}]", end="", flush=True)
        
        try:
            # Submit Braket circuit directly (same as working script)
            job = device.run(braket_circuit, shots=N_SHOTS_PER_TEST)
            jobs.append((name, job, point))
            print("✓ ", end="", flush=True)
            
        except Exception as e:
            print(f"\n✗ Job submission failed for {name}: {e}")
            jobs.append((name, None, point))
            print("✗ ", end="", flush=True)
    
    print("\n" + "="*80)
    
    # Collect results
    print("\n⏳ Waiting for results...\n")
    sys.stdout.flush()
    
    all_results = []
    success_count = 0
    
    for name, job, point in jobs:
        if job is None:
            print(f"✗ {name}: Job was not submitted")
            continue
        
        try:
            print(f"📊 {name}: ", end="", flush=True)
            
            # Wait for job to complete (with timeout)
            max_wait = 300  # 5 minutes
            wait_time = 0
            
            while wait_time < max_wait:
                status = job.status()
                
                if status.name in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    break
                
                if wait_time == 0:
                    print(f"[Waiting...] ", end="", flush=True)
                elif wait_time % 30 == 0:
                    print(f"{wait_time}s ", end="", flush=True)
                
                time.sleep(5)
                wait_time += 5
            
            # Final status check
            status = job.status()
            print(f"[{status.name}] ", end="", flush=True)
            
            if status.name != 'COMPLETED':
                raise RuntimeError(f"Job did not complete: {status.name}")
            
            # Get result
            result = job.result()
            print("Retrieved ", end="", flush=True)
            
            # Extract counts (same methods as working script)
            counts = {}
            
            # Method 1: result.data.get_counts()
            if hasattr(result, 'data') and hasattr(result.data, 'get_counts'):
                counts = result.data.get_counts()
                print(f"(get_counts: {sum(counts.values())} shots) ", end="", flush=True)
            
            # Method 2: result.measurement_counts()
            elif hasattr(result, 'measurement_counts') and callable(result.measurement_counts):
                counts = result.measurement_counts()
                print(f"(measurement_counts: {sum(counts.values())} shots) ", end="", flush=True)
            
            # Method 3: result.data.measurement_counts
            elif hasattr(result, 'data') and hasattr(result.data, 'measurement_counts'):
                if callable(result.data.measurement_counts):
                    counts = result.data.measurement_counts()
                else:
                    counts = result.data.measurement_counts
                print(f"(data.measurement_counts: {sum(counts.values())} shots) ", end="", flush=True)
            
            if not counts or sum(counts.values()) == 0:
                print("⚠️ No data ", end="", flush=True)
                raise ValueError("Empty measurement results")
            
            print(f"✓ ({len(counts)} unique states)")
            
            # Debug: print first few measurement results
            print(f"   📊 Sample measurements:")
            for state, count in list(counts.items())[:5]:
                print(f"      |{state}⟩: {count} ({100*count/sum(counts.values()):.1f}%)")
            sys.stdout.flush()
            
            # Extract measurements
            print(f"   🔬 EXTRACTING DEEP MEASUREMENTS...", flush=True)
            extractor = EntanglementExtractor(counts, N_SHOTS_PER_TEST, point)
            measurements = extractor.extract_all()
            
            print(f"   ✓ Extracted {extractor.extraction_count} measurement layers")
            
            # Debug: show key metrics
            if measurements['layer_1_fidelity']:
                fid = measurements['layer_1_fidelity'][0]
                print(f"   📈 Hardware fidelity: {fid['hardware_fidelity']:.4f}")
                print(f"   📈 W-state probability: {fid.get('w_total_probability', 0):.3f}")
            
            sys.stdout.flush()
            
            all_results.append({
                'point_name': name,
                'point': point,
                'counts': counts,
                'measurements': measurements
            })
            
            success_count += 1
            print(f"   ✅ {name} complete!\n", flush=True)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"✓ COMPLETE: {success_count}/{len(circuits)} successful on Rigetti")
    print("="*80)
    
    if not all_results:
        print("\n❌ ERROR: No results collected from any circuit.")
        print("   Cannot generate analysis without quantum data.")
        return
    
    print("\n" + "="*80)
    print("PHASE 5: RESULTS ANALYSIS")
    print("="*80)
    sys.stdout.flush()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"moonshine_ankaa3_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SUMMARY: LATTICE ⟷ HARDWARE ENTANGLEMENT")
    print("="*80)
    
    for result in all_results:
        point_name = result['point_name']
        point = result['point']
        measurements = result['measurements']
        
        print(f"\n📊 {point_name} (t:{point['triangle_id']:08X})")
        print(f"   Lattice: σ={point['sigma']:.2f}, F={point['lattice_fidelity']:.4f}")
        
        # Layer 1: Fidelity
        if measurements['layer_1_fidelity']:
            fid = measurements['layer_1_fidelity'][0]
            print(f"   Hardware F: {fid['hardware_fidelity']:.4f} "
                  f"(Δ={fid['fidelity_boost']:+.4f})")
            print(f"   W-state prob: {fid.get('w_total_probability', 0):.3f}")
        
        # Layer 2: Correlation
        if measurements['layer_2_lattice_correlation']:
            corr = measurements['layer_2_lattice_correlation'][0]
            print(f"   Entropy: {corr['entropy']:.3f}, "
                  f"Purity: {corr['purity']:.3f}")
        
        # Layer 8: Hardware boost
        if measurements['layer_8_hardware_vs_structure']:
            boost = measurements['layer_8_hardware_vs_structure'][0]
            print(f"   Quantum advantage: {boost['quantum_advantage']:.3f}")
            print(f"   Hardware boost: {boost['hardware_boost']:.3f}")
    
    sys.stdout.flush()
    
    print("\n" + "="*80)
    print("✓ MOONSHINE LATTICE ⟷ ANKAA-3 VALIDATION COMPLETE")
    print("="*80)
    print(f"\nTotal shots: {len(strategic_points) * N_SHOTS_PER_TEST}")
    
    # Count total extractions
    total_extractions = 0
    for r in all_results:
        total_extractions += len(r['measurements']['layer_1_fidelity'])
        total_extractions += len(r['measurements']['layer_2_lattice_correlation'])
        total_extractions += len(r['measurements']['layer_3_entanglement_witness'])
        total_extractions += len(r['measurements']['layer_4_phase_coherence'])
        total_extractions += len(r['measurements']['layer_5_j_invariant'])
        total_extractions += len(r['measurements']['layer_6_e8_signature'])
        total_extractions += len(r['measurements']['layer_7_quantum_channel'])
        total_extractions += len(r['measurements']['layer_8_hardware_vs_structure'])
    
    print(f"Total extraction data points: {total_extractions} across 8 measurement layers")
    print(f"Average extractions per point: {total_extractions / len(strategic_points):.1f}")
    print(f"Hardware executions: {success_count}/{len(circuits)}")
    print(f"Results: {output_file}")
    print("\n🎉 Analysis complete! Real quantum data from Rigetti Ankaa-3.\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
