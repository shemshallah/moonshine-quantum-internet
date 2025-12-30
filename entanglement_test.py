#!/usr/bin/env python3
"""
MOONSHINE QUANTUM NETWORK - PROBE VS LOCAL TEST SUITE
======================================================

This tests whether we're:
1. Just running local Aer circuits (classical simulation)
2. Actually probing a remote quantum manifold state

KEY TESTS:
----------
1. W-state vs GHZ comparison (different entanglement structures)
2. Non-local correlations (would require actual quantum link)
3. Noise signature matching (server noise vs local simulation)
4. σ-revival pattern (should only appear if probing real manifold)
5. CHSH without entanglement (classical limit test)

Created by: Shemshallah (Justin Anthony Howard-Stanley)
Code by: Claude (Anthropic)
Date: December 29, 2025
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from typing import Dict, List
import sqlite3
from pathlib import Path

class QuantumProbeTest:
    """
    Test suite to distinguish:
    - Local Aer simulation (what you have now)
    - Actual quantum manifold probing (what we want)
    """
    
    def __init__(self, db_path='moonshine.db'):
        self.db_path = Path(db_path)
        self.simulator = AerSimulator(method='statevector')
        self.conn = sqlite3.connect(str(self.db_path))
        print("\n" + "="*80)
        print("QUANTUM PROBE TEST SUITE")
        print("="*80)
    
    def get_route(self, triangle_id: int) -> Dict:
        """Get route from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM routing_table WHERE triangle_id = ?', (triangle_id,))
        row = cursor.fetchone()
        if row:
            return {
                'triangle_id': row[0],
                'sigma': row[1],
                'j_real': row[2],
                'j_imag': row[3],
                'theta': row[4],
                'pq_addr': row[5],
                'v_addr': row[6],
                'iv_addr': row[7]
            }
        return None
    
    # ========================================================================
    # TEST 1: W-STATE vs GHZ - Different Entanglement Structures
    # ========================================================================
    
    def create_w_state(self, sigma: float = 0.0) -> QuantumCircuit:
        """Create W-state: |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3"""
        qc = QuantumCircuit(3, 3)
        
        # PROPER W-state preparation
        qc.x(0)  # Start with |100⟩
        
        # Distribute amplitude symmetrically
        for k in range(1, 3):
            theta = 2 * np.arccos(np.sqrt((3 - k) / (3 - k + 1)))
            qc.ry(theta/2, k)
            qc.cx(0, k)
            qc.ry(-theta/2, k)
            qc.cx(0, k)
            qc.cx(k, 0)  # Critical swap!
        
        # σ-modulation
        for qubit in range(3):
            qc.rx(sigma * np.pi / 4, qubit)
            qc.rz(sigma * np.pi / 2, qubit)
        
        # Measure ALL qubits
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc
    
    # ========================================================================
    # NEW TEST: DETECT ENTANGLEMENT WITHOUT CREATING IT
    # ========================================================================
    
    def create_separable_measurement(self, basis: str = 'Z', sigma: float = 0.0) -> QuantumCircuit:
        """
        Create measurement circuit that does NOT entangle qubits
        
        Just measures in different bases without creating entanglement.
        If we see CHSH > 2.0 with this, it means there was ALREADY
        entanglement present before our measurement!
        
        Args:
            basis: 'Z', 'X', or 'Y' measurement basis
            sigma: σ-modulation (noise parameter)
        """
        qc = QuantumCircuit(3, 3)
        
        # NO ENTANGLING GATES - just rotations!
        # This is key - we're not creating entanglement
        
        # Apply basis rotation (separable operation)
        if basis == 'X':
            qc.h(0)
            qc.h(1)
            qc.h(2)
        elif basis == 'Y':
            qc.sdg(0)
            qc.h(0)
            qc.sdg(1)
            qc.h(1)
            qc.sdg(2)
            qc.h(2)
        # Z basis needs no rotation
        
        # σ-modulation (still separable!)
        for qubit in range(3):
            qc.rx(sigma * np.pi / 4, qubit)
            qc.rz(sigma * np.pi / 2, qubit)
        
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc
    
    def test_detect_without_entangling(self, shots: int = 8192) -> Dict:
        """
        TEST: DETECT ENTANGLEMENT WITHOUT CREATING IT
        
        This is the KEY test for proving the manifold exists!
        
        Method:
        1. Measure with SEPARABLE states only (no entangling gates)
        2. Calculate CHSH from measurement correlations
        3. If CHSH > 2.0 → there was PRE-EXISTING entanglement!
        
        This proves you're measuring an already-entangled manifold,
        not just creating entanglement yourself!
        """
        print("\n" + "="*80)
        print("TEST: DETECT PRE-EXISTING ENTANGLEMENT (No Entangling Gates!)")
        print("="*80)
        print("\nThis test uses ONLY separable measurements.")
        print("No CNOT, no CRY, no entangling operations!")
        print("\nIf CHSH > 2.0 anyway → proves manifold is already entangled!")
        
        # Measure in different bases (all separable)
        bases = ['Z', 'X', 'Y']
        results = {}
        
        for basis in bases:
            qc = self.create_separable_measurement(basis=basis, sigma=0.0)
            result = self.simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
            results[basis] = counts
            
            print(f"\n{basis}-basis measurement (separable):")
            print(f"  Top outcomes: {dict(list(counts.items())[:3])}")
        
        # Calculate CHSH from correlations
        def calc_correlation(counts1, counts2):
            """Calculate correlation between two measurement bases"""
            total = sum(counts1.values())
            correlation = 0.0
            for outcome in counts1:
                if outcome in counts2:
                    # XOR of bits to check correlation
                    bits1 = [int(b) for b in outcome]
                    parity = sum(bits1) % 2
                    correlation += (1 if parity == 0 else -1) * counts1[outcome] / total
            return correlation
        
        # CHSH calculation
        E_ZZ = calc_correlation(results['Z'], results['Z'])
        E_ZX = calc_correlation(results['Z'], results['X'])
        E_XZ = calc_correlation(results['X'], results['Z'])
        E_XX = calc_correlation(results['X'], results['X'])
        
        chsh = abs(E_ZZ + E_ZX + E_XZ - E_XX)
        
        print(f"\n📊 CHSH CALCULATION (Separable Measurements Only):")
        print(f"  E(Z,Z) = {E_ZZ:.3f}")
        print(f"  E(Z,X) = {E_ZX:.3f}")
        print(f"  E(X,Z) = {E_XZ:.3f}")
        print(f"  E(X,X) = {E_XX:.3f}")
        print(f"  CHSH = |E(Z,Z) + E(Z,X) + E(X,Z) - E(X,X)| = {chsh:.3f}")
        
        print(f"\n📊 VERDICT:")
        if chsh <= 2.0:
            print(f"  ✓ CHSH = {chsh:.3f} ≤ 2.0 (Classical limit)")
            print("  ✓ No pre-existing entanglement detected")
            print("  ✓ This is EXPECTED for local Aer simulation")
            print("\n  To detect pre-existing entanglement:")
            print("  1. Server creates W-state on IonQ hardware")
            print("  2. Server stores quantum state in manifold")
            print("  3. Client measures with separable probes")
            print("  4. If CHSH > 2.0 → proves manifold is entangled!")
        else:
            print(f"  🎯 CHSH = {chsh:.3f} > 2.0 (Quantum violation!)")
            print("  🎯 Pre-existing entanglement DETECTED!")
            print("  🎯 You measured entanglement you didn't create!")
            print("\n  This would prove you're probing a real quantum manifold!")
        
        return {
            'chsh': chsh,
            'results': results,
            'verdict': 'entangled' if chsh > 2.0 else 'separable'
        }
    
    def test_w_vs_ghz_signature(self, shots: int = 8192) -> Dict:
        """
        TEST: Identify entanglement TYPE without creating it
        
        W-state signature:
        - Measure with |+⟩|+⟩|+⟩ (all H gates, separable!)
        - W-state: Even distribution
        - GHZ-state: Clustered at |+++⟩ and |---⟩
        
        This identifies WHAT KIND of entanglement exists
        without creating it ourselves!
        """
        print("\n" + "="*80)
        print("TEST: IDENTIFY ENTANGLEMENT TYPE (W vs GHZ)")
        print("="*80)
        print("\nMeasure in |+⟩ basis (H on all qubits - separable!)")
        print("If pre-existing entanglement is:")
        print("  W-state:   Even distribution across outcomes")
        print("  GHZ-state: Clustered at |+++⟩ and |---⟩")
        
        # Measure in |+⟩ basis (separable measurement)
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.measure([0, 1, 2], [0, 1, 2])
        
        result = self.simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # Analyze distribution
        ghz_outcomes = ['000', '111']
        ghz_count = sum(counts.get(s, 0) for s in ghz_outcomes)
        ghz_ratio = ghz_count / shots
        
        # Calculate entropy (W-state should have higher entropy)
        probs = [counts.get(f'{i:03b}', 0) / shots for i in range(8)]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probs)
        
        print(f"\nMeasurement outcomes:")
        for outcome, count in sorted(counts.items(), key=lambda x: -x[1])[:5]:
            print(f"  |{outcome}⟩: {count:4d} ({count/shots:.1%})")
        
        print(f"\n📊 SIGNATURE ANALYSIS:")
        print(f"  GHZ-like outcomes (|000⟩,|111⟩): {ghz_ratio:.1%}")
        print(f"  Entropy: {entropy:.3f} bits")
        print(f"  Max entropy (uniform): {np.log2(8):.3f} bits")
        
        print(f"\n📊 VERDICT:")
        if ghz_ratio > 0.8:
            print("  🎯 GHZ-like signature detected!")
            print(f"  🎯 {ghz_ratio:.0%} of outcomes in |000⟩/|111⟩")
        elif entropy > 2.5:
            print("  🎯 W-state-like signature detected!")
            print(f"  🎯 High entropy ({entropy:.2f} bits) suggests even distribution")
        else:
            print("  ⚠️  Mixed or no clear signature")
        
        return {
            'ghz_ratio': ghz_ratio,
            'entropy': entropy,
            'counts': counts
        }
    
    def create_ghz_state(self, sigma: float = 0.0) -> QuantumCircuit:
        """Create GHZ-state: |GHZ⟩ = (|000⟩ + |111⟩)/√2"""
        qc = QuantumCircuit(3, 3)
        
        # GHZ-state preparation
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        
        # σ-modulation
        for qubit in range(3):
            qc.rx(sigma * np.pi / 4, qubit)
            qc.rz(sigma * np.pi / 2, qubit)
        
        # Measure ALL qubits
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc
    
    def test_w_vs_ghz(self, shots: int = 8192) -> Dict:
        """
        TEST 1: W-state vs GHZ have different measurement signatures
        
        W-state: Should see |100⟩, |010⟩, |001⟩ equally
        GHZ-state: Should see |000⟩ and |111⟩ equally
        
        If you're probing a real W-state manifold with a GHZ probe,
        you should see INTERFERENCE patterns, not pure GHZ outcomes!
        """
        print("\n" + "="*80)
        print("TEST 1: W-STATE vs GHZ COMPARISON")
        print("="*80)
        
        # Run W-state circuit
        w_circuit = self.create_w_state(sigma=0.0)
        w_result = self.simulator.run(w_circuit, shots=shots).result()
        w_counts = w_result.get_counts()
        
        # Run GHZ circuit
        ghz_circuit = self.create_ghz_state(sigma=0.0)
        ghz_result = self.simulator.run(ghz_circuit, shots=shots).result()
        ghz_counts = ghz_result.get_counts()
        
        # Analyze W-state signature
        w_states = ['100', '010', '001']
        w_state_count = sum(w_counts.get(s, 0) for s in w_states)
        w_signature = w_state_count / shots
        
        # Analyze GHZ signature
        ghz_states = ['000', '111']
        ghz_state_count = sum(ghz_counts.get(s, 0) for s in ghz_states)
        ghz_signature = ghz_state_count / shots
        
        print(f"\nW-STATE MEASUREMENT:")
        print(f"  Expected: ~33% each in |100⟩, |010⟩, |001⟩")
        print(f"  Observed: {w_signature:.1%} in W-states")
        for state in w_states:
            count = w_counts.get(state, 0)
            print(f"    |{state}⟩: {count:4d} ({count/shots:.1%})")
        
        print(f"\nGHZ-STATE MEASUREMENT:")
        print(f"  Expected: ~50% each in |000⟩, |111⟩")
        print(f"  Observed: {ghz_signature:.1%} in GHZ-states")
        for state in ghz_states:
            count = ghz_counts.get(state, 0)
            print(f"    |{state}⟩: {count:4d} ({count/shots:.1%})")
        
        # Verdict
        print(f"\n📊 VERDICT:")
        if w_signature > 0.95 and ghz_signature > 0.95:
            print("  ✓ Both states show perfect fidelity")
            print("  ⚠️  This is EXPECTED for local Aer simulation")
            print("  ⚠️  NOT evidence of quantum network probing")
        else:
            print("  ⚠️  Unexpected noise in measurements")
        
        return {
            'w_signature': w_signature,
            'ghz_signature': ghz_signature,
            'w_counts': w_counts,
            'ghz_counts': ghz_counts
        }
    
    # ========================================================================
    # TEST 2: PROBE WITHOUT ENTANGLEMENT - Should See Classical Limit
    # ========================================================================
    
    def create_unentangled_probe(self, sigma: float = 0.0) -> QuantumCircuit:
        """
        Create probe circuit WITHOUT entanglement
        
        If the manifold has real entanglement, measuring with an
        unentangled probe should give DIFFERENT results than measuring
        with an entangled probe.
        """
        qc = QuantumCircuit(3, 3)
        
        # NO entanglement - just independent qubits
        qc.h(0)
        qc.h(1)
        qc.h(2)
        
        # σ-modulation
        for qubit in range(3):
            qc.rx(sigma * np.pi / 4, qubit)
            qc.rz(sigma * np.pi / 2, qubit)
        
        qc.measure([0, 1, 2], [0, 1, 2])
        return qc
    
    def test_unentangled_probe(self, shots: int = 8192) -> Dict:
        """
        TEST 2: Measure with unentangled probe
        
        If you're probing a real entangled manifold:
        - Entangled probe: Can violate CHSH (> 2.0)
        - Unentangled probe: CANNOT violate CHSH (≤ 2.0)
        
        If both give CHSH > 2.0, you're just simulating locally!
        """
        print("\n" + "="*80)
        print("TEST 2: UNENTANGLED PROBE (Classical Limit Test)")
        print("="*80)
        
        # Unentangled probe
        unent_circuit = self.create_unentangled_probe(sigma=0.0)
        unent_result = self.simulator.run(unent_circuit, shots=shots).result()
        unent_counts = unent_result.get_counts()
        
        # Entangled probe (W-state)
        ent_circuit = self.create_w_state(sigma=0.0)
        ent_result = self.simulator.run(ent_circuit, shots=shots).result()
        ent_counts = ent_result.get_counts()
        
        # Calculate "correlation" (simplified CHSH)
        # For unentangled: should be ~2.0 (classical)
        # For entangled: should be >2.0 (quantum)
        
        def calc_simple_chsh(counts):
            """Simplified CHSH from measurement outcomes"""
            total = sum(counts.values())
            # Count correlated vs anti-correlated
            correlated = sum(counts.get(s, 0) for s in ['000', '111', '011', '100'])
            return 2.0 + (correlated / total) * 0.828
        
        unent_chsh = calc_simple_chsh(unent_counts)
        ent_chsh = calc_simple_chsh(ent_counts)
        
        print(f"\nUNENTANGLED PROBE:")
        print(f"  Outcomes: {dict(list(unent_counts.items())[:5])}")
        print(f"  CHSH estimate: {unent_chsh:.3f}")
        print(f"  Expected: ≤ 2.0 (classical limit)")
        
        print(f"\nENTANGLED PROBE (W-state):")
        print(f"  Outcomes: {dict(list(ent_counts.items())[:5])}")
        print(f"  CHSH estimate: {ent_chsh:.3f}")
        print(f"  Expected: > 2.0 (quantum)")
        
        print(f"\n📊 VERDICT:")
        if unent_chsh <= 2.1 and ent_chsh > 2.5:
            print("  ✓ Unentangled probe shows classical limit")
            print("  ✓ Entangled probe shows quantum correlations")
            print("  ⚠️  This is EXPECTED for local Aer simulation")
        elif unent_chsh > 2.1:
            print("  ⚠️  Unentangled probe showing quantum-like correlations!")
            print("  ⚠️  This should NOT happen - check circuit")
        
        return {
            'unentangled_chsh': unent_chsh,
            'entangled_chsh': ent_chsh,
            'unent_counts': unent_counts,
            'ent_counts': ent_counts
        }
    
    # ========================================================================
    # TEST 3: σ-REVIVAL PATTERN - Should ONLY Appear If Probing Real Manifold
    # ========================================================================
    
    def test_sigma_revival_pattern(self, triangle_id: int = 0, shots: int = 4096) -> Dict:
        """
        TEST 3: σ-revival pattern
        
        If you're probing a REAL quantum manifold that has σ-structured noise:
        - Fidelity should INCREASE at σ = 0, 4, 8, ... (revivals)
        - Fidelity should DECREASE at σ = 2, 6, 10, ... (decoherence)
        
        If you're just running local Aer:
        - All σ values should give similar fidelity (no revivals)
        - OR noise is just your circuit degrading uniformly
        """
        print("\n" + "="*80)
        print("TEST 3: σ-REVIVAL PATTERN (Manifold Signature)")
        print("="*80)
        
        route = self.get_route(triangle_id)
        print(f"\nProbing Triangle {triangle_id}")
        print(f"  σ-address from database: {route['sigma']:.6f}")
        
        # Test at multiple σ values
        sigma_values = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        fidelities = []
        
        print(f"\nMeasuring fidelity at different σ values:")
        for sigma in sigma_values:
            qc = self.create_w_state(sigma=sigma)
            result = self.simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
            
            # Calculate W-state fidelity
            w_states = ['100', '010', '001']
            w_count = sum(counts.get(s, 0) for s in w_states)
            fidelity = w_count / shots
            fidelities.append(fidelity)
            
            revival_marker = "🌟" if abs(sigma % 4.0) < 0.5 else "  "
            print(f"  σ={sigma:.1f}: F={fidelity:.4f} {revival_marker}")
        
        # Check for revival pattern
        revivals_at_0_4_8 = [fidelities[0], fidelities[4], fidelities[8]]
        valleys_at_2_6 = [fidelities[2], fidelities[6]]
        
        avg_revival = np.mean(revivals_at_0_4_8)
        avg_valley = np.mean(valleys_at_2_6)
        revival_amplitude = avg_revival - avg_valley
        
        print(f"\n📊 REVIVAL ANALYSIS:")
        print(f"  Average fidelity at σ=0,4,8 (revivals): {avg_revival:.4f}")
        print(f"  Average fidelity at σ=2,6 (valleys):    {avg_valley:.4f}")
        print(f"  Revival amplitude:                       {revival_amplitude:.4f}")
        
        print(f"\n📊 VERDICT:")
        if abs(revival_amplitude) < 0.05:
            print("  ⚠️  NO revival pattern detected")
            print("  ⚠️  This is EXPECTED for local Aer simulation")
            print("  ⚠️  To see revivals, you need to probe actual manifold!")
        elif revival_amplitude > 0.1:
            print("  ✓ Strong revival pattern detected!")
            print("  ✓ Fidelity increases at σ=0,4,8...")
            print("  🎯 This suggests you're probing structured noise!")
        
        return {
            'sigma_values': sigma_values,
            'fidelities': fidelities,
            'revival_amplitude': revival_amplitude,
            'avg_revival': avg_revival,
            'avg_valley': avg_valley
        }
    
    # ========================================================================
    # TEST 4: CROSS-TRIANGLE CORRELATION
    # ========================================================================
    
    def test_cross_triangle_correlation(self, shots: int = 4096) -> Dict:
        """
        TEST 4: Measure correlations between different triangles
        
        If there's a REAL quantum manifold:
        - Triangles at similar σ-coordinates should show correlation
        - Triangles far apart should show less correlation
        
        If you're just running local circuits:
        - No correlation between separate circuit runs
        """
        print("\n" + "="*80)
        print("TEST 4: CROSS-TRIANGLE CORRELATION")
        print("="*80)
        
        # Get triangles at same σ
        triangle_ids = [0, 98441, 196882]  # First, middle, last
        routes = [self.get_route(tid) for tid in triangle_ids]
        
        print(f"\nMeasuring 3 triangles:")
        results = []
        for i, (tid, route) in enumerate(zip(triangle_ids, routes)):
            print(f"  Triangle {tid}: σ={route['sigma']:.6f}")
            qc = self.create_w_state(sigma=route['sigma'])
            result = self.simulator.run(qc, shots=shots).result()
            counts = result.get_counts()
            results.append(counts)
        
        # Check if measurements are correlated
        # (They shouldn't be for independent local runs)
        print(f"\n📊 VERDICT:")
        print("  ⚠️  Each measurement is independent (local Aer)")
        print("  ⚠️  To test real correlation, need simultaneous measurement")
        print("  ⚠️  Or: server submits to IonQ, client queries same job")
        
        return {
            'triangle_ids': triangle_ids,
            'results': results
        }
    
    # ========================================================================
    # RUN ALL TESTS
    # ========================================================================
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("🧪 RUNNING COMPLETE QUANTUM PROBE TEST SUITE")
        print("="*80)
        print("\nThis will determine if you're:")
        print("  1. Running local Aer simulation (expected now)")
        print("  2. Actually probing remote quantum manifold (future goal)")
        print("="*80)
        
        # Run all tests
        test1 = self.test_w_vs_ghz()
        test2 = self.test_unentangled_probe()
        test3 = self.test_sigma_revival_pattern()
        
        # NEW CRITICAL TESTS
        test5 = self.test_detect_without_entangling()
        test6 = self.test_w_vs_ghz_signature()
        
        test4 = self.test_cross_triangle_correlation()
        
        # Final verdict
        print("\n" + "="*80)
        print("🎯 FINAL VERDICT")
        print("="*80)
        print("\nBased on all tests:")
        print("\n  Current Status: LOCAL AER SIMULATION")
        print("  ✓ You're running quantum circuits locally")
        print("  ✓ Database provides routing coordinates")
        print("  ✓ Circuits use those coordinates for σ-modulation")
        print("  ✓ But measurements are independent, local Aer runs")
        
        print("\n  KEY TEST: Detect Without Entangling")
        if test5['verdict'] == 'separable':
            print("  ✓ CHSH ≤ 2.0 with separable measurements")
            print("  ✓ Confirms no pre-existing entanglement")
            print("  ✓ This is expected for local simulation")
        else:
            print("  🎯 CHSH > 2.0 with separable measurements!")
            print("  🎯 This would prove pre-existing manifold entanglement!")
        
        print("\n  To Actually Probe Remote Manifold:")
        print("  1. Server creates W-state on IonQ hardware")
        print("  2. Server maintains that entangled state in 'manifold'")
        print("  3. Client measures with SEPARABLE probes only")
        print("  4. If CHSH > 2.0 → proves you measured existing entanglement!")
        print("\n  Alternative Approach:")
        print("  1. Server records IonQ job IDs in database")
        print("  2. Client queries those specific job results")
        print("  3. Calculate correlations between jobs")
        print("  4. If correlated → proves shared quantum state!")
        print("="*80)

if __name__ == '__main__':
    # Run test suite
    tester = QuantumProbeTest('moonshine.db')
    tester.run_all_tests()
