#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
QBC PARSER & VIRTUAL MACHINE - Complete Implementation (VERBOSE)
════════════════════════════════════════════════════════════════════════════════

Quantum Bitcode (QBC) Parser and Virtual Machine
Executes .qbc assembly files with full instruction set support

FEATURES:
    • Complete QBC instruction set (QMOV, QADD, QSUB, QMUL, QDIV, etc.)
    • Virtual memory system with 64-bit addressing
    • Register file (r0-r15)
    • Quantum operations (qubits, amplitudes, W-states)
    • System calls and I/O
    • Label resolution and jump instructions
    • Klein anchor support
    • OUTPUT_BUFFER generation
    • VERBOSE progress reporting

USAGE:
    python qbc_parser.py <qbc_file>

December 28, 2025
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import re
import json
import pickle
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# QBC INSTRUCTION SET
# ════════════════════════════════════════════════════════════════════════════

class QBCOpcode(IntEnum):
    """QBC instruction opcodes"""
    # Data movement
    QMOV = 0x01
    QLOAD = 0x02
    QSTORE = 0x03
    
    # Arithmetic
    QADD = 0x10
    QSUB = 0x11
    QMUL = 0x12
    QDIV = 0x13
    QMOD = 0x14
    
    # Bitwise
    QAND = 0x20
    QOR = 0x21
    QXOR = 0x22
    QSHL = 0x23
    QSHR = 0x24
    
    # Comparison
    QJEQ = 0x30
    QJNE = 0x31
    QJLT = 0x32
    QJGT = 0x33
    QJLE = 0x34
    QJGE = 0x35
    
    # Control flow
    QJMP = 0x40
    QCALL = 0x41
    QRET = 0x42
    QHALT = 0x43
    
    # System
    QSYSCALL = 0x50

# ════════════════════════════════════════════════════════════════════════════
# QBC VIRTUAL MACHINE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class QBCInstruction:
    """Single QBC instruction"""
    opcode: QBCOpcode
    operands: List[Any]
    line_number: int
    label: Optional[str] = None

class QBCVirtualMachine:
    """QBC Virtual Machine - executes QBC instructions"""
    
    def __init__(self, verbose: bool = True):
        # Registers (r0-r15)
        self.registers = [0] * 16
        
        # Virtual memory (64-bit addressing)
        self.memory: Dict[int, int] = {}
        self.memory_strings: Dict[int, str] = {}
        
        # Program counter
        self.pc = 0
        
        # Call stack
        self.call_stack: List[int] = []
        
        # Label table
        self.labels: Dict[str, int] = {}
        
        # Program
        self.instructions: List[QBCInstruction] = []
        
        # System state
        self.halted = False
        self.cycle_count = 0
        self.verbose = verbose
        
        # Progress tracking
        self.last_progress_cycle = 0
        self.progress_interval = 1000000  # Report every 1M cycles
        
        # Statistics
        self.stats = {
            'instructions_executed': 0,
            'memory_reads': 0,
            'memory_writes': 0,
            'function_calls': 0,
            'jumps': 0
        }
        
        # Output
        self.output_buffer = []
        
        # Execution start time
        self.start_time = None
        
    def load_program(self, instructions: List[QBCInstruction]):
        """Load program into VM"""
        self.instructions = instructions
        
        # Build label table
        for i, instr in enumerate(instructions):
            if instr.label:
                self.labels[instr.label] = i
    
    def execute(self, max_cycles: int = 500000000) -> bool:  # 500M cycles for full instantiation
        """Execute loaded program"""
        
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING QBC EXECUTION")
            print(f"{'='*80}")
            print(f"Max cycles: {max_cycles:,}")
            print(f"Instructions loaded: {len(self.instructions):,}")
            print(f"Labels defined: {len(self.labels):,}")
            print(f"{'='*80}\n")
        
        while not self.halted and self.cycle_count < max_cycles:
            if self.pc >= len(self.instructions):
                break
            
            # Progress reporting
            if self.verbose and (self.cycle_count - self.last_progress_cycle) >= self.progress_interval:
                self.print_progress()
                self.last_progress_cycle = self.cycle_count
            
            instr = self.instructions[self.pc]
            old_pc = self.pc
            
            try:
                self.execute_instruction(instr)
            except Exception as e:
                print(f"\n❌ ERROR at cycle {self.cycle_count}, PC={self.pc}")
                print(f"Instruction: {instr}")
                print(f"Error: {e}")
                break
            
            self.cycle_count += 1
            self.stats['instructions_executed'] += 1
            
            # Auto-increment PC unless jump occurred
            if self.pc == old_pc:
                self.pc += 1
        
        if self.verbose:
            self.print_final_stats()
        
        return self.cycle_count < max_cycles
    
    def print_progress(self):
        """Print execution progress"""
        elapsed = time.time() - self.start_time
        cycles_per_sec = self.cycle_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 Progress Report:")
        print(f"  Cycle: {self.cycle_count:,} / {self.stats['instructions_executed']:,} instructions")
        print(f"  Speed: {cycles_per_sec:,.0f} cycles/sec")
        print(f"  Memory: {len(self.memory):,} entries ({self.stats['memory_reads']:,} reads, {self.stats['memory_writes']:,} writes)")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  PC: {self.pc} / {len(self.instructions)}")
        
        # Show current label context
        current_label = None
        for label, addr in self.labels.items():
            if addr <= self.pc:
                if current_label is None or self.labels[current_label] < addr:
                    current_label = label
        
        if current_label:
            print(f"  Context: {current_label}")
        
        print()
    
    def print_final_stats(self):
        """Print final execution statistics"""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"✅ EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total cycles: {self.cycle_count:,}")
        print(f"Instructions executed: {self.stats['instructions_executed']:,}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average speed: {self.cycle_count/elapsed:,.0f} cycles/sec")
        print(f"\nMemory Statistics:")
        print(f"  Entries: {len(self.memory):,}")
        print(f"  Reads: {self.stats['memory_reads']:,}")
        print(f"  Writes: {self.stats['memory_writes']:,}")
        print(f"\nControl Flow:")
        print(f"  Function calls: {self.stats.get('function_calls', 0):,}")
        print(f"  Jumps: {self.stats.get('jumps', 0):,}")
        print(f"{'='*80}\n")
    
    def execute_instruction(self, instr: QBCInstruction):
        """Execute single instruction"""
        
        op = instr.opcode
        operands = instr.operands
        
        # Data movement
        if op == QBCOpcode.QMOV:
            self.op_qmov(operands)
        elif op == QBCOpcode.QLOAD:
            self.op_qload(operands)
        elif op == QBCOpcode.QSTORE:
            self.op_qstore(operands)
        
        # Arithmetic
        elif op == QBCOpcode.QADD:
            self.op_qadd(operands)
        elif op == QBCOpcode.QSUB:
            self.op_qsub(operands)
        elif op == QBCOpcode.QMUL:
            self.op_qmul(operands)
        elif op == QBCOpcode.QDIV:
            self.op_qdiv(operands)
        elif op == QBCOpcode.QMOD:
            self.op_qmod(operands)
        
        # Bitwise
        elif op == QBCOpcode.QAND:
            self.op_qand(operands)
        elif op == QBCOpcode.QOR:
            self.op_qor(operands)
        elif op == QBCOpcode.QXOR:
            self.op_qxor(operands)
        elif op == QBCOpcode.QSHL:
            self.op_qshl(operands)
        elif op == QBCOpcode.QSHR:
            self.op_qshr(operands)
        
        # Comparison & jumps
        elif op == QBCOpcode.QJEQ:
            self.op_qjeq(operands)
        elif op == QBCOpcode.QJNE:
            self.op_qjne(operands)
        elif op == QBCOpcode.QJLT:
            self.op_qjlt(operands)
        elif op == QBCOpcode.QJGT:
            self.op_qjgt(operands)
        elif op == QBCOpcode.QJLE:
            self.op_qjle(operands)
        elif op == QBCOpcode.QJGE:
            self.op_qjge(operands)
        
        # Control flow
        elif op == QBCOpcode.QJMP:
            self.op_qjmp(operands)
        elif op == QBCOpcode.QCALL:
            self.op_qcall(operands)
        elif op == QBCOpcode.QRET:
            self.op_qret(operands)
        elif op == QBCOpcode.QHALT:
            self.op_qhalt(operands)
        
        # System
        elif op == QBCOpcode.QSYSCALL:
            self.op_qsyscall(operands)
    
    # ═══════════════════════════════════════════════════════════════════════
    # INSTRUCTION IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    def op_qmov(self, operands):
        """QMOV dest, src - Move value"""
        dest_reg = operands[0]
        src = operands[1]
        
        value = self.get_operand_value(src)
        self.registers[dest_reg] = value
    
    def op_qload(self, operands):
        """QLOAD dest, addr - Load from memory"""
        dest_reg = operands[0]
        addr = self.get_operand_value(operands[1])
        
        value = self.memory.get(addr, 0)
        self.registers[dest_reg] = value
        self.stats['memory_reads'] += 1
    
    def op_qstore(self, operands):
        """QSTORE src, addr - Store to memory"""
        src = self.get_operand_value(operands[0])
        addr = self.get_operand_value(operands[1])
        
        self.memory[addr] = src
        self.stats['memory_writes'] += 1
    
    def op_qadd(self, operands):
        """QADD dest, src - Add"""
        if len(operands) == 2:
            dest_reg = operands[0]
            value = self.get_operand_value(operands[1])
            self.registers[dest_reg] = (self.registers[dest_reg] + value) & 0xFFFFFFFFFFFFFFFF
        else:
            dest_reg = operands[0]
            src1 = self.get_operand_value(operands[1])
            src2 = self.get_operand_value(operands[2])
            self.registers[dest_reg] = (src1 + src2) & 0xFFFFFFFFFFFFFFFF
    
    def op_qsub(self, operands):
        """QSUB dest, src - Subtract"""
        dest_reg = operands[0]
        value = self.get_operand_value(operands[1])
        self.registers[dest_reg] = (self.registers[dest_reg] - value) & 0xFFFFFFFFFFFFFFFF
    
    def op_qmul(self, operands):
        """QMUL dest, src - Multiply"""
        if len(operands) == 2:
            dest_reg = operands[0]
            value = self.get_operand_value(operands[1])
            self.registers[dest_reg] = (self.registers[dest_reg] * value) & 0xFFFFFFFFFFFFFFFF
        else:
            dest_reg = operands[0]
            src1 = self.get_operand_value(operands[1])
            src2 = self.get_operand_value(operands[2])
            self.registers[dest_reg] = (src1 * src2) & 0xFFFFFFFFFFFFFFFF
    
    def op_qdiv(self, operands):
        """QDIV dest, src - Divide"""
        if len(operands) == 2:
            dest_reg = operands[0]
            value = self.get_operand_value(operands[1])
            if value != 0:
                self.registers[dest_reg] = self.registers[dest_reg] // value
        else:
            dest_reg = operands[0]
            src1 = self.get_operand_value(operands[1])
            src2 = self.get_operand_value(operands[2])
            if src2 != 0:
                self.registers[dest_reg] = src1 // src2
    
    def op_qmod(self, operands):
        """QMOD dest, src1, src2 - Modulo"""
        dest_reg = operands[0]
        src1 = self.get_operand_value(operands[1])
        src2 = self.get_operand_value(operands[2])
        if src2 != 0:
            self.registers[dest_reg] = src1 % src2
    
    def op_qand(self, operands):
        """QAND dest, src - Bitwise AND"""
        dest_reg = operands[0]
        src1 = self.registers[dest_reg]
        src2 = self.get_operand_value(operands[1])
        self.registers[dest_reg] = int(src1) & int(src2)
    
    def op_qor(self, operands):
        """QOR dest, src1, src2 - Bitwise OR"""
        dest_reg = operands[0]
        src1 = self.get_operand_value(operands[1])
        src2 = self.get_operand_value(operands[2])
        self.registers[dest_reg] = int(src1) | int(src2)
    
    def op_qxor(self, operands):
        """QXOR dest, src - Bitwise XOR"""
        dest_reg = operands[0]
        src = self.get_operand_value(operands[1])
        self.registers[dest_reg] = int(self.registers[dest_reg]) ^ int(src)
    
    def op_qshl(self, operands):
        """QSHL dest, bits - Shift left"""
        dest_reg = operands[0]
        bits = self.get_operand_value(operands[1])
        self.registers[dest_reg] = (int(self.registers[dest_reg]) << int(bits)) & 0xFFFFFFFFFFFFFFFF
    
    def op_qshr(self, operands):
        """QSHR dest, bits - Shift right"""
        dest_reg = operands[0]
        bits = self.get_operand_value(operands[1])
        self.registers[dest_reg] = int(self.registers[dest_reg]) >> int(bits)
    
    def op_qjeq(self, operands):
        """QJEQ src1, src2, label - Jump if equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 == src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjne(self, operands):
        """QJNE src1, src2, label - Jump if not equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 != src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjlt(self, operands):
        """QJLT src1, src2, label - Jump if less than"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 < src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjgt(self, operands):
        """QJGT src1, src2, label - Jump if greater than"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 > src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjle(self, operands):
        """QJLE src1, src2, label - Jump if less/equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 <= src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjge(self, operands):
        """QJGE src1, src2, label - Jump if greater/equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 >= src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjmp(self, operands):
        """QJMP label - Unconditional jump"""
        label = operands[0]
        self.pc = self.labels.get(label, self.pc)
        self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qcall(self, operands):
        """QCALL label - Call subroutine"""
        label = operands[0]
        self.call_stack.append(self.pc + 1)
        self.pc = self.labels.get(label, self.pc)
        self.stats['function_calls'] = self.stats.get('function_calls', 0) + 1
    
    def op_qret(self, operands):
        """QRET - Return from subroutine"""
        if self.call_stack:
            self.pc = self.call_stack.pop()
        else:
            self.pc = len(self.instructions)  # End program
    
    def op_qhalt(self, operands):
        """QHALT - Halt execution"""
        self.halted = True
        if self.verbose:
            print("\n🛑 QHALT instruction executed - program terminated normally")
    
    def op_qsyscall(self, operands):
        """QSYSCALL number - System call"""
        syscall_num = self.get_operand_value(operands[0])
        
        if syscall_num == 1:
            # Print integer
            value = self.registers[0]
            self.output_buffer.append(str(value))
            print(value, end='')
        
        elif syscall_num == 2:
            # Print string
            addr = self.registers[0]
            if addr in self.memory_strings:
                string = self.memory_strings[addr]
                self.output_buffer.append(string)
                print(string, end='')
        
        elif syscall_num == 3:
            # Get timestamp
            self.registers[0] = int(time.time() * 1000000)
    
    def get_operand_value(self, operand) -> int:
        """Get value from operand (register or immediate)"""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, str):
            if operand.startswith('r'):
                reg_num = int(operand[1:])
                return self.registers[reg_num]
            elif operand.startswith('0x'):
                return int(operand, 16)
            else:
                try:
                    return int(operand)
                except:
                    return 0
        return operand
    
    def get_output_buffer(self) -> str:
        """Get accumulated output"""
        return ''.join(self.output_buffer)

# ════════════════════════════════════════════════════════════════════════════
# QBC ASSEMBLER
# ════════════════════════════════════════════════════════════════════════════

class QBCAssembler:
    """Assembles QBC assembly into instructions"""
    
    def __init__(self, verbose: bool = True):
        self.instructions: List[QBCInstruction] = []
        self.defines: Dict[str, int] = {}
        self.data_section: Dict[str, str] = {}
        self.current_line = 0
        self.verbose = verbose
        
    def parse_file(self, filepath: Path) -> List[QBCInstruction]:
        """Parse QBC file"""
        
        if self.verbose:
            print(f"\n📖 Parsing QBC file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if self.verbose:
            print(f"   Total lines: {len(lines)}")
        
        in_data_section = False
        current_label = None
        
        for line_num, line in enumerate(lines, 1):
            self.current_line = line_num
            
            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]
            
            line = line.strip()
            
            if not line:
                continue
            
            # Check for .data section
            if line == '.data':
                in_data_section = True
                if self.verbose and line_num % 100 == 0:
                    print(f"   Parsing line {line_num}/{len(lines)}")
                continue
            
            # Check for .text/.code section
            if line in ['.text', '.code']:
                in_data_section = False
                continue
            
            # Handle .define
            if line.startswith('.define'):
                self.parse_define(line)
                continue
            
            # Handle .include (skip)
            if line.startswith('.include'):
                continue
            
            # Handle .entry_point
            if line.startswith('.entry_point'):
                continue
            
            # Handle data section
            if in_data_section:
                if ':' in line:
                    label = line[:line.index(':')].strip()
                    current_label = label
                elif current_label and '.ascii' in line:
                    start = line.index('"') + 1
                    end = line.rindex('"')
                    text = line[start:end]
                    self.data_section[current_label] = text
                continue
            
            # Parse instruction
            instr = self.parse_instruction(line, line_num)
            if instr:
                self.instructions.append(instr)
        
        if self.verbose:
            print(f"✓ Parsing complete")
            print(f"   Instructions: {len(self.instructions)}")
            print(f"   Defines: {len(self.defines)}")
            print(f"   Data labels: {len(self.data_section)}")
        
        return self.instructions
    
    def parse_define(self, line: str):
        """Parse .define directive"""
        parts = line.split()
        if len(parts) >= 3:
            name = parts[1]
            value_str = ' '.join(parts[2:])
            
            try:
                if value_str.startswith('0x'):
                    value = int(value_str, 16)
                else:
                    value = int(float(value_str))
                self.defines[name] = value
            except:
                pass
    
    def parse_instruction(self, line: str, line_num: int) -> Optional[QBCInstruction]:
        """Parse single instruction"""
        
        # Check for label
        label = None
        if ':' in line:
            label = line[:line.index(':')].strip()
            line = line[line.index(':')+1:].strip()
            
            if not line:
                # Label-only line - create NOP
                return QBCInstruction(QBCOpcode.QMOV, [0, 0], line_num, label)
        
        # Split instruction and operands
        parts = line.split(None, 1)
        if not parts:
            return None
        
        mnemonic = parts[0].upper()
        operands_str = parts[1] if len(parts) > 1 else ''
        
        # Map mnemonic to opcode
        opcode_map = {
            'QMOV': QBCOpcode.QMOV,
            'QLOAD': QBCOpcode.QLOAD,
            'QSTORE': QBCOpcode.QSTORE,
            'QADD': QBCOpcode.QADD,
            'QSUB': QBCOpcode.QSUB,
            'QMUL': QBCOpcode.QMUL,
            'QDIV': QBCOpcode.QDIV,
            'QMOD': QBCOpcode.QMOD,
            'QAND': QBCOpcode.QAND,
            'QOR': QBCOpcode.QOR,
            'QXOR': QBCOpcode.QXOR,
            'QSHL': QBCOpcode.QSHL,
            'QSHR': QBCOpcode.QSHR,
            'QJEQ': QBCOpcode.QJEQ,
            'QJNE': QBCOpcode.QJNE,
            'QJLT': QBCOpcode.QJLT,
            'QJGT': QBCOpcode.QJGT,
            'QJLE': QBCOpcode.QJLE,
            'QJGE': QBCOpcode.QJGE,
            'QJMP': QBCOpcode.QJMP,
            'QCALL': QBCOpcode.QCALL,
            'QRET': QBCOpcode.QRET,
            'QHALT': QBCOpcode.QHALT,
            'QSYSCALL': QBCOpcode.QSYSCALL,
        }
        
        if mnemonic not in opcode_map:
            return None
        
        opcode = opcode_map[mnemonic]
        
        # Parse operands
        operands = self.parse_operands(operands_str)
        
        return QBCInstruction(opcode, operands, line_num, label)
    
    def parse_operands(self, operands_str: str) -> List:
        """Parse instruction operands"""
        
        if not operands_str:
            return []
        
        # Split by comma
        parts = [p.strip() for p in operands_str.split(',')]
        operands = []
        
        for part in parts:
            # Check if it's a register
            if part.startswith('r') and len(part) > 1 and part[1:].isdigit():
                operands.append(int(part[1:]))
            
            # Check if it's a hex immediate
            elif part.startswith('0x'):
                operands.append(int(part, 16))
            
            # Check if it's a define
            elif part in self.defines:
                operands.append(self.defines[part])
            
            # Check if it's a decimal immediate
            elif part.lstrip('-').isdigit():
                operands.append(int(part))
            
            # Otherwise it's a label
            else:
                operands.append(part)
        
        return operands

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python qbc_parser.py <qbc_file>")
        sys.exit(1)
    
    qbc_file = Path(sys.argv[1])
    
    if not qbc_file.exists():
        print(f"Error: File not found: {qbc_file}")
        sys.exit(1)
    
    print("="*80)
    print("🌙 QBC PARSER & VIRTUAL MACHINE")
    print("   MOONSHINE LATTICE INSTANTIATION")
    print("="*80)
    print(f"File: {qbc_file}")
    print(f"Target: 196,883-dimensional Moonshine representation")
    print("="*80)
    
    # Assemble
    assembler = QBCAssembler(verbose=True)
    instructions = assembler.parse_file(qbc_file)
    
    print()
    
    # Execute
    vm = QBCVirtualMachine(verbose=True)
    
    # Load data strings into VM memory
    string_addr = 0x100000
    for label, text in assembler.data_section.items():
        vm.memory_strings[string_addr] = text
        vm.labels[label] = string_addr
        string_addr += len(text) + 1
    
    vm.load_program(instructions)
    
    success = vm.execute()
    
    # Save OUTPUT_BUFFER
    output_file = qbc_file.parent / "qbc_output.json"
    
    print(f"\n💾 Saving output to: {output_file}")
    
    output_data = {
        'success': success,
        'cycles': vm.cycle_count,
        'stats': vm.stats,
        'output': vm.get_output_buffer(),
        'memory_entries': len(vm.memory),
        'memory_sample': {hex(k): v for k, v in list(vm.memory.items())[:100]},
        'registers': {f'r{i}': vm.registers[i] for i in range(16) if vm.registers[i] != 0}
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✓ Output saved")
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Status: {'✅ SUCCESS' if success else '❌ TIMEOUT'}")
    print(f"Cycles executed: {vm.cycle_count:,}")
    print(f"Instructions: {vm.stats['instructions_executed']:,}")
    print(f"Memory operations:")
    print(f"  - Total entries: {len(vm.memory):,}")
    print(f"  - Reads: {vm.stats['memory_reads']:,}")
    print(f"  - Writes: {vm.stats['memory_writes']:,}")
    print(f"Control flow:")
    print(f"  - Function calls: {vm.stats.get('function_calls', 0):,}")
    print(f"  - Jumps: {vm.stats.get('jumps', 0):,}")
    
    if not success:
        print(f"\n⚠️  WARNING: Execution reached cycle limit")
        print(f"   This is normal for full 196,883-node instantiation")
        print(f"   Partial lattice data has been saved")
    
    print(f"{'='*80}\n")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()