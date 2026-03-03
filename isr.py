"""
ISR — Intent Semantic Representation
=====================================
The heart of the system. A structured, typed, constraint-rich representation
of human intent that carries MORE information than any programming language AST.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json


# ─────────────────────────────────────────────
# TYPE SYSTEM
# ─────────────────────────────────────────────

class PrimitiveType(Enum):
    I8   = "i8";  I16  = "i16";  I32  = "i32";  I64  = "i64"
    U8   = "u8";  U16  = "u16";  U32  = "u32";  U64  = "u64"
    F32  = "f32"; F64  = "f64";  BOOL = "bool";  VOID = "void"

class MemoryRegion(Enum):
    STACK = "stack"; HEAP = "heap"; STATIC = "static"; REGISTER = "register"

class Mutability(Enum):
    MUTABLE = "mutable"; IMMUTABLE = "immutable"; CONST = "const"

class Complexity(Enum):
    O1 = "O(1)"; OLOGN = "O(log n)"; ON = "O(n)"; ONLOGN = "O(n log n)"; ON2 = "O(n^2)"

class TargetArch(Enum):
    X86_64 = "x86_64"; ARM64 = "aarch64"; WASM32 = "wasm32"; NATIVE = "native"

class CallingConvention(Enum):
    C = "ccc"; FAST = "fastcc"; COLD = "coldcc"


# ─────────────────────────────────────────────
# TYPE NODES
# ─────────────────────────────────────────────

@dataclass
class ISRType:
    pass

@dataclass
class PrimType(ISRType):
    kind: PrimitiveType
    def to_llvm(self) -> str: return self.kind.value

@dataclass
class PointerType(ISRType):
    pointee: ISRType
    nullable: bool = False
    def to_llvm(self) -> str: return f"{self.pointee.to_llvm()}*"

@dataclass
class ArrayType(ISRType):
    element: ISRType
    size: Optional[int] = None
    def to_llvm(self) -> str:
        return f"[{self.size} x {self.element.to_llvm()}]" if self.size else f"{self.element.to_llvm()}*"

@dataclass
class StringType(ISRType):
    max_len: Optional[int] = None
    encoding: str = "utf8"
    def to_llvm(self) -> str: return "i8*"

@dataclass
class StructRef(ISRType):
    name: str
    def to_llvm(self) -> str: return f"%{self.name}"


# ─────────────────────────────────────────────
# FIELD, PARAM, METHOD
# ─────────────────────────────────────────────

@dataclass
class ISRConstraint:
    kind: str
    params: dict = field(default_factory=dict)
    def __repr__(self): return f"Constraint({self.kind}, {self.params})"

@dataclass
class ISRField:
    name: str
    type: ISRType
    mutability: Mutability = Mutability.MUTABLE
    memory_hint: MemoryRegion = MemoryRegion.STACK
    constraints: list = field(default_factory=list)
    default_value: Optional[str] = None
    access_frequency: str = "normal"  # "hot", "cold", "normal"
    doc: str = ""

@dataclass
class ISRParameter:
    name: str
    type: ISRType
    mutability: Mutability = Mutability.IMMUTABLE
    constraints: list = field(default_factory=list)

@dataclass
class ISRMethod:
    name: str
    params: list
    return_type: ISRType
    calling_conv: CallingConvention = CallingConvention.C
    complexity: Complexity = Complexity.O1
    side_effects: list = field(default_factory=list)
    is_pure: bool = False
    is_inline_hint: bool = False
    behavior_description: str = ""
    doc: str = ""


# ─────────────────────────────────────────────
# TOP-LEVEL ISR ENTITIES
# ─────────────────────────────────────────────

@dataclass
class MemoryLayout:
    strategy: str = "cache_optimized"  # "packed", "aligned", "cache_optimized"
    alignment: int = 8
    hot_fields_first: bool = True
    pad_to_cache_line: bool = False

@dataclass
class PerformanceContract:
    latency_critical: bool = False
    throughput_critical: bool = False
    memory_bound: bool = False
    expected_instance_count: Optional[int] = None
    access_pattern: str = "random"   # "sequential", "random"
    enable_simd: bool = True
    enable_prefetch: bool = False

@dataclass
class ISRStruct:
    name: str
    fields: list
    methods: list
    memory_layout: MemoryLayout = field(default_factory=MemoryLayout)
    perf_contract: PerformanceContract = field(default_factory=PerformanceContract)
    needs_constructor: bool = True
    needs_destructor: bool = False
    thread_safe: bool = False
    doc: str = ""

    def hot_fields(self): return [f for f in self.fields if f.access_frequency == "hot"]
    def has_heap_fields(self): return any(isinstance(f.type, (StringType, ArrayType)) for f in self.fields)

@dataclass
class ISRFunction:
    name: str
    method: ISRMethod

@dataclass
class ISRProgram:
    structs: list = field(default_factory=list)
    functions: list = field(default_factory=list)
    target_arch: TargetArch = TargetArch.NATIVE
    optimize_level: int = 3
    debug_info: bool = False
    original_intent: str = ""
    confidence_score: float = 1.0

    def to_json(self) -> str:
        def serialize(obj):
            if isinstance(obj, Enum): return obj.value
            if hasattr(obj, '__dataclass_fields__'): return {k: serialize(v) for k, v in obj.__dict__.items()}
            if isinstance(obj, list): return [serialize(i) for i in obj]
            if isinstance(obj, dict): return {k: serialize(v) for k, v in obj.items()}
            return obj
        return json.dumps(serialize(self), indent=2)

    def summary(self) -> str:
        lines = [f"\n╔══ ISR Program ══════════════════════════════",
                 f"║  Intent : {self.original_intent[:65]}",
                 f"║  Target : {self.target_arch.value}   Opt: -O{self.optimize_level}",
                 f"║  Entities: {len(self.structs)} structs, {len(self.functions)} functions"]
        for s in self.structs:
            lines.append(f"║\n║  Struct '{s.name}':")
            for f in s.fields:
                c = f" [{', '.join(str(c) for c in f.constraints)}]" if f.constraints else ""
                lines.append(f"║    field  {f.name:<15} : {f.type.to_llvm():<10} ({f.memory_hint.value}){c}")
            for m in s.methods:
                ps = ", ".join(f"{p.name}:{p.type.to_llvm()}" for p in m.params)
                lines.append(f"║    method {m.name}({ps}) -> {m.return_type.to_llvm()} [{m.complexity.value}]")
        lines.append("╚══════════════════════════════════════════════")
        return "\n".join(lines)
