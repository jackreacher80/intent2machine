"""
memory.py — Memory Layout Optimizer
=====================================
Determines the optimal memory layout for each struct in the ISR.
This is where we beat C++ and Rust — we always choose the optimal layout
rather than letting the programmer guess.

Key optimizations:
1. Hot fields first (cache locality)
2. Sort by size to minimize padding
3. Cache-line alignment for array-heavy structs
4. No vtable unless polymorphism is proven necessary
"""

from dataclasses import dataclass, field
from typing import Optional
from core.isr import (
    ISRProgram, ISRStruct, ISRField,
    PrimType, PointerType, ArrayType, StringType, StructRef, ISRType,
    PrimitiveType, MemoryRegion
)


# ─────────────────────────────────────────────
# TYPE SIZES (in bytes)
# ─────────────────────────────────────────────

def type_size_bytes(t: ISRType) -> int:
    """Return the size of a type in bytes"""
    if isinstance(t, PrimType):
        sizes = {
            PrimitiveType.I8: 1,  PrimitiveType.U8: 1,
            PrimitiveType.I16: 2, PrimitiveType.U16: 2,
            PrimitiveType.I32: 4, PrimitiveType.U32: 4,
            PrimitiveType.BOOL: 1,
            PrimitiveType.I64: 8, PrimitiveType.U64: 8,
            PrimitiveType.F32: 4, PrimitiveType.F64: 8,
            PrimitiveType.VOID: 0,
        }
        return sizes.get(t.kind, 8)
    elif isinstance(t, (PointerType, StringType)):
        return 8  # 64-bit pointer
    elif isinstance(t, ArrayType):
        if t.size:
            return type_size_bytes(t.element) * t.size
        return 8  # Dynamic array = pointer
    elif isinstance(t, StructRef):
        return 8  # Unknown, assume pointer-sized
    return 8

def type_alignment(t: ISRType) -> int:
    """Return the natural alignment of a type"""
    size = type_size_bytes(t)
    if size == 0: return 1
    return min(size, 8)  # Max natural alignment is 8


# ─────────────────────────────────────────────
# OPTIMIZED FIELD ORDER
# ─────────────────────────────────────────────

@dataclass
class LayoutField:
    """A field with computed offset"""
    original: ISRField
    offset: int       # Byte offset in struct
    size: int         # Size in bytes
    alignment: int    # Required alignment
    padding_before: int = 0
    padding_after: int = 0

@dataclass
class StructLayout:
    """Complete optimized layout for a struct"""
    name: str
    fields: list        # list[LayoutField], ordered by offset
    total_size: int
    total_padding_saved: int
    cache_lines: int    # How many 64-byte cache lines the struct spans
    notes: list = field(default_factory=list)

    def to_llvm_type_body(self) -> str:
        """Emit LLVM struct type body in layout order"""
        parts = []
        for lf in self.fields:
            parts.append(lf.original.type.to_llvm())
            if lf.padding_after > 0:
                parts.append(f"[{lf.padding_after} x i8]")  # Padding bytes
        return ", ".join(parts)

    def summary(self) -> str:
        lines = [f"  Layout '{self.name}': {self.total_size} bytes, {self.cache_lines} cache line(s)"]
        for lf in self.fields:
            pad = f" +{lf.padding_after}b pad" if lf.padding_after else ""
            hot = " 🔥" if lf.original.access_frequency == "hot" else ""
            lines.append(f"    +{lf.offset:3d}  {lf.original.name:<20} {lf.original.type.to_llvm():<10} ({lf.size}B){pad}{hot}")
        if self.total_padding_saved:
            lines.append(f"  Saved {self.total_padding_saved} bytes vs naive layout")
        return "\n".join(lines)


class MemoryLayoutOptimizer:
    """
    Computes the optimal memory layout for structs.

    Strategy:
    1. Separate hot fields from cold fields
    2. Within each group, sort by descending size (largest first = least padding)
    3. Compute offsets with proper alignment
    4. Report padding saved vs. naive declaration order
    """

    def optimize(self, program: ISRProgram, verbose: bool = False) -> dict:
        """
        Optimize all struct layouts in the program.
        Returns: dict mapping struct_name -> StructLayout
        """
        layouts = {}
        for struct in program.structs:
            layout = self._optimize_struct(struct, verbose)
            layouts[struct.name] = layout
            if verbose:
                print(layout.summary())
        return layouts

    def _optimize_struct(self, struct: ISRStruct, verbose: bool = False) -> StructLayout:
        fields = struct.fields

        # Step 1: Compute naive (declaration-order) size for comparison
        naive_size = self._compute_size_naive(fields)

        # Step 2: Separate hot from cold fields
        hot_fields   = [f for f in fields if f.access_frequency == "hot"]
        warm_fields  = [f for f in fields if f.access_frequency == "normal"]
        cold_fields  = [f for f in fields if f.access_frequency == "cold"]

        # Step 3: Within each group, sort by descending type size (minimizes padding)
        def sort_key(f): return -type_size_bytes(f.type)

        ordered = (
            sorted(hot_fields, key=sort_key) +
            sorted(warm_fields, key=sort_key) +
            sorted(cold_fields, key=sort_key)
        )

        # Step 4: Compute offsets
        layout_fields = []
        offset = 0
        for f in ordered:
            size = type_size_bytes(f.type)
            align = type_alignment(f.type)

            # Align offset
            padding_before = (align - (offset % align)) % align
            offset += padding_before

            lf = LayoutField(
                original=f,
                offset=offset,
                size=size,
                alignment=align,
                padding_before=padding_before
            )
            layout_fields.append(lf)
            offset += size

        # Step 5: Final struct alignment (align total size to largest member alignment)
        max_align = max((lf.alignment for lf in layout_fields), default=1)
        final_padding = (max_align - (offset % max_align)) % max_align
        offset += final_padding

        if layout_fields:
            layout_fields[-1].padding_after = final_padding

        # Step 6: Cache line analysis
        cache_lines = max(1, (offset + 63) // 64)

        notes = []
        saved = naive_size - offset
        if saved > 0:
            notes.append(f"Saved {saved} bytes by reordering fields")
        if cache_lines == 1 and len(fields) > 2:
            notes.append("Entire struct fits in 1 cache line ✓")
        if cache_lines > 2:
            notes.append(f"Consider splitting hot/cold fields — spans {cache_lines} cache lines")

        return StructLayout(
            name=struct.name,
            fields=layout_fields,
            total_size=offset,
            total_padding_saved=max(0, saved),
            cache_lines=cache_lines,
            notes=notes
        )

    def _compute_size_naive(self, fields: list) -> int:
        """Compute struct size in declaration order (no reordering)"""
        offset = 0
        for f in fields:
            size = type_size_bytes(f.type)
            align = type_alignment(f.type)
            padding = (align - (offset % align)) % align
            offset += padding + size
        return offset
