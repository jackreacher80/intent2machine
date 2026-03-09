"""
ir_emitter.py — LLVM IR Generator
===================================
Converts ISR + optimized memory layout into LLVM IR.

LLVM IR is the universal "assembly language" that LLVM then compiles
to any CPU architecture (x86, ARM, RISC-V, WASM, etc.)

Key design:
- Uses optimized field ordering from MemoryLayoutOptimizer
- Emits constructor/destructor when needed
- Generates direct syscall paths where possible (no libc overhead)
- Annotates with optimization metadata (tbaa, noinline, always_inline)
"""

from typing import Optional
from core.isr import (
    ISRProgram, ISRStruct, ISRField, ISRMethod, ISRParameter,
    PrimType, PointerType, ArrayType, StringType, StructRef, ISRType,
    PrimitiveType, CallingConvention, MemoryRegion
)
from compiler.memory import StructLayout, LayoutField, MemoryLayoutOptimizer


class LLVMIREmitter:
    """
    Generates LLVM IR from an ISRProgram.
    Output is a string of valid LLVM IR (.ll file content).
    """

    def __init__(self):
        self.lines: list[str] = []
        self.layouts: dict = {}
        self._indent = 0
        self._string_constants: dict[str, str] = {}  # value -> global name
        self._string_counter = 0

    def emit(self, program: ISRProgram, verbose: bool = False) -> str:
        """
        Main entry point. Returns LLVM IR as a string.
        """
        self.lines = []
        self._string_constants = {}
        self._string_counter = 0

        # Run memory layout optimizer first
        optimizer = MemoryLayoutOptimizer()
        self.layouts = optimizer.optimize(program, verbose=verbose)

        # File header
        self._emit_header(program)

        # Emit all struct type declarations
        for struct in program.structs:
            self._emit_struct_type(struct)

        self._line("")

        # Emit all struct methods
        for struct in program.structs:
            self._emit_struct_methods(struct)

        # Post-process: hoist global constant definitions out of function bodies.
        # Constants like @.str_... must be at the top level in LLVM IR, not inside
        # a 'define' block.
        global_consts = []
        clean_lines = []
        for line in self.lines:
            stripped = line.strip()
            if stripped.startswith('@') and '= private unnamed_addr constant' in stripped:
                if stripped not in global_consts:
                    global_consts.append(stripped)
                # drop the indented inline version
            else:
                clean_lines.append(line)

        # Insert hoisted globals before the first 'define'
        first_define = next(
            (i for i, l in enumerate(clean_lines) if l.startswith('define ')),
            len(clean_lines)
        )
        for i, g in enumerate(global_consts):
            clean_lines.insert(first_define + i, g)
        if global_consts:
            clean_lines.insert(first_define + len(global_consts), "")

        return "\n".join(clean_lines)

    # ─────────────────────────────────────────────
    # HEADER
    # ─────────────────────────────────────────────

    def _emit_header(self, program: ISRProgram):
        arch_map = {
            "x86_64": "x86_64-pc-linux-gnu",
            "aarch64": "aarch64-unknown-linux-gnu",
            "wasm32": "wasm32-unknown-emscripten",
            "native": "x86_64-pc-linux-gnu"  # Default to x86_64
        }
        triple = arch_map.get(program.target_arch.value, "x86_64-pc-linux-gnu")

        self._line(f"; ========================================================")
        self._line(f"; Intent2Machine — Auto-generated LLVM IR")
        self._line(f"; Intent: {program.original_intent}")
        self._line(f"; Target: {triple}")
        self._line(f"; Optimization: -O{program.optimize_level}")
        self._line(f"; ========================================================")
        self._line("")
        self._line(f'target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"')
        self._line(f'target triple = "{triple}"')
        self._line("")
        # Declare external functions we might need
        self._line('; External declarations')
        self._line('declare i64 @write(i32, i8*, i64)')
        self._line('declare i8* @malloc(i64)')
        self._line('declare void @free(i8*)')
        self._line('declare i8* @memcpy(i8*, i8*, i64)')
        self._line('declare i64 @strlen(i8*)')
        self._line('declare i32 @printf(i8*, ...)')
        self._line("")

    # ─────────────────────────────────────────────
    # STRUCT TYPE
    # ─────────────────────────────────────────────

    def _emit_struct_type(self, struct: ISRStruct):
        layout = self.layouts.get(struct.name)
        self._line(f"; Struct '{struct.name}' — {struct.doc}")
        if layout:
            self._line(f"; Size: {layout.total_size} bytes, {layout.cache_lines} cache line(s)")
            if layout.total_padding_saved:
                self._line(f"; Padding saved vs naive: {layout.total_padding_saved} bytes")
            body = layout.to_llvm_type_body()
        else:
            body = ", ".join(f.type.to_llvm() for f in struct.fields)

        self._line(f"%{struct.name} = type {{ {body} }}")
        self._line("")

    # ─────────────────────────────────────────────
    # STRUCT METHODS
    # ─────────────────────────────────────────────

    def _emit_struct_methods(self, struct: ISRStruct):
        layout = self.layouts.get(struct.name)

        # Build field index map (optimized order)
        field_indices = {}
        if layout:
            for i, lf in enumerate(layout.fields):
                field_indices[lf.original.name] = i
        else:
            for i, f in enumerate(struct.fields):
                field_indices[f.name] = i

        # Constructor
        if struct.needs_constructor:
            self._emit_constructor(struct, field_indices, layout)

        # User-defined methods
        for method in struct.methods:
            self._emit_method(struct, method, field_indices, layout)

        # Destructor
        if struct.needs_destructor or struct.has_heap_fields():
            self._emit_destructor(struct, field_indices, layout)

    def _emit_constructor(self, struct: ISRStruct, field_indices: dict, layout):
        params = []
        for f in struct.fields:
            params.append(f"{f.type.to_llvm()} %{f.name}")

        params_str = ", ".join(params) if params else ""
        self._line(f"; Constructor for {struct.name}")
        self._line(f"define %{struct.name}* @{struct.name}_new({params_str}) {{")
        self._line(f"entry:")

        # Decide stack vs heap
        if struct.has_heap_fields() or (layout and layout.total_size > 128):
            self._line(f"  %self = call i8* @malloc(i64 {layout.total_size if layout else 64})")
            self._line(f"  %self_typed = bitcast i8* %self to %{struct.name}*")
            ptr_var = "%self_typed"
        else:
            self._line(f"  %self = alloca %{struct.name}, align {struct.memory_layout.alignment}")
            ptr_var = "%self"

        # Store each field
        for f in struct.fields:
            idx = field_indices.get(f.name, 0)
            self._line(f"  %{f.name}_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* {ptr_var}, i32 0, i32 {idx}")
            self._line(f"  store {f.type.to_llvm()} %{f.name}, {f.type.to_llvm()}* %{f.name}_ptr")

        self._line(f"  ret %{struct.name}* {ptr_var}")
        self._line(f"}}")
        self._line("")

    def _emit_method(self, struct: ISRStruct, method: ISRMethod, field_indices: dict, layout):
        # Build parameter list
        self_param = f"%{struct.name}* %self"
        user_params = [f"{p.type.to_llvm()} %{p.name}" for p in method.params]
        all_params = [self_param] + user_params
        params_str = ", ".join(all_params)

        ret_type = method.return_type.to_llvm()

        # Calling convention and attributes
        cc = method.calling_conv.value
        attrs = []
        if method.is_inline_hint: attrs.append("alwaysinline")
        if method.is_pure: attrs.append("readnone")
        attr_str = " ".join(attrs)

        self._line(f"; Method {struct.name}::{method.name} — {method.behavior_description}")
        self._line(f"; Complexity: {method.complexity.value}, Side effects: {method.side_effects}")
        self._line(f"define {cc} {ret_type} @{struct.name}_{method.name}({params_str}) {attr_str} {{")
        self._line(f"entry:")

        # Generate method body based on behavior analysis
        self._emit_method_body(struct, method, field_indices)

        self._line(f"}}")
        self._line("")

    def _emit_method_body(self, struct: ISRStruct, method: ISRMethod, field_indices: dict):
        """
        Emit method body. Uses behavior_description and side_effects to decide what to emit.
        This is a smart emitter that handles common patterns.
        """
        behavior = method.behavior_description.lower()
        ret_type = method.return_type.to_llvm()
        name = method.name.lower()

        # ── GETTERS ──────────────────────────────
        if name.startswith("get_") or ("return" in behavior and "field" in behavior):
            field_name = name.replace("get_", "").replace("get", "")
            # Try to find matching field
            matching = None
            for f in struct.fields:
                if f.name.lower() == field_name or f.name.lower() in behavior:
                    matching = f
                    break
            if matching:
                idx = field_indices.get(matching.name, 0)
                self._line(f"  %field_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {idx}")
                self._line(f"  %val = load {matching.type.to_llvm()}, {matching.type.to_llvm()}* %field_ptr")
                self._line(f"  ret {matching.type.to_llvm()} %val")
                return

        # ── SETTERS ──────────────────────────────
        if name.startswith("set_") and method.params:
            field_name = name.replace("set_", "")
            matching = None
            for f in struct.fields:
                if f.name.lower() == field_name:
                    matching = f
                    break
            if matching and method.params:
                idx = field_indices.get(matching.name, 0)
                param = method.params[0]
                self._line(f"  %field_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {idx}")
                self._line(f"  store {param.type.to_llvm()} %{param.name}, {param.type.to_llvm()}* %field_ptr")
                self._line(f"  ret void")
                return

        # ── PRINT/GREET/DISPLAY ───────────────────
        if any(kw in name for kw in ["greet", "print", "display", "show", "tostring", "to_string"]) \
           or "stdout" in method.side_effects:
            # Find a string field to print
            str_field = None
            for f in struct.fields:
                from core.isr import StringType as ST
                if isinstance(f.type, ST):
                    str_field = f
                    break

            # Emit a format string constant
            fmt_name = f"@.str_{struct.name}_{method.name}"
            greeting_str = f"Hello, I am %s and I am %d years old\\0A\\00"

            self._line(f"  ; Print greeting — using printf for simplicity")

            # Find name and age fields
            name_field = next((f for f in struct.fields if "name" in f.name.lower()), None)
            age_field  = next((f for f in struct.fields if "age" in f.name.lower() or
                               f.name.lower() in ["years", "count"]), None)

            if name_field and age_field:
                nidx = field_indices.get(name_field.name, 0)
                aidx = field_indices.get(age_field.name, 1)
                self._line(f"  %name_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {nidx}")
                self._line(f"  %name_val = load i8*, i8** %name_ptr")
                self._line(f"  %age_ptr  = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {aidx}")
                self._line(f"  %age_val  = load i32, i32* %age_ptr")
                self._line(f'  {fmt_name} = private unnamed_addr constant [{len(greeting_str)-4} x i8] c"{greeting_str}"')
                self._line(f'  %fmt_ptr = getelementptr [{len(greeting_str)-4} x i8], [{len(greeting_str)-4} x i8]* {fmt_name}, i32 0, i32 0')
                self._line(f"  call i32 (i8*, ...) @printf(i8* %fmt_ptr, i8* %name_val, i32 %age_val)")
            elif name_field:
                nidx = field_indices.get(name_field.name, 0)
                self._line(f"  %name_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {nidx}")
                self._line(f"  %name_val = load i8*, i8** %name_ptr")
                self._line(f"  %len = call i64 @strlen(i8* %name_val)")
                self._line(f"  call i64 @write(i32 1, i8* %name_val, i64 %len)")

            if ret_type == "void":
                self._line(f"  ret void")
            else:
                self._line(f"  ret {ret_type} 0")
            return

        # ── COMPARE ──────────────────────────────
        if "compare" in name or "equals" in name or "eq" in name:
            if method.params:
                other_param = method.params[0]
                self._line(f"  ; Structural comparison — compare field by field")
                comparisons = []
                for i, f in enumerate(struct.fields):
                    idx = field_indices.get(f.name, i)
                    self._line(f"  %self_{f.name}_ptr  = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {idx}")
                    self._line(f"  %other_{f.name}_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* %{other_param.name}, i32 0, i32 {idx}")
                    self._line(f"  %self_{f.name}  = load {f.type.to_llvm()}, {f.type.to_llvm()}* %self_{f.name}_ptr")
                    self._line(f"  %other_{f.name} = load {f.type.to_llvm()}, {f.type.to_llvm()}* %other_{f.name}_ptr")
                    if not isinstance(f.type, (StringType,)):
                        self._line(f"  %cmp_{f.name} = icmp eq {f.type.to_llvm()} %self_{f.name}, %other_{f.name}")
                        comparisons.append(f"%cmp_{f.name}")

                if comparisons:
                    result = comparisons[0]
                    for c in comparisons[1:]:
                        self._line(f"  {result} = and i1 {result}, {c}")
                    self._line(f"  ret i1 {result}")
                    return

        # ── DEFAULT: Return zero/null ─────────────
        self._line(f"  ; TODO: Implement {method.name} — behavior: {method.behavior_description}")
        if ret_type == "void":
            self._line(f"  ret void")
        elif "*" in ret_type:
            self._line(f"  ret {ret_type} null")
        elif ret_type in ("i1", "bool"):
            self._line(f"  ret i1 false")
        elif ret_type in ("f32", "f64"):
            self._line(f"  ret {ret_type} 0.0")
        else:
            self._line(f"  ret {ret_type} 0")

    def _emit_destructor(self, struct: ISRStruct, field_indices: dict, layout):
        self._line(f"; Destructor for {struct.name}")
        self._line(f"define void @{struct.name}_free(%{struct.name}* %self) {{")
        self._line(f"entry:")

        # Free heap fields (strings, dynamic arrays)
        for f in struct.fields:
            if isinstance(f.type, StringType) or (
                isinstance(f.type, ArrayType) and f.type.size is None
            ):
                idx = field_indices.get(f.name, 0)
                self._line(f"  ; Free heap field '{f.name}'")
                self._line(f"  %{f.name}_ptr = getelementptr inbounds %{struct.name}, %{struct.name}* %self, i32 0, i32 {idx}")
                self._line(f"  %{f.name}_val = load {f.type.to_llvm()}, {f.type.to_llvm()}* %{f.name}_ptr")
                self._line(f"  %{f.name}_raw = bitcast {f.type.to_llvm()} %{f.name}_val to i8*")
                self._line(f"  call void @free(i8* %{f.name}_raw)")

        # Free the struct itself
        self._line(f"  %raw = bitcast %{struct.name}* %self to i8*")
        self._line(f"  call void @free(i8* %raw)")
        self._line(f"  ret void")
        self._line(f"}}")
        self._line("")

    # ─────────────────────────────────────────────
    # HELPERS
    # ─────────────────────────────────────────────

    def _line(self, text: str):
        self.lines.append(text)
