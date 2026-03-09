"""
semantic.py — LLM-Powered Intent → ISR Parser
==============================================
Uses Claude to parse natural language task descriptions into a
fully-typed, constraint-annotated ISR (Intent Semantic Representation).

This is the "front end" of the entire pipeline.
"""

import json
import os
import re
from typing import Optional
import anthropic

# Add parent dir to path so config is importable
from ..config import config

from .isr import (
    ISRProgram, ISRStruct, ISRFunction, ISRField, ISRMethod, ISRParameter,
    ISRConstraint, MemoryLayout, PerformanceContract,
    PrimType, PointerType, ArrayType, StringType, StructRef, ISRType,
    PrimitiveType, Mutability, MemoryRegion, Complexity,
    TargetArch, CallingConvention
)


# ─────────────────────────────────────────────
# SYSTEM PROMPT — The core intelligence
# ─────────────────────────────────────────────

SEMANTIC_SYSTEM_PROMPT = """
You are the Semantic Understanding Engine of a system that converts natural language
directly into native machine code. Your job is to parse a human task description into
a precise JSON structure called ISR (Intent Semantic Representation).

ISR is richer than any programming language AST. It captures:
- WHAT: The data structures and operations needed
- TYPE: Precise machine-level types (i32, i64, f64, i8*, etc.)
- MEMORY: Where data lives (stack/heap/static)
- CONSTRAINTS: Value ranges, nullability, invariants
- PERFORMANCE: Access patterns, hotness, SIMD opportunities
- BEHAVIOR: What each method does semantically

## Output Format

Return ONLY a valid JSON object. No markdown, no explanation.

```
{
  "original_intent": "<the input string>",
  "confidence": 0.95,
  "target_arch": "native",
  "optimize_level": 3,
  "structs": [
    {
      "name": "StructName",
      "doc": "brief description",
      "needs_constructor": true,
      "needs_destructor": false,
      "thread_safe": false,
      "memory_layout": {
        "strategy": "cache_optimized",
        "alignment": 8,
        "hot_fields_first": true
      },
      "perf_contract": {
        "latency_critical": false,
        "throughput_critical": false,
        "access_pattern": "random",
        "enable_simd": false,
        "enable_prefetch": false
      },
      "fields": [
        {
          "name": "field_name",
          "type": {"kind": "primitive", "value": "i32"},
          "mutability": "mutable",
          "memory_hint": "stack",
          "access_frequency": "hot",
          "constraints": [
            {"kind": "range", "min": 0, "max": 150}
          ],
          "doc": "description"
        },
        {
          "name": "name",
          "type": {"kind": "string", "max_len": null},
          "mutability": "mutable",
          "memory_hint": "heap",
          "access_frequency": "normal",
          "constraints": [{"kind": "non_null"}],
          "doc": "person's name"
        }
      ],
      "methods": [
        {
          "name": "method_name",
          "return_type": {"kind": "primitive", "value": "void"},
          "params": [],
          "complexity": "O(1)",
          "is_pure": false,
          "is_inline_hint": true,
          "side_effects": ["stdout"],
          "calling_conv": "ccc",
          "behavior_description": "prints a greeting using the name field",
          "doc": "greets the user"
        }
      ]
    }
  ],
  "functions": []
}
```

## Type Encoding Rules:
- Integer: {"kind": "primitive", "value": "i32"} (use i64 for large numbers, i8 for bytes)
- Float:   {"kind": "primitive", "value": "f64"}
- Bool:    {"kind": "primitive", "value": "bool"}
- Void:    {"kind": "primitive", "value": "void"}
- String:  {"kind": "string", "max_len": null}
- Array:   {"kind": "array", "element": {...}, "size": null}
- Pointer: {"kind": "pointer", "pointee": {...}, "nullable": false}
- Struct:  {"kind": "struct_ref", "name": "OtherStruct"}

## Memory Rules:
- Integers, floats, bools → "stack"
- Strings, dynamic arrays → "heap"
- Constants, lookup tables → "static"
- Frequently-accessed small fields → "register" hint

## Performance Rules:
- Fields accessed in every method call → access_frequency: "hot"
- Fields accessed rarely → access_frequency: "cold"
- Enable SIMD when struct may be used in arrays/batches
- Enable prefetch when access_pattern is "sequential"

## Constraint Rules:
- Numeric with known range → {"kind": "range", "min": X, "max": Y}
- Never-null pointers → {"kind": "non_null"}
- Non-empty strings → {"kind": "non_empty"}
- Power-of-2 values → {"kind": "power_of_2"}
- Positive numbers → {"kind": "positive"}

Always return valid JSON. Be precise with types. Prefer stack allocation.
"""


# ─────────────────────────────────────────────
# TYPE DESERIALIZER
# ─────────────────────────────────────────────

def deserialize_type(t: dict) -> ISRType:
    """Convert JSON type descriptor to ISRType object"""
    kind = t.get("kind")
    if kind == "primitive":
        val = t["value"].upper()
        # Normalize type names
        mapping = {
            "INT": "I32", "INTEGER": "I32", "LONG": "I64", "SHORT": "I16",
            "BYTE": "I8", "FLOAT": "F32", "DOUBLE": "F64",
            "BOOLEAN": "BOOL", "CHAR": "I8"
        }
        val = mapping.get(val, val)
        try:
            return PrimType(PrimitiveType[val])
        except KeyError:
            return PrimType(PrimitiveType.I32)  # fallback
    elif kind == "string":
        return StringType(max_len=t.get("max_len"))
    elif kind == "array":
        elem = deserialize_type(t["element"])
        return ArrayType(element=elem, size=t.get("size"))
    elif kind == "pointer":
        pointee = deserialize_type(t["pointee"])
        return PointerType(pointee=pointee, nullable=t.get("nullable", False))
    elif kind == "struct_ref":
        return StructRef(name=t["name"])
    else:
        return PrimType(PrimitiveType.VOID)


def deserialize_constraint(c: dict) -> ISRConstraint:
    kind = c.pop("kind", "unknown")
    return ISRConstraint(kind=kind, params=c)


def deserialize_field(f: dict) -> ISRField:
    return ISRField(
        name=f["name"],
        type=deserialize_type(f["type"]),
        mutability=Mutability(f.get("mutability", "mutable")),
        memory_hint=MemoryRegion(f.get("memory_hint", "stack")),
        constraints=[deserialize_constraint(dict(c)) for c in f.get("constraints", [])],
        default_value=f.get("default_value"),
        access_frequency=f.get("access_frequency", "normal"),
        doc=f.get("doc", "")
    )


def deserialize_param(p: dict) -> ISRParameter:
    return ISRParameter(
        name=p["name"],
        type=deserialize_type(p["type"]),
        mutability=Mutability(p.get("mutability", "immutable")),
        constraints=[deserialize_constraint(dict(c)) for c in p.get("constraints", [])]
    )


def deserialize_method(m: dict) -> ISRMethod:
    complexity_map = {
        "O(1)": Complexity.O1, "O(log n)": Complexity.OLOGN,
        "O(n)": Complexity.ON, "O(n log n)": Complexity.ONLOGN,
        "O(n^2)": Complexity.ON2
    }
    conv_map = {"ccc": CallingConvention.C, "fastcc": CallingConvention.FAST, "coldcc": CallingConvention.COLD}

    return ISRMethod(
        name=m["name"],
        params=[deserialize_param(p) for p in m.get("params", [])],
        return_type=deserialize_type(m["return_type"]),
        calling_conv=conv_map.get(m.get("calling_conv", "ccc"), CallingConvention.C),
        complexity=complexity_map.get(m.get("complexity", "O(1)"), Complexity.O1),
        side_effects=m.get("side_effects", []),
        is_pure=m.get("is_pure", False),
        is_inline_hint=m.get("is_inline_hint", False),
        behavior_description=m.get("behavior_description", ""),
        doc=m.get("doc", "")
    )


def deserialize_struct(s: dict) -> ISRStruct:
    ml = s.get("memory_layout", {})
    pc = s.get("perf_contract", {})
    return ISRStruct(
        name=s["name"],
        fields=[deserialize_field(f) for f in s.get("fields", [])],
        methods=[deserialize_method(m) for m in s.get("methods", [])],
        memory_layout=MemoryLayout(
            strategy=ml.get("strategy", "cache_optimized"),
            alignment=ml.get("alignment", 8),
            hot_fields_first=ml.get("hot_fields_first", True),
        ),
        perf_contract=PerformanceContract(
            latency_critical=pc.get("latency_critical", False),
            throughput_critical=pc.get("throughput_critical", False),
            access_pattern=pc.get("access_pattern", "random"),
            enable_simd=pc.get("enable_simd", False),
            enable_prefetch=pc.get("enable_prefetch", False),
        ),
        needs_constructor=s.get("needs_constructor", True),
        needs_destructor=s.get("needs_destructor", False),
        thread_safe=s.get("thread_safe", False),
        doc=s.get("doc", "")
    )


def json_to_isr(data: dict) -> ISRProgram:
    """Convert raw JSON dict (from LLM) to ISRProgram"""
    arch_map = {
        "x86_64": TargetArch.X86_64, "aarch64": TargetArch.ARM64,
        "wasm32": TargetArch.WASM32, "native": TargetArch.NATIVE
    }
    return ISRProgram(
        structs=[deserialize_struct(s) for s in data.get("structs", [])],
        functions=[],  # TODO: standalone functions
        target_arch=arch_map.get(data.get("target_arch", "native"), TargetArch.NATIVE),
        optimize_level=data.get("optimize_level", 3),
        original_intent=data.get("original_intent", ""),
        confidence_score=data.get("confidence", 1.0)
    )


# ─────────────────────────────────────────────
# MAIN SEMANTIC PARSER
# ─────────────────────────────────────────────

class SemanticParser:
    """
    Converts natural language intent into ISR using Claude.
    This is Stage 1 of the pipeline.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        # Priority: passed arg > config file > env var
        resolved_key = api_key or config.ANTHROPIC_API_KEY
        config.validate()  # Will raise clear error if key is missing
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model or config.MODEL

    def parse(self, intent: str, verbose: bool = False) -> ISRProgram:
        """
        Parse a natural language intent string into a full ISRProgram.

        Args:
            intent: Natural language description, e.g. "Design a Person class with name and age"
            verbose: Print intermediate steps

        Returns:
            ISRProgram ready for the compiler pipeline
        """
        if verbose:
            print(f"🧠 Parsing intent: '{intent}'")

        # Ask Claude to convert intent → ISR JSON
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SEMANTIC_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": f"Parse this intent into ISR JSON:\n\n{intent}"}
            ]
        )

        raw_text = response.content[0].text

        if verbose:
            print(f"📥 Raw LLM response:\n{raw_text[:500]}...")

        # Extract JSON (handle if LLM wraps in markdown)
        json_str = self._extract_json(raw_text)

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse ISR JSON from LLM response: {e}\n\nRaw:\n{raw_text}")

        isr = json_to_isr(data)
        isr.original_intent = intent  # Ensure original is preserved

        if verbose:
            print(isr.summary())

        return isr

    def _extract_json(self, text: str) -> str:
        """Extract JSON from LLM response, handling markdown code blocks"""
        # Try to find JSON block in markdown
        patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'(\{.*\})'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()

        # Return as-is if no wrapper found
        return text.strip()

    def refine(self, isr: ISRProgram, feedback: str) -> ISRProgram:
        """
        Refine an existing ISR based on user feedback.
        E.g., "Make the age field unsigned" or "Add a toString method"
        """
        current_json = isr.to_json()

        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=SEMANTIC_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content":
                    f"Here is an existing ISR:\n```json\n{current_json}\n```\n\n"
                    f"Modify it based on this feedback: {feedback}\n\n"
                    f"Return the complete updated ISR JSON."}
            ]
        )

        json_str = self._extract_json(response.content[0].text)
        data = json.loads(json_str)
        return json_to_isr(data)
