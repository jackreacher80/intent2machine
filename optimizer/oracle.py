"""
oracle.py - AI Optimization Oracle
Uses Claude to apply semantic-aware optimizations to LLVM IR.
Classical compilers optimize syntax. We optimize with full intent context.
"""
import os, re
from typing import Optional
import anthropic
from ..core.isr import ISRProgram

from ..config import config

ORACLE_SYSTEM_PROMPT = """You are an expert LLVM compiler optimization engineer.
Given LLVM IR and performance context, return an optimized version that:
- Applies escape analysis (heap malloc -> alloca where safe)
- Adds alwaysinline to small getters/setters (under 5 instructions)
- Adds SIMD/vectorization hints and loop metadata
- Annotates branches with probability metadata
- Removes redundant loads/stores
- Uses shl/shr instead of multiply/divide by power of 2
- Adds inbounds to safe GEP instructions

Return ONLY valid LLVM IR. No markdown, no explanation. Preserve all function signatures."""


class OptimizationOracle:
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        resolved_key = api_key or config.ANTHROPIC_API_KEY
        self.client = anthropic.Anthropic(api_key=resolved_key)
        self.model = model or config.MODEL

    def optimize(self, ir: str, program: ISRProgram, verbose: bool = False) -> str:
        """Apply AI-powered optimizations to LLVM IR with semantic context."""
        if verbose:
            print("   Running AI Optimization Oracle...")
        context = self._build_context(program)
        response = self.client.messages.create(
            model=self.model, max_tokens=8192,
            system=ORACLE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content":
                f"Intent: {program.original_intent}\n\n"
                f"Performance context:\n{context}\n\n"
                f"LLVM IR to optimize:\n{ir}"}]
        )
        optimized = response.content[0].text
        m = re.search(r'```(?:llvm)?\s*(.*?)\s*```', optimized, re.DOTALL)
        if m:
            optimized = m.group(1).strip()
        if verbose:
            print(f"   Lines: {len(ir.splitlines())} -> {len(optimized.splitlines())}")
        return optimized

    def _build_context(self, program: ISRProgram) -> str:
        lines = []
        for s in program.structs:
            pc = s.perf_contract
            lines.append(f"Struct '{s.name}':")
            if pc.latency_critical:    lines.append("  LATENCY CRITICAL - minimize instructions")
            if pc.throughput_critical: lines.append("  THROUGHPUT CRITICAL - vectorize loops")
            if pc.enable_simd:         lines.append("  SIMD enabled - find vectorization spots")
            if pc.enable_prefetch:     lines.append("  Sequential access - add prefetch hints")
            for f in s.fields:
                if f.access_frequency == "hot":
                    lines.append(f"  Field '{f.name}' is HOT - must stay in first cache line")
                for c in f.constraints:
                    if c.kind == "range":
                        lines.append(f"  Field '{f.name}' always in [{c.params.get('min')},{c.params.get('max')}] - omit bounds checks")
                    if c.kind == "non_null":
                        lines.append(f"  Field '{f.name}' is never null - skip null checks")
        return "\n".join(lines) or "No specific requirements"

    def analyze_ir(self, ir: str, verbose: bool = False) -> dict:
        """Analyze IR for optimization opportunities without calling LLM."""
        issues, opts = [], []
        for i, line in enumerate(ir.splitlines()):
            s = line.strip()
            if 'call i8* @malloc' in s:
                issues.append(f"L{i+1}: heap alloc - check if escape analysis can promote to stack")
            mul = re.search(r'mul (?:nsw |nuw )?i\d+ %\w+, (\d+)', s)
            if mul:
                n = int(mul.group(1))
                if n > 0 and (n & (n-1)) == 0:
                    opts.append(f"L{i+1}: mul {n} can be shl {n.bit_length()-1}")
            if 'getelementptr ' in s and 'inbounds' not in s:
                opts.append(f"L{i+1}: GEP missing 'inbounds' flag")
        if verbose:
            print(f"  {len(issues)} issues, {len(opts)} opportunities")
            for x in issues + opts: print(f"    {x}")
        return {"issues": issues, "opportunities": opts}
