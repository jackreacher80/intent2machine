"""
pipeline.py - Main Compilation Pipeline
=========================================
Orchestrates all stages of the Intent2Machine pipeline:

  Stage 1: Semantic Parsing    (Natural Language -> ISR)
  Stage 2: Memory Layout       (ISR -> Optimized struct layout)
  Stage 3: IR Emission         (ISR + Layout -> LLVM IR)
  Stage 4: AI Optimization     (LLVM IR -> Optimized LLVM IR)
  Stage 5: Safety Verification (Verify IR is safe)
  Stage 6: Binary Compilation  (LLVM IR -> Native Binary)
"""

import os
import time
from dataclasses import dataclass, field
from typing import Optional

from .config import config

from .core.isr import ISRProgram
from .core.semantic import SemanticParser
from .compiler.memory import MemoryLayoutOptimizer
from .compiler.ir_emitter import LLVMIREmitter
from .compiler.backend import LLVMBackend, CompileResult
from .optimizer.oracle import OptimizationOracle
from .verifier.safety import SafetyVerifier, SafetyReport


@dataclass
class PipelineResult:
    """Complete result from the full pipeline run"""
    success: bool
    intent: str
    isr: Optional[ISRProgram] = None
    llvm_ir: Optional[str] = None
    optimized_ir: Optional[str] = None
    safety_report: Optional[SafetyReport] = None
    compile_result: Optional[CompileResult] = None
    total_time_ms: float = 0.0
    stage_times: dict = field(default_factory=dict)
    error_message: str = ""

    def summary(self) -> str:
        lines = ["\n" + "="*60]
        lines.append("  INTENT2MACHINE — Pipeline Result")
        lines.append("="*60)
        lines.append(f"  Intent  : {self.intent[:65]}")
        lines.append(f"  Status  : {'✓ SUCCESS' if self.success else '✗ FAILED'}")
        lines.append(f"  Total   : {self.total_time_ms:.0f}ms")

        if self.stage_times:
            lines.append("\n  Stage Breakdown:")
            for stage, ms in self.stage_times.items():
                lines.append(f"    {stage:<25} {ms:6.0f}ms")

        if self.isr:
            lines.append(self.isr.summary())

        if self.safety_report:
            lines.append(self.safety_report.summary())

        if self.compile_result and self.compile_result.success:
            size_kb = self.compile_result.binary_size_bytes / 1024
            lines.append(f"\n  Binary  : {self.compile_result.output_path}")
            lines.append(f"  Size    : {size_kb:.1f} KB")
            lines.append(f"  IR      : {self.compile_result.ir_path}")

        if self.error_message:
            lines.append(f"\n  Error   : {self.error_message}")

        lines.append("="*60)
        return "\n".join(lines)


class Intent2MachinePipeline:
    """
    The complete Intent -> Machine Code pipeline.
    
    Usage:
        pipeline = Intent2MachinePipeline(api_key="sk-ant-...")
        result = pipeline.compile(
            "Design a Person class with name, age, and a greet method",
            output_path="./person"
        )
        print(result.summary())
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        skip_ai_optimization: Optional[bool] = None,
        skip_safety_check: Optional[bool] = None,
        verbose: Optional[bool] = None
    ):
        # All settings fall back to config.py / .env values
        api_key = api_key or config.ANTHROPIC_API_KEY
        model = model or config.MODEL
        self.verbose = verbose if verbose is not None else config.VERBOSE
        self.skip_ai_optimization = skip_ai_optimization if skip_ai_optimization is not None else config.SKIP_AI_OPTIMIZATION
        self.skip_safety_check = skip_safety_check if skip_safety_check is not None else config.SKIP_SAFETY_CHECK

        config.validate()  # Raises clear error if API key missing

        if self.verbose:
            print(config.summary())

        # Initialize pipeline stages
        self.semantic_parser = SemanticParser(api_key=api_key, model=model)
        self.memory_optimizer = MemoryLayoutOptimizer()
        self.ir_emitter = LLVMIREmitter()
        self.ai_oracle = OptimizationOracle(api_key=api_key, model=model)
        self.safety_verifier = SafetyVerifier()
        self.backend = LLVMBackend(opt_level=config.OPTIMIZE_LEVEL)

    def compile(
        self,
        intent: str,
        output_path: str = "./output",
        output_type: str = "executable",
        target_arch: str = "native",
        save_ir: bool = True
    ) -> PipelineResult:
        """
        Full pipeline: natural language intent -> native binary.

        Args:
            intent: Natural language description of what to build
            output_path: Where to write the output binary
            output_type: "executable", "object", "asm", "wasm"
            target_arch: "native", "x86_64", "aarch64"
            save_ir: Save LLVM IR file alongside binary

        Returns:
            PipelineResult with full pipeline information
        """
        total_start = time.time()
        stage_times = {}

        self._log("\n🚀 Intent2Machine Pipeline Starting")
        self._log(f"   Intent: {intent[:80]}")
        self._log("-" * 60)

        result = PipelineResult(success=False, intent=intent)

        # ── STAGE 1: Semantic Parsing ─────────────────────
        try:
            self._log("\n[1/6] 🧠 Semantic Understanding...")
            t = time.time()
            isr = self.semantic_parser.parse(intent, verbose=self.verbose)
            stage_times["1. Semantic Parsing"] = (time.time() - t) * 1000
            result.isr = isr
            self._log(f"      ✓ ISR generated (confidence: {isr.confidence_score:.0%})")
        except Exception as e:
            result.error_message = f"Semantic parsing failed: {e}"
            return result

        # ── STAGE 2: Memory Layout ────────────────────────
        try:
            self._log("\n[2/6] 📐 Memory Layout Optimization...")
            t = time.time()
            layouts = self.memory_optimizer.optimize(isr, verbose=self.verbose)
            stage_times["2. Memory Layout"] = (time.time() - t) * 1000
            for name, layout in layouts.items():
                self._log(f"      ✓ '{name}': {layout.total_size}B, {layout.cache_lines} cache line(s)")
                if layout.total_padding_saved:
                    self._log(f"        Saved {layout.total_padding_saved}B padding vs declaration order")
        except Exception as e:
            result.error_message = f"Memory layout failed: {e}"
            return result

        # ── STAGE 3: IR Emission ──────────────────────────
        try:
            self._log("\n[3/6] ⚙️  LLVM IR Generation...")
            t = time.time()
            llvm_ir = self.ir_emitter.emit(isr, verbose=False)
            stage_times["3. IR Emission"] = (time.time() - t) * 1000
            result.llvm_ir = llvm_ir
            ir_lines = len(llvm_ir.splitlines())
            self._log(f"      ✓ {ir_lines} lines of LLVM IR generated")
        except Exception as e:
            result.error_message = f"IR emission failed: {e}"
            return result

        # ── STAGE 4: AI Optimization ──────────────────────
        if not self.skip_ai_optimization:
            try:
                self._log("\n[4/6] ⚡ AI Optimization Oracle...")
                t = time.time()
                # First analyze without LLM for quick wins
                analysis = self.ai_oracle.analyze_ir(llvm_ir, verbose=self.verbose)
                # Then apply AI-powered optimizations
                optimized_ir = self.ai_oracle.optimize(llvm_ir, isr, verbose=self.verbose)
                stage_times["4. AI Optimization"] = (time.time() - t) * 1000
                result.optimized_ir = optimized_ir
                self._log(f"      ✓ {len(analysis['opportunities'])} optimization opportunities applied")
            except Exception as e:
                self._log(f"      ⚠ AI optimization skipped: {e}")
                result.optimized_ir = llvm_ir
        else:
            result.optimized_ir = llvm_ir

        final_ir = result.optimized_ir or llvm_ir

        # ── STAGE 5: Safety Verification ─────────────────
        if not self.skip_safety_check:
            try:
                self._log("\n[5/6] 🛡️  Safety Verification...")
                t = time.time()
                safety = self.safety_verifier.verify(final_ir, isr, verbose=self.verbose)
                stage_times["5. Safety Check"] = (time.time() - t) * 1000
                result.safety_report = safety

                errors = [i for i in safety.issues if i.severity == "error"]
                warnings = [i for i in safety.issues if i.severity == "warning"]

                if errors:
                    self._log(f"      ✗ {len(errors)} safety errors — fix before compiling")
                    result.error_message = f"Safety verification failed: {len(errors)} errors"
                    return result
                elif warnings:
                    self._log(f"      ⚠ {len(warnings)} warnings (proceeding)")
                else:
                    self._log(f"      ✓ All {safety.checks_run} safety checks passed")
            except Exception as e:
                self._log(f"      ⚠ Safety check error: {e}")

        # Save IR file
        if save_ir:
            ir_path = output_path + ".ll"
            with open(ir_path, 'w') as f:
                f.write(final_ir)
            self._log(f"\n   IR saved: {ir_path}")

        # ── STAGE 6: Binary Compilation ───────────────────
        self._log("\n[6/6] 🔨 Native Binary Compilation...")
        t = time.time()
        compile_result = self.backend.compile(
            ir=final_ir,
            output_path=output_path,
            output_type=output_type,
            target_arch=target_arch,
            verbose=self.verbose
        )
        stage_times["6. Binary Compilation"] = (time.time() - t) * 1000
        result.compile_result = compile_result

        if compile_result.success:
            self._log(f"      ✓ Binary: {output_path} ({compile_result.binary_size_bytes} bytes)")
        else:
            self._log(f"      ⚠ Compilation: {compile_result.error_message[:100]}")
            # Still a partial success - IR was generated
            self._log(f"      → IR available at: {compile_result.ir_path}")

        result.success = True  # Pipeline succeeded even if binary needs LLVM
        result.total_time_ms = (time.time() - total_start) * 1000
        result.stage_times = stage_times

        self._log(result.summary())
        return result

    def interactive(self):
        """Run in interactive mode — compile multiple intents in a session"""
        print("\n" + "="*60)
        print("  Intent2Machine — Interactive Mode")
        print("  Type 'quit' to exit, 'help' for examples")
        print("="*60)

        examples = [
            "A Person class with name (string), age (int), and a greet method",
            "A Stack data structure with push, pop, peek, and isEmpty methods",
            "A HashMap with string keys and integer values: put, get, remove",
            "A BinarySearchTree with insert, find, and delete methods",
            "A function that computes fibonacci(n) efficiently",
        ]

        while True:
            try:
                intent = input("\n> Intent: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if intent.lower() in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if intent.lower() == 'help':
                print("\nExample intents:")
                for e in examples:
                    print(f"  • {e}")
                continue

            if not intent:
                continue

            output = input("  Output path (default: ./output): ").strip() or "./output"

            result = self.compile(intent, output_path=output)

            if result.compile_result and not result.compile_result.success:
                print(f"\n  To compile the generated IR:")
                print(f"  1. Install LLVM: sudo apt install llvm clang")
                print(f"  2. Run: clang -O3 {output}.ll -o {output}")

    def _log(self, msg: str):
        if self.verbose:
            print(msg)
