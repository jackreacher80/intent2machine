"""
backend.py - LLVM Backend Compiler
=====================================
Takes optimized LLVM IR and produces native machine code using the LLVM toolchain.

Pipeline:
  LLVM IR (.ll)
      -> opt (LLVM optimizer with -O3 passes)
      -> llc (LLVM static compiler -> .s assembly)
      -> clang/gcc (assembler + linker -> native binary)

Also supports:
- WebAssembly output (--target wasm32)
- Shared library output (.so / .dylib)
- Object file output for linking into larger programs
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class CompileResult:
    success: bool
    output_path: Optional[str]
    binary_size_bytes: int = 0
    compile_time_ms: float = 0.0
    error_message: str = ""
    ir_path: str = ""
    asm_path: str = ""
    stdout: str = ""
    stderr: str = ""


class LLVMBackend:
    """
    Compiles LLVM IR to native machine code.
    Requires LLVM toolchain installed (llc, opt, clang or gcc).
    Install: apt install llvm clang  OR  brew install llvm
    """

    def __init__(self, opt_level: int = 3, work_dir: Optional[str] = None):
        self.opt_level = opt_level
        self.work_dir = work_dir or tempfile.mkdtemp(prefix="i2m_")
        self._check_toolchain()

    def _check_toolchain(self):
        """Detect available LLVM tools"""
        self.llc   = shutil.which("llc")
        self.opt   = shutil.which("opt")
        self.clang = shutil.which("clang") or shutil.which("clang-15") or shutil.which("clang-14")
        self.lld   = shutil.which("ld.lld") or shutil.which("lld")

        self.has_llvm  = bool(self.llc)
        self.has_clang = bool(self.clang)

    def toolchain_status(self) -> str:
        lines = ["LLVM Toolchain Status:"]
        for tool, path in [("llc", self.llc), ("opt", self.opt),
                           ("clang", self.clang), ("lld", self.lld)]:
            status = f"✓ {path}" if path else "✗ NOT FOUND"
            lines.append(f"  {tool:<8}: {status}")
        if not self.has_llvm:
            lines.append("\n  Install: sudo apt install llvm clang")
            lines.append("  OR:      brew install llvm")
        return "\n".join(lines)

    def compile(
        self,
        ir: str,
        output_path: str,
        output_type: str = "executable",  # "executable", "object", "asm", "wasm"
        target_arch: str = "native",
        verbose: bool = False
    ) -> CompileResult:
        """
        Compile LLVM IR to native binary.

        Args:
            ir: LLVM IR string
            output_path: Where to write the final binary
            output_type: Type of output to produce
            target_arch: "native", "x86_64", "aarch64", "wasm32"
            verbose: Print compilation steps
        """
        import time
        start = time.time()

        if not self.has_llvm:
            # Fallback: save IR and provide instructions
            ir_path = output_path + ".ll"
            with open(ir_path, 'w') as f:
                f.write(ir)
            return CompileResult(
                success=False,
                output_path=None,
                ir_path=ir_path,
                error_message=(
                    f"LLVM toolchain not found. IR saved to: {ir_path}\n"
                    f"To compile manually:\n"
                    f"  opt -O3 {ir_path} -o {output_path}.opt.bc\n"
                    f"  llc -O3 {output_path}.opt.bc -o {output_path}.s\n"
                    f"  clang {output_path}.s -o {output_path}"
                )
            )

        # Step 1: Write IR to temp file
        ir_path = os.path.join(self.work_dir, "input.ll")
        with open(ir_path, 'w') as f:
            f.write(ir)

        if verbose:
            print(f"   IR written to: {ir_path}")

        # Step 2: Run LLVM optimizer (opt)
        opt_bc_path = os.path.join(self.work_dir, "optimized.bc")
        if self.opt:
            opt_result = self._run(
                [self.opt, f"-O{self.opt_level}",
                 "-vectorize-loops", "-loop-unroll",
                 ir_path, "-o", opt_bc_path],
                verbose=verbose, step="opt"
            )
            if opt_result.returncode != 0:
                # If opt fails, continue with unoptimized IR
                if verbose: print(f"   opt failed, using unoptimized IR")
                opt_bc_path = ir_path
        else:
            opt_bc_path = ir_path

        # Step 3: LLC — compile to assembly
        asm_path = os.path.join(self.work_dir, "output.s")
        llc_args = [
            self.llc,
            f"-O{self.opt_level}",
            "--filetype=asm",
        ]
        if target_arch != "native":
            llc_args.extend([f"--march={target_arch}"])
        if output_type == "wasm":
            llc_args.extend(["--mtriple=wasm32-unknown-unknown"])

        llc_args.extend([opt_bc_path, "-o", asm_path])

        llc_result = self._run(llc_args, verbose=verbose, step="llc")
        if llc_result.returncode != 0:
            return CompileResult(
                success=False, output_path=None, ir_path=ir_path,
                error_message=f"llc failed:\n{llc_result.stderr}"
            )

        if output_type == "asm":
            shutil.copy(asm_path, output_path)
            return CompileResult(success=True, output_path=output_path,
                                 ir_path=ir_path, asm_path=asm_path,
                                 compile_time_ms=(time.time()-start)*1000)

        # Step 4: Assemble and link
        if output_type == "object":
            link_args = [self.clang, "-c", asm_path, "-o", output_path]
        else:
            link_args = [
                self.clang,
                f"-O{self.opt_level}",
                "-fno-exceptions",    # No C++ exception overhead
                asm_path, "-o", output_path
            ]
            if output_type == "shared":
                link_args.insert(1, "-shared")

        link_result = self._run(link_args, verbose=verbose, step="clang")
        if link_result.returncode != 0:
            return CompileResult(
                success=False, output_path=None, ir_path=ir_path, asm_path=asm_path,
                error_message=f"linker failed:\n{link_result.stderr}"
            )

        size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        elapsed = (time.time() - start) * 1000

        if verbose:
            print(f"   ✓ Binary: {output_path} ({size} bytes, {elapsed:.0f}ms)")

        return CompileResult(
            success=True,
            output_path=output_path,
            binary_size_bytes=size,
            compile_time_ms=elapsed,
            ir_path=ir_path,
            asm_path=asm_path
        )

    def _run(self, args: list, verbose: bool, step: str) -> subprocess.CompletedProcess:
        if verbose:
            print(f"   [{step}] {' '.join(args[:4])}...")
        return subprocess.run(args, capture_output=True, text=True)
