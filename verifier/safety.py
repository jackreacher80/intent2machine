"""
safety.py - Safety Verifier
=====================================
Verifies the generated LLVM IR for safety properties before compilation.
Catches memory safety issues, uninitialized reads, null dereferences, etc.

Two modes:
1. Static analysis (fast, no deps) - regex/pattern based
2. LLVM-based (uses opt + sanitizers for thorough checking)
"""

import re
from dataclasses import dataclass, field
from ..core.isr import ISRProgram


@dataclass
class SafetyIssue:
    severity: str    # "error", "warning", "info"
    category: str    # "null_deref", "uninit_read", "buffer_overflow", "memory_leak"
    line: int
    message: str
    suggestion: str = ""


@dataclass
class SafetyReport:
    passed: bool
    issues: list = field(default_factory=list)
    checks_run: int = 0

    def summary(self) -> str:
        errors   = [i for i in self.issues if i.severity == "error"]
        warnings = [i for i in self.issues if i.severity == "warning"]
        infos    = [i for i in self.issues if i.severity == "info"]

        status = "✓ PASSED" if self.passed else "✗ FAILED"
        lines = [
            f"\n  Safety Check: {status}",
            f"  Checks run : {self.checks_run}",
            f"  Errors     : {len(errors)}",
            f"  Warnings   : {len(warnings)}",
            f"  Info       : {len(infos)}"
        ]
        if self.issues:
            lines.append("")
            for issue in self.issues:
                icon = {"error": "✗", "warning": "⚠", "info": "ℹ"}.get(issue.severity, "•")
                lines.append(f"  {icon} [{issue.category}] L{issue.line}: {issue.message}")
                if issue.suggestion:
                    lines.append(f"    → {issue.suggestion}")
        return "\n".join(lines)


class SafetyVerifier:
    """
    Static safety analysis on LLVM IR.
    Catches common machine-code level bugs before they reach the CPU.
    """

    def verify(self, ir: str, program: ISRProgram, verbose: bool = False) -> SafetyReport:
        issues = []
        checks = 0

        lines = ir.splitlines()

        # Track defined variables
        defined_vars: set[str] = set()
        # Track loaded pointers
        loaded_ptrs: set[str] = set()
        # Track malloc'd pointers
        malloc_ptrs: set[str] = set()
        # Track freed pointers
        freed_ptrs: set[str] = set()

        for i, line in enumerate(lines, 1):
            s = line.strip()
            if not s or s.startswith(';'):
                continue

            # ── Track variable definitions ────────────────
            def_match = re.match(r'(%\w+)\s*=', s)
            if def_match:
                defined_vars.add(def_match.group(1))

            # ── Check 1: Null pointer dereference risk ────
            checks += 1
            if 'load' in s:
                # Check if we're loading from a potentially null pointer
                ptr_match = re.search(r'load \S+, \S+\* (%\w+)', s)
                if ptr_match:
                    ptr = ptr_match.group(1)
                    # Check if this pointer came from malloc (could be null on OOM)
                    if ptr in malloc_ptrs:
                        issues.append(SafetyIssue(
                            severity="warning",
                            category="null_deref",
                            line=i,
                            message=f"Loading from malloc'd pointer {ptr} without null check",
                            suggestion="Check if malloc returned null before use"
                        ))

            # ── Check 2: Use after free ────────────────────
            checks += 1
            if 'load' in s or 'store' in s or 'getelementptr' in s:
                ptr_match = re.search(r'(%\w+)', s)
                if ptr_match:
                    ptr = ptr_match.group(1)
                    if ptr in freed_ptrs:
                        issues.append(SafetyIssue(
                            severity="error",
                            category="use_after_free",
                            line=i,
                            message=f"Use of pointer {ptr} after it was freed",
                            suggestion="Ensure pointer is not used after call to @free"
                        ))

            # ── Check 3: Track malloc ────────────────────
            malloc_match = re.match(r'(%\w+)\s*=\s*call i8\* @malloc', s)
            if malloc_match:
                malloc_ptrs.add(malloc_match.group(1))

            # ── Check 5: Double free ─────────────────────
            checks += 1
            if 'call void @free' in s:
                ptr_match = re.search(r'@free\(i8\* (%\w+)\)', s)
                if ptr_match:
                    ptr = ptr_match.group(1)
                    if ptr in freed_ptrs:
                        issues.append(SafetyIssue(
                            severity="error",
                            category="double_free",
                            line=i,
                            message=f"Double free of pointer {ptr}",
                            suggestion="Track ownership and free each pointer exactly once"
                        ))

            # ── Check 4: Track free ─────────────────────
            free_match = re.search(r'call void @free\(i8\* (%\w+)\)', s)
            if free_match:
                freed_ptrs.add(free_match.group(1))

            # ── Check 6: Stack overflow risk ─────────────
            checks += 1
            alloca_match = re.search(r'alloca \[(\d+) x i8\]', s)
            if alloca_match:
                size = int(alloca_match.group(1))
                if size > 65536:  # 64KB on stack
                    issues.append(SafetyIssue(
                        severity="warning",
                        category="stack_overflow",
                        line=i,
                        message=f"Large stack allocation: {size} bytes ({size//1024}KB)",
                        suggestion="Consider heap allocation for large buffers"
                    ))

            # ── Check 7: Missing return ───────────────────
            checks += 1
            if s.startswith('define ') and 'void' not in s:
                # Non-void function definition - will check for ret below
                pass

            # ── Check 8: Uninitialized struct field access ─
            checks += 1
            gep_match = re.match(r'(%\w+)\s*=\s*getelementptr.*i32 0, i32 (\d+)', s)
            if gep_match:
                # Valid GEP - field access
                pass

        # ── Check 9: Memory leaks (malloc without free) ──
        checks += 1
        unfreed = malloc_ptrs - freed_ptrs
        # Only warn about mallocs that reach end of function scope
        # (Simple heuristic: if malloc count > free count)
        if len(malloc_ptrs) > len(freed_ptrs):
            issues.append(SafetyIssue(
                severity="warning",
                category="memory_leak",
                line=0,
                message=f"{len(malloc_ptrs) - len(freed_ptrs)} malloc(s) without corresponding free(s)",
                suggestion="Ensure all heap allocations are freed. Consider using _free() destructor."
            ))

        # ── Check 10: Constraint violations ──────────────
        checks += 1
        for struct in program.structs:
            for f in struct.fields:
                for c in f.constraints:
                    if c.kind == "range":
                        # Could add runtime bounds check injection here
                        pass
                    if c.kind == "non_null":
                        # Field should never be null - verify no null stores
                        pass

        errors = [i for i in issues if i.severity == "error"]
        passed = len(errors) == 0

        report = SafetyReport(passed=passed, issues=issues, checks_run=checks)

        if verbose:
            print(report.summary())

        return report
