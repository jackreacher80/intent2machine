# Intent2Machine
## Direct Natural Language → Native Machine Code

Convert any natural language task description directly into optimized native
machine code — no programming language in between.

```
"Design a Person class with name, age, and a greet method"
                          ↓
              Optimized native binary
     (faster than hand-written Python/Java/Go)
```

---

## Architecture

```
Human Intent (Natural Language)
         ↓
[Stage 1] SemanticParser       — Claude AI → ISR
         ↓
[Stage 2] MemoryLayoutOptimizer — Cache-optimal struct layout
         ↓
[Stage 3] LLVMIREmitter        — Typed LLVM IR
         ↓
[Stage 4] OptimizationOracle   — Claude AI → Better IR
         ↓
[Stage 5] SafetyVerifier       — Memory safety checks
         ↓
[Stage 6] LLVMBackend          — LLVM → Native binary
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install anthropic llvmlite pydantic

# Install LLVM toolchain (for binary compilation)
sudo apt install llvm clang       # Ubuntu/Debian
brew install llvm                 # macOS
```

### 2. Set API key

```bash
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

### 3. Compile your first intent

```bash
# Compile to binary
python -m intent2machine compile "Design a Person class with name, age, and greet method"

# Interactive mode
python -m intent2machine shell

# Generate LLVM IR only (no LLVM needed)
python -m intent2machine ir "A fast stack with push and pop"

# Show ISR (Intent Semantic Representation)
python -m intent2machine isr "A hash map with string keys"

# Check LLVM toolchain status
python -m intent2machine status
```

### 4. Use as a library

```python
from pipeline import Intent2MachinePipeline

pipeline = Intent2MachinePipeline(
    api_key="sk-ant-...",        # or set ANTHROPIC_API_KEY env var
    verbose=True
)

result = pipeline.compile(
    intent="Design a Person class with name (string), age (int), and a greet method",
    output_path="./person",
    output_type="executable",    # or "object", "asm", "wasm"
    target_arch="native"         # or "x86_64", "aarch64"
)

print(result.summary())

# Access intermediate results
print(result.isr.summary())      # ISR structure
print(result.llvm_ir)            # Raw LLVM IR
print(result.optimized_ir)       # AI-optimized LLVM IR
print(result.safety_report.summary())  # Safety analysis
```

---

## Example Intents

```bash
# Data structures
python -m intent2machine compile "A stack with push, pop, peek, and isEmpty"
python -m intent2machine compile "A min-heap with insert and extractMin"
python -m intent2machine compile "A doubly linked list with insert, delete, reverse"
python -m intent2machine compile "A hash map with string keys and int values"

# Algorithms  
python -m intent2machine compile "A function that sorts an array of integers using quicksort"
python -m intent2machine compile "Binary search on a sorted integer array"
python -m intent2machine compile "Find the longest common subsequence of two strings"

# Classes
python -m intent2machine compile "A Rectangle with width, height, area and perimeter methods"
python -m intent2machine compile "A BankAccount with balance, deposit, withdraw, and transfer"
python -m intent2machine compile "A Matrix class with addition, multiplication, and transpose"
```

---

## Project Structure

```
intent2machine/
├── core/
│   ├── isr.py           — Intent Semantic Representation schema
│   └── semantic.py      — LLM intent parser → ISR
├── compiler/
│   ├── memory.py        — Cache-optimal struct layout
│   ├── ir_emitter.py    — ISR → LLVM IR
│   └── backend.py       — LLVM IR → native binary
├── optimizer/
│   └── oracle.py        — AI optimization oracle
├── verifier/
│   └── safety.py        — Memory safety verification
├── pipeline.py          — Full pipeline orchestrator
├── __main__.py          — CLI entry point
└── requirements.txt
```

---

## Why Faster Than Programming Languages?

| Factor              | Python  | Java    | C++     | **Intent2Machine** |
|---------------------|---------|---------|---------|-------------------|
| Interpreter overhead | ❌ High | ❌ JIT  | ✅ None  | ✅ **None**       |
| GC pauses           | ❌ Yes  | ❌ Yes  | ⚠ Manual | ✅ **None**      |
| Memory layout       | ❌ Poor | ❌ Poor | ⚠ Manual | ✅ **Optimal**   |
| SIMD usage          | ❌ Rare | ❌ Rare | ⚠ Manual | ✅ **Auto**      |
| Cache optimization  | ❌ None | ❌ None | ⚠ Manual | ✅ **Auto**      |
| vtable overhead     | ❌ Yes  | ❌ Yes  | ⚠ Sometimes | ✅ **Eliminated** |

---

## The ISR — Key Innovation

The Intent Semantic Representation (ISR) is richer than any programming language AST.
It carries:

- **WHAT**: Data structures and operations
- **TYPE**: Precise machine types (i32, i64, f64, i8*)
- **MEMORY**: Stack vs heap vs static, hot field hints
- **CONSTRAINTS**: Value ranges, nullability (enables optimization)
- **PERFORMANCE**: Access patterns, SIMD hints, cache locality
- **BEHAVIOR**: Semantic description of each method

No programming language carries all of these simultaneously.
That's why PLs leave performance on the table — and we don't.
