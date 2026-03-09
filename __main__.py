"""
__main__.py - CLI Entry Point
==============================
Run with: python -m intent2machine <command> [options]

Commands:
  compile  "your intent here"     - Compile intent to binary
  shell                           - Interactive mode
  status                          - Check toolchain status
  ir       "your intent here"     - Generate IR only (no compilation)
"""

import sys
import os
import argparse


def main():
    parser = argparse.ArgumentParser(
        prog="intent2machine",
        description="Intent2Machine — Convert natural language directly to native machine code"
    )

    subparsers = parser.add_subparsers(dest="command")

    # compile command
    compile_parser = subparsers.add_parser("compile", help="Compile intent to binary")
    compile_parser.add_argument("intent", help="Natural language description")
    compile_parser.add_argument("-o", "--output", default="./output", help="Output path")
    compile_parser.add_argument("--type", choices=["executable","object","asm","wasm"],
                                 default="executable", help="Output type")
    compile_parser.add_argument("--arch", default="native", help="Target architecture")
    compile_parser.add_argument("--no-ai-opt", action="store_true",
                                 help="Skip AI optimization oracle (faster)")
    compile_parser.add_argument("--no-safety", action="store_true",
                                 help="Skip safety verification")
    compile_parser.add_argument("-q", "--quiet", action="store_true")

    # ir command
    ir_parser = subparsers.add_parser("ir", help="Generate LLVM IR only")
    ir_parser.add_argument("intent", help="Natural language description")
    ir_parser.add_argument("-o", "--output", default="./output.ll")
    ir_parser.add_argument("-q", "--quiet", action="store_true")

    # isr command
    isr_parser = subparsers.add_parser("isr", help="Show ISR (Intent Semantic Representation)")
    isr_parser.add_argument("intent", help="Natural language description")

    # shell command
    subparsers.add_parser("shell", help="Interactive mode")

    # status command
    subparsers.add_parser("status", help="Check LLVM toolchain status")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Check API key via config (handles .env file too)
    if args.command in ("compile", "ir", "isr", "shell"):
        from .config import config
        try:
            config.validate()
        except EnvironmentError as e:
            print(e)
            sys.exit(1)

    if args.command == "status":
        from .compiler.backend import LLVMBackend
        backend = LLVMBackend()
        print(backend.toolchain_status())

    elif args.command == "isr":
        from .core.semantic import SemanticParser
        parser_obj = SemanticParser()
        isr = parser_obj.parse(args.intent, verbose=True)
        print("\n--- ISR JSON ---")
        print(isr.to_json())

    elif args.command == "ir":
        from .core.semantic import SemanticParser
        from .compiler.ir_emitter import LLVMIREmitter
        from .optimizer.oracle import OptimizationOracle

        print(f"Generating IR for: {args.intent}")
        sem = SemanticParser()
        isr = sem.parse(args.intent, verbose=not args.quiet)

        emitter = LLVMIREmitter()
        ir = emitter.emit(isr)

        oracle = OptimizationOracle()
        ir = oracle.optimize(ir, isr, verbose=not args.quiet)

        output = args.output
        if not output.endswith('.ll'):
            output += '.ll'
        with open(output, 'w') as f:
            f.write(ir)
        print(f"\n✓ IR saved to: {output}")
        print(f"\nTo compile:")
        print(f"  clang -O3 {output} -o ./output")

    elif args.command == "compile":
        from .pipeline import Intent2MachinePipeline
        pipeline = Intent2MachinePipeline(
            skip_ai_optimization=args.no_ai_opt,
            skip_safety_check=args.no_safety,
            verbose=not args.quiet
        )
        result = pipeline.compile(
            intent=args.intent,
            output_path=args.output,
            output_type=args.type,
            target_arch=args.arch
        )
        sys.exit(0 if result.success else 1)

    elif args.command == "shell":
        from .pipeline import Intent2MachinePipeline
        pipeline = Intent2MachinePipeline()
        pipeline.interactive()


if __name__ == "__main__":
    main()
