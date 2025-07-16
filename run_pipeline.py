#!/usr/bin/env python3
"""Run the WVS analysis pipeline from Step 3 to completion."""
import subprocess
import sys
from pathlib import Path
import os


def run_command(command: str, description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}", file=sys.stderr)
        
        if result.returncode != 0:
            print(f"❌ Failed with return code: {result.returncode}")
            return False
        else:
            print(f"✅ {description} completed successfully")
            return True
    except Exception as e:
        print(f"❌ Error running command: {e}")
        return False


def main():
    """Execute the full pipeline from Step 3 onwards."""
    print("Starting WVS Analysis Pipeline (Steps 3-5)")
    print("=" * 60)
    
    # Change to project root directory
    project_root = Path(__file__).parent
    os.chdir(project_root)
    print(f"Working directory: {project_root}")
    
    # Create necessary directories
    Path("spec").mkdir(exist_ok=True)
    Path("outputs").mkdir(exist_ok=True)
    
    # Step 3: Generate Research Ideas
    if not run_command(
        "uv run python src/generate_research_ideas.py",
        "Step 3: Generate research hypotheses"
    ):
        print("\n❌ Failed at Step 3. Exiting.")
        return 1
    
    # Step 4: Run Analysis
    if not run_command(
        "uv run python src/analysis.py",
        "Step 4: Execute statistical analyses"
    ):
        print("\n❌ Failed at Step 4. Exiting.")
        return 1
    
    # Step 5: Generate Paper
    if not run_command(
        "uv run python src/generate_paper.py",
        "Step 5: Generate LaTeX paper"
    ):
        print("\n❌ Failed at Step 5. Exiting.")
        return 1
    
    # Compile LaTeX to PDF
    print("\n" + "="*60)
    print("Compiling LaTeX to PDF...")
    print("="*60)
    
    # Change to outputs directory for LaTeX compilation
    outputs_dir = project_root / "outputs"
    os.chdir(outputs_dir)
    
    # Run pdflatex twice to resolve references
    latex_success = True
    for i in range(2):
        if not run_command(
            "pdflatex -interaction=nonstopmode paper.tex",
            f"LaTeX compilation pass {i+1}"
        ):
            latex_success = False
            break
    
    # Change back to project root
    os.chdir(project_root)
    
    # Final summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print("\nGenerated files:")
    print("  - spec/research.yaml          (Research hypotheses)")
    print("  - outputs/analysis_report.md  (Analysis results)")
    print("  - outputs/research*.png       (Visualizations)")
    print("  - outputs/paper.tex          (LaTeX paper)")
    
    if latex_success and (outputs_dir / "paper.pdf").exists():
        print("  - outputs/paper.pdf          (Final PDF)")
    else:
        print("\n⚠️  Note: PDF compilation failed. You may need to:")
        print("    1. Install LaTeX (e.g., 'brew install --cask mactex' on macOS)")
        print("    2. Manually compile: cd outputs && pdflatex paper.tex")
    
    print("\n✅ Pipeline execution completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())