#!/usr/bin/env python3
"""Run the complete dynamic WVS analysis pipeline."""
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_research_ideas import save_research_proposals
from analysis import DynamicWVSAnalysis
from generate_paper import DynamicPaperGenerator


def run_pipeline():
    """Run the complete analysis pipeline."""
    print("=" * 60)
    print("WVS Wave 7 Dynamic Analysis Pipeline")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return
    
    try:
        # Step 1: Generate research ideas
        print("\n[Step 1] Generating research ideas from questionnaire...")
        save_research_proposals()
        print("✓ Research ideas generated successfully")
        
        # Step 2: Run dynamic analysis
        print("\n[Step 2] Running dynamic analysis based on generated ideas...")
        analysis = DynamicWVSAnalysis()
        results = analysis.run_all_analyses()
        print("✓ Analysis completed successfully")
        
        # Step 3: Generate paper
        print("\n[Step 3] Generating academic paper from analysis results...")
        generator = DynamicPaperGenerator()
        paper_path = generator.save_paper_with_feedback()
        print("✓ Paper generated successfully")
        
        print("\n" + "=" * 60)
        print("Pipeline completed successfully!")
        print("\nOutputs generated:")
        print("- Research proposals: spec/research.yaml")
        print("- Analysis report: outputs/dynamic_analysis_report.md")
        print("- LaTeX paper: outputs/dynamic_paper.tex")
        print("- PDF paper: outputs/dynamic_paper.pdf (if LaTeX compiled)")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_pipeline()