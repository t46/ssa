#!/usr/bin/env python3
"""Run the complete dynamic WVS analysis pipeline."""
import os
import sys
import traceback
from pathlib import Path
from typing import Dict, Any, List

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from generate_research_ideas import save_research_proposals
from analysis import DynamicWVSAnalysis
from generate_paper import DynamicPaperGenerator


def run_pipeline():
    """Run the complete analysis pipeline with robust error handling."""
    print("=" * 60)
    print("WVS Wave 7 Analysis Pipeline")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("\nError: ANTHROPIC_API_KEY environment variable not set.")
        print("Please set it with: export ANTHROPIC_API_KEY='your-api-key'")
        return False
    
    pipeline_results = {
        'research_ideas': False,
        'analysis': False,
        'paper': False,
        'errors': []
    }
    
    try:
        # Step 1: Generate research ideas
        print("\n[Step 1] Generating research ideas from questionnaire...")
        try:
            save_research_proposals()
            print("✓ Research ideas generated successfully")
            pipeline_results['research_ideas'] = True
        except Exception as e:
            error_msg = f"Failed to generate research ideas: {str(e)}"
            print(f"✗ {error_msg}")
            pipeline_results['errors'].append(error_msg)
            print("Continuing with existing research.yaml if available...")
        
        # Step 2: Run dynamic analysis
        print("\n[Step 2] Running dynamic analysis based on generated ideas...")
        try:
            analysis = DynamicWVSAnalysis()
            results = analysis.run_all_analyses()
            print("✓ Analysis completed successfully")
            pipeline_results['analysis'] = True
        except Exception as e:
            error_msg = f"Failed to run analysis: {str(e)}"
            print(f"✗ {error_msg}")
            pipeline_results['errors'].append(error_msg)
            print("Continuing to paper generation with available results...")
        
        # Step 3: Generate paper
        print("\n[Step 3] Generating academic paper from analysis results...")
        try:
            generator = DynamicPaperGenerator()
            paper_path = generator.save_paper_with_feedback()
            print("✓ Paper generated successfully")
            pipeline_results['paper'] = True
        except Exception as e:
            error_msg = f"Failed to generate paper: {str(e)}"
            print(f"✗ {error_msg}")
            pipeline_results['errors'].append(error_msg)
        
        # Summary report
        print("\n" + "=" * 60)
        print("Pipeline Summary:")
        print("=" * 60)
        
        steps_completed = sum([pipeline_results['research_ideas'], 
                             pipeline_results['analysis'], 
                             pipeline_results['paper']])
        
        if steps_completed == 3:
            print("🎉 All steps completed successfully!")
        elif steps_completed > 0:
            print(f"⚠️  Partial success: {steps_completed}/3 steps completed")
        else:
            print("❌ Pipeline failed completely")
        
        print(f"\nStep Results:")
        print(f"- Research ideas: {'✓' if pipeline_results['research_ideas'] else '✗'}")
        print(f"- Analysis: {'✓' if pipeline_results['analysis'] else '✗'}")
        print(f"- Paper generation: {'✓' if pipeline_results['paper'] else '✗'}")
        
        if pipeline_results['errors']:
            print(f"\nErrors encountered:")
            for i, error in enumerate(pipeline_results['errors'], 1):
                print(f"{i}. {error}")
        
        print("\nOutputs generated:")
        if pipeline_results['research_ideas']:
            print("- Research proposals: spec/research.yaml")
        if pipeline_results['analysis']:
            print("- Analysis report: outputs/dynamic_analysis_report.md")
        if pipeline_results['paper']:
            print("- LaTeX paper: outputs/dynamic_paper.tex")
            print("- PDF paper: outputs/dynamic_paper.pdf (if LaTeX compiled)")
        
        print("=" * 60)
        
        return steps_completed > 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error in pipeline: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

def run_pipeline_with_retry(max_retries: int = 3):
    """Run pipeline with retry mechanism for failed steps."""
    print("=" * 60)
    print("WVS Wave 7 Analysis Pipeline (with retry)")
    print("=" * 60)
    
    for attempt in range(max_retries + 1):
        print(f"\n--- Attempt {attempt}/{max_retries} ---")
        
        success = run_pipeline()
        
        if success:
            print(f"\n🎉 Pipeline completed successfully on attempt {attempt}")
            return True     
        elif attempt < max_retries:
            print(f"\n⚠️  Pipeline failed on attempt {attempt}. Retrying...")
            print("Waiting 5 seconds before retry...")
            import time
            time.sleep(5)
        else:
            print(f"\n❌ Pipeline failed after {max_retries} attempts")
            return False


if __name__ == "__main__":
    # Check if retry mode is requested
    if len(sys.argv) > 1 and sys.argv[1] == "--retry":
        run_pipeline_with_retry()
    else:
        run_pipeline()