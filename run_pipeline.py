#!/usr/bin/env python3
"""
WVS Wave 7 Social Science Research Automation Pipeline

This script executes the complete research pipeline described in the PRD:
- Step 3: LLM research idea generation  
- Step 4: Dynamic analysis code generation and execution
- Step 5: Academic paper generation in LaTeX format

Prerequisites:
- Steps 0-2 should be completed (data download, preprocessing, semantic search setup)
- ANTHROPIC_API_KEY_SSA environment variable must be set
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from generate_research_ideas import save_research_proposals
from generate_and_execute_analysis import AgentBasedWVSAnalysis
from generate_paper import PaperGenerator
from terminal_formatter import formatter, MessageType


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""
    required_files = [
        "data/processed/usa_w7.csv",
        "data/code-maps/codebook_map.json", 
        "data/code-maps/questionnaire_map.json",
        "data/raw/F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        formatter.print("Prerequisites not met. Missing files:", MessageType.ERROR)
        for file in missing_files:
            formatter.print(f"- {file}", MessageType.ERROR, indent=2)
        formatter.print("Please ensure Steps 0-2 have been completed.", MessageType.WARNING)
        return False
    
    if not os.getenv("ANTHROPIC_API_KEY_SSA"):
        formatter.print("ANTHROPIC_API_KEY_SSA environment variable not set.", MessageType.ERROR)
        return False
    
    return True


def step_3_generate_research_ideas() -> bool:
    """Execute Step 3: LLM research idea generation."""
    formatter.print("STEP 3: Generating research ideas using LLM", MessageType.SECTION)
    
    try:
        proposals = save_research_proposals()
        formatter.print(f"Successfully generated {len(proposals)} research proposal(s)", MessageType.SUCCESS)
        formatter.print("Research proposals saved to spec/research.yaml", MessageType.SUCCESS)
        return True
    except Exception as e:
        formatter.print(f"Error in Step 3: {e}", MessageType.ERROR)
        return False


def step_4_run_analysis() -> bool:
    """Execute Step 4: Dynamic analysis code generation and execution."""
    formatter.print("STEP 4: Running dynamic analysis", MessageType.SECTION)
    
    try:
        analysis = AgentBasedWVSAnalysis()
        results = analysis.run_all_analyses()
        
        successful_analyses = [r for r in results if 'error' not in r]
        failed_analyses = [r for r in results if 'error' in r]
        
        formatter.print(f"Analysis completed: {len(successful_analyses)} successful, {len(failed_analyses)} failed", MessageType.SUCCESS)
        formatter.print("Analysis report saved to outputs/dynamic_analysis_report.md", MessageType.SUCCESS)
        
        if failed_analyses:
            formatter.print("Failed analyses:", MessageType.WARNING)
            for analysis in failed_analyses:
                formatter.print(f"Research {analysis['research_id']}: {analysis.get('error', 'Unknown error')}", MessageType.ERROR, indent=2)
        
        return len(successful_analyses) > 0
    except Exception as e:
        formatter.print(f"Error in Step 4: {e}", MessageType.ERROR)
        return False


def step_5_generate_paper() -> bool:
    """Execute Step 5: Academic paper generation."""
    formatter.print("STEP 5: Generating academic paper", MessageType.SECTION)
    
    try:
        generator = PaperGenerator()
        latex_path = generator.save_paper_with_multi_step_generation()
        
        if latex_path:
            formatter.print("LaTeX paper generated successfully", MessageType.SUCCESS)
            formatter.print(f"Paper saved to: {latex_path}", MessageType.SUCCESS)
            
            # Check if PDF was generated
            pdf_path = latex_path.with_suffix('.pdf')
            if pdf_path.exists():
                formatter.print(f"PDF compiled successfully: {pdf_path}", MessageType.SUCCESS)
            else:
                formatter.print("LaTeX file generated but PDF compilation may have failed", MessageType.WARNING)
            
            return True
        else:
            formatter.print("Failed to generate LaTeX paper", MessageType.ERROR)
            return False
    except Exception as e:
        formatter.print(f"Error in Step 5: {e}", MessageType.ERROR)
        return False


def main():
    """Execute the complete research pipeline."""
    formatter.print("WVS Wave 7 Social Science Research Automation Pipeline", MessageType.SECTION)
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    formatter.print("All prerequisites met", MessageType.SUCCESS)
    
    # Execute pipeline steps
    steps = [
        ("Step 3: Research Ideas Generation", step_3_generate_research_ideas),
        ("Step 4: Dynamic Analysis", step_4_run_analysis),
        ("Step 5: Paper Generation", step_5_generate_paper)
    ]
    
    results = {}
    for step_name, step_func in steps:
        success = step_func()
        results[step_name] = success
        
        if not success:
            formatter.print(f"Warning: {step_name} failed. Continuing with next step...", MessageType.WARNING)
    
    # Summary
    formatter.print("PIPELINE EXECUTION SUMMARY", MessageType.SECTION)
    
    for step_name, success in results.items():
        if success:
            formatter.print(f"{step_name:<35} SUCCESS", MessageType.SUCCESS)
        else:
            formatter.print(f"{step_name:<35} FAILED", MessageType.ERROR)
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    formatter.print(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully", MessageType.INFO)
    
    if successful_steps == total_steps:
        formatter.print("Pipeline completed successfully!", MessageType.SUCCESS)
        formatter.print("\nGenerated outputs:", MessageType.INFO)
        formatter.print("- spec/research.yaml (research proposals)", MessageType.INFO, indent=2)
        formatter.print("- outputs/dynamic_analysis_report.md (analysis results)", MessageType.INFO, indent=2)
        formatter.print("- outputs/paper.tex (academic paper)", MessageType.INFO, indent=2)
        if Path("outputs/paper.pdf").exists():
            formatter.print("- outputs/paper.pdf (compiled paper)", MessageType.INFO, indent=2)
    else:
        formatter.print(f"Pipeline completed with {total_steps - successful_steps} errors", MessageType.WARNING)
        formatter.print("Please check the error messages above for details", MessageType.WARNING)
        sys.exit(1)


if __name__ == "__main__":
    main()