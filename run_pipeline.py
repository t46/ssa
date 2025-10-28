#!/usr/bin/env python3
"""
WVS Wave 7 Social Science Research Automation Pipeline

This script executes the complete research pipeline described in the PRD:
- Step 3: LLM research idea generation  
- Step 4: Analysis code generation and execution
- Step 5: Academic paper generation in LaTeX format

Prerequisites:
- Steps 0-2 should be completed (data download, preprocessing, semantic search setup)
- ANTHROPIC_API_KEY_SSA environment variable must be set
"""

import sys
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from pipeline_base import (
    check_prerequisites, 
    step_3_generate_research_ideas, 
    step_4_run_analysis, 
    step_5_generate_paper,
    cleanup_directories,
    execute_pipeline_steps,
    execute_pipeline_for_proposal,
    display_pipeline_summary
)
from terminal_formatter import formatter, MessageType
import yaml


def copy_paper_outputs(proposal_id: int):
    """Copy paper outputs to numbered files for the given proposal."""
    outputs_dir = Path("outputs")
    
    # Copy paper.tex to paper_{proposal_id}.tex
    if (outputs_dir / "paper.tex").exists():
        shutil.copy(outputs_dir / "paper.tex", outputs_dir / f"paper_{proposal_id + 1}.tex")
        formatter.print(f"Saved paper_{proposal_id + 1}.tex", MessageType.SUCCESS)
    
    # Copy paper.pdf to paper_{proposal_id}.pdf
    if (outputs_dir / "paper.pdf").exists():
        shutil.copy(outputs_dir / "paper.pdf", outputs_dir / f"paper_{proposal_id + 1}.pdf")
        formatter.print(f"Saved paper_{proposal_id + 1}.pdf", MessageType.SUCCESS)
    
    # Copy references.bib if it exists
    if (outputs_dir / "references.bib").exists() and proposal_id == 0:
        # Only copy references.bib once since it's typically shared
        pass


def main():
    """Execute the complete research pipeline."""
    # Load environment variables from .env file
    load_dotenv()
    
    formatter.print("WVS Wave 7 Social Science Research Automation Pipeline", MessageType.SECTION)
    
    # Display LLM provider information
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    formatter.print(f"Using LLM provider: {llm_provider.upper()}", MessageType.INFO)
    
    # Clean up directories before execution
    formatter.print("Cleaning up previous outputs...", MessageType.INFO)
    cleanup_directories(["outputs", "spec"])
    
    # Check prerequisites
    if not check_prerequisites():
        sys.exit(1)
    
    formatter.print("All prerequisites met", MessageType.SUCCESS)
    
    # Step 3: Generate research ideas
    formatter.print("\nSTEP 3: Generating research ideas using LLM", MessageType.SECTION)
    if not step_3_generate_research_ideas():
        formatter.print("Failed to generate research ideas", MessageType.ERROR)
        sys.exit(1)
    
    # Load research.yaml to get number of proposals
    research_path = Path("spec/research.yaml")
    with open(research_path, 'r') as f:
        research_config = yaml.safe_load(f)
    
    proposals = research_config.get('research_proposals', [])
    num_proposals = len(proposals)
    
    formatter.print(f"\nGenerated {num_proposals} research proposal(s)", MessageType.SUCCESS)
    formatter.print("Research proposals saved to spec/research.yaml", MessageType.SUCCESS)
    
    # Process each proposal
    all_results = {}
    for i in range(num_proposals):
        proposal_results = execute_pipeline_for_proposal(i)
        all_results.update(proposal_results)
        
        # Copy outputs to numbered files
        copy_paper_outputs(i)
    
    # Summary
    successful_steps, total_steps = display_pipeline_summary(all_results)
    
    formatter.print(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully", MessageType.INFO)
    
    if successful_steps == total_steps:
        formatter.print("Pipeline completed successfully!", MessageType.SUCCESS)
        formatter.print("\nGenerated outputs:", MessageType.INFO)
        formatter.print("- spec/research.yaml (research proposals)", MessageType.INFO, indent=2)
        
        for i in range(num_proposals):
            formatter.print(f"\nProposal {i + 1}:", MessageType.INFO, indent=2)
            if Path(f"outputs/paper_{i + 1}.tex").exists():
                formatter.print(f"- outputs/paper_{i + 1}.tex (LaTeX)", MessageType.INFO, indent=4)
            if Path(f"outputs/paper_{i + 1}.pdf").exists():
                formatter.print(f"- outputs/paper_{i + 1}.pdf (PDF)", MessageType.INFO, indent=4)
    else:
        formatter.print(f"Pipeline completed with {total_steps - successful_steps} errors", MessageType.WARNING)
        formatter.print("Please check the error messages above for details", MessageType.WARNING)
        sys.exit(1)


if __name__ == "__main__":
    main()