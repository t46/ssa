#!/usr/bin/env python3
"""
Base classes and functions for WVS research pipeline scripts.
This module contains shared functionality to reduce code duplication.
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    
    # Check LLM API keys based on provider
    llm_provider = os.getenv("LLM_PROVIDER", "anthropic").lower()
    
    if llm_provider == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY_SSA"):
            formatter.print("ANTHROPIC_API_KEY_SSA environment variable not set.", MessageType.ERROR)
            return False
    elif llm_provider == "openai":
        if not os.getenv("OPENAI_API_KEY_SSA"):
            formatter.print("OPENAI_API_KEY_SSA environment variable not set.", MessageType.ERROR)
            return False
    elif llm_provider == "ollama":
        # Ollama doesn't require API keys, just check if server is running
        try:
            import ollama
            ollama.list()  # Test connection to Ollama server
        except Exception as e:
            formatter.print(f"Failed to connect to Ollama server: {e}", MessageType.ERROR)
            formatter.print("Make sure Ollama server is running with 'ollama serve'", MessageType.WARNING)
            return False
    else:
        formatter.print(f"Unsupported LLM provider: {llm_provider}", MessageType.ERROR)
        return False
    
    return True


def step_3_generate_research_ideas(run_id: str = None) -> bool:
    """Execute Step 3: LLM research idea generation."""
    if run_id:
        formatter.print(f"STEP 3 (Run {run_id}): Generating research ideas using LLM", MessageType.SECTION)
    else:
        formatter.print("STEP 3: Generating research ideas using LLM", MessageType.SECTION)
    
    try:
        proposals = save_research_proposals()
        formatter.print(f"Successfully generated {len(proposals)} research proposal(s)", MessageType.SUCCESS)
        formatter.print("Research proposals saved to spec/research.yaml", MessageType.SUCCESS)
        return True
    except Exception as e:
        formatter.print(f"Error in Step 3: {e}", MessageType.ERROR)
        return False


def step_4_run_analysis(run_id: str = None, proposal_id: int = None) -> bool:
    """Execute Step 4: Analysis code generation and execution.
    
    Args:
        run_id: Optional run identifier
        proposal_id: Optional proposal index (0-based). If None, runs all analyses.
        
    Returns:
        bool: True if analysis completed successfully
    """
    if run_id:
        formatter.print(f"STEP 4 (Run {run_id}): Running analysis", MessageType.SECTION)
    else:
        formatter.print("STEP 4: Running analysis", MessageType.SECTION)
    
    try:
        analysis = AgentBasedWVSAnalysis()
        
        if proposal_id is not None:
            # Run analysis for a specific proposal
            formatter.print(f"Running analysis for proposal {proposal_id + 1}", MessageType.INFO)
            result = analysis.run_single_analysis(proposal_id)
            
            if 'error' in result:
                formatter.print(f"Analysis failed: {result.get('error', 'Unknown error')}", MessageType.ERROR)
                return False
            else:
                formatter.print("Analysis completed successfully", MessageType.SUCCESS)
                return True
        else:
            # Run all analyses
            results = analysis.run_all_analyses()
            
            successful_analyses = [r for r in results if 'error' not in r]
            failed_analyses = [r for r in results if 'error' in r]
            
            formatter.print(f"Analysis completed: {len(successful_analyses)} successful, {len(failed_analyses)} failed", MessageType.SUCCESS)
            formatter.print("Analysis report saved to outputs/analysis_report.md", MessageType.SUCCESS)
            
            if failed_analyses:
                formatter.print("Failed analyses:", MessageType.WARNING)
                for analysis_result in failed_analyses:
                    formatter.print(f"Research {analysis_result['research_id']}: {analysis_result.get('error', 'Unknown error')}", MessageType.ERROR, indent=2)
            
            return len(successful_analyses) > 0
    except Exception as e:
        formatter.print(f"Error in Step 4: {e}", MessageType.ERROR)
        return False


def step_5_generate_paper(run_id: str = None, proposal_id: int = None) -> bool:
    """Execute Step 5: Academic paper generation.
    
    Args:
        run_id: Optional run identifier
        proposal_id: Optional proposal index (0-based). If None, generates paper for first proposal.
        
    Returns:
        bool: True if paper generation completed successfully
    """
    if run_id:
        formatter.print(f"STEP 5 (Run {run_id}): Generating academic paper", MessageType.SECTION)
    else:
        formatter.print("STEP 5: Generating academic paper", MessageType.SECTION)
    
    try:
        generator = PaperGenerator(proposal_id=proposal_id)
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


def cleanup_directories(directories_to_clean: List[str]) -> None:
    """Clean up specified directories."""
    for dir_name in directories_to_clean:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                formatter.print(f"Cleaned up {dir_name} directory", MessageType.INFO)
            except Exception as e:
                formatter.print(f"Warning: Could not clean up {dir_name} directory: {e}", MessageType.WARNING)
        else:
            formatter.print(f"{dir_name} directory does not exist, skipping cleanup", MessageType.INFO)


def execute_pipeline_steps(run_id: str = None) -> Dict[str, bool]:
    """Execute the standard pipeline steps and return results."""
    steps = [
        ("Step 3: Research Ideas Generation", step_3_generate_research_ideas),
        ("Step 4: Analysis", step_4_run_analysis),
        ("Step 5: Paper Generation", step_5_generate_paper)
    ]
    
    results = {}
    for step_name, step_func in steps:
        success = step_func(run_id) if run_id else step_func()
        results[step_name] = success
        
        if not success:
            formatter.print(f"Warning: {step_name} failed. Continuing with next step...", MessageType.WARNING)
    
    return results


def execute_pipeline_for_proposal(proposal_id: int, run_id: str = None) -> Dict[str, bool]:
    """Execute analysis and paper generation for a specific research proposal.
    
    Args:
        proposal_id: Index of the research proposal (0-based)
        run_id: Optional run identifier
        
    Returns:
        Dict mapping step names to success status
    """
    formatter.print(f"\n{'='*70}", MessageType.INFO)
    formatter.print(f"Processing Research Proposal {proposal_id + 1}", MessageType.SECTION)
    formatter.print(f"{'='*70}\n", MessageType.INFO)
    
    steps = [
        ("Step 4: Analysis", lambda rid: step_4_run_analysis(rid, proposal_id)),
        ("Step 5: Paper Generation", lambda rid: step_5_generate_paper(rid, proposal_id))
    ]
    
    results = {}
    for step_name, step_func in steps:
        success = step_func(run_id) if run_id else step_func(None)
        results[f"{step_name} (Proposal {proposal_id + 1})"] = success
        
        if not success:
            formatter.print(f"Warning: {step_name} for proposal {proposal_id + 1} failed.", MessageType.WARNING)
    
    return results


def display_pipeline_summary(results: Dict[str, bool]) -> None:
    """Display pipeline execution summary."""
    formatter.print("PIPELINE EXECUTION SUMMARY", MessageType.SECTION)
    
    for step_name, success in results.items():
        if success:
            formatter.print(f"{step_name:<35} SUCCESS", MessageType.SUCCESS)
        else:
            formatter.print(f"{step_name:<35} FAILED", MessageType.ERROR)
    
    successful_steps = sum(results.values())
    total_steps = len(results)
    
    formatter.print(f"\nOverall: {successful_steps}/{total_steps} steps completed successfully", MessageType.INFO)
    
    return successful_steps, total_steps


class BasePipeline:
    """Base class for pipeline management with archiving capabilities."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent  # Go up one level from src/
        self.outputs_dir = self.base_dir / "outputs"
        self.spec_dir = self.base_dir / "spec"
        self.archive_dir = self.base_dir / "archived_runs"
        self.run_history_file = self.archive_dir / "run_history.json"
        
    def get_next_run_id(self) -> str:
        """Get the next run ID (01, 02, 03, ...)."""
        if not self.archive_dir.exists():
            return "01"
        
        # Find existing run directories
        run_dirs = [d for d in self.archive_dir.iterdir() 
                   if d.is_dir() and d.name.startswith("run_")]
        
        if not run_dirs:
            return "01"
        
        # Extract run numbers and find the next one
        run_numbers = []
        for run_dir in run_dirs:
            try:
                run_num = int(run_dir.name.split("_")[1])
                run_numbers.append(run_num)
            except (IndexError, ValueError):
                continue
        
        if run_numbers:
            next_num = max(run_numbers) + 1
        else:
            next_num = 1
        
        return f"{next_num:02d}"
    
    def archive_current_run(self, run_id: str, run_info: Dict[str, Any]) -> None:
        """Archive the current run's outputs and spec files."""
        self.archive_dir.mkdir(exist_ok=True)
        
        run_archive_dir = self.archive_dir / f"run_{run_id}"
        run_archive_dir.mkdir(exist_ok=True)
        
        # Archive outputs directory
        if self.outputs_dir.exists():
            archived_outputs = run_archive_dir / "outputs"
            if archived_outputs.exists():
                shutil.rmtree(archived_outputs)
            shutil.copytree(self.outputs_dir, archived_outputs)
            formatter.print(f"Archived outputs to {archived_outputs}", MessageType.INFO)
        
        # Archive spec directory
        if self.spec_dir.exists():
            archived_spec = run_archive_dir / "spec"
            if archived_spec.exists():
                shutil.rmtree(archived_spec)
            shutil.copytree(self.spec_dir, archived_spec)
            formatter.print(f"Archived spec to {archived_spec}", MessageType.INFO)
        
        # Save run metadata
        run_metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "success": run_info.get("success", False),
            "steps_completed": run_info.get("steps_completed", 0),
            "total_steps": run_info.get("total_steps", 0),
            "errors": run_info.get("errors", []),
            "pdf_generated": (archived_outputs / "paper.pdf").exists() if archived_outputs.exists() else False,
            "execution_time": run_info.get("execution_time", 0)
        }
        
        metadata_file = run_archive_dir / "run_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(run_metadata, f, indent=2, ensure_ascii=False)
        
        # Update run history
        self.update_run_history(run_metadata)
        
        formatter.print(f"Run {run_id} archived successfully", MessageType.SUCCESS)
    
    def update_run_history(self, run_metadata: Dict[str, Any]) -> None:
        """Update the global run history file."""
        history = []
        
        if self.run_history_file.exists():
            try:
                with open(self.run_history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                history = []
        
        history.append(run_metadata)
        
        with open(self.run_history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    
    def cleanup_current_directories(self) -> None:
        """Clean up current spec and outputs directories to start fresh each run."""
        directories_to_clean = [self.spec_dir, self.outputs_dir]
        
        for dir_path in directories_to_clean:
            if dir_path.exists():
                try:
                    shutil.rmtree(dir_path)
                    formatter.print(f"Cleaned up {dir_path.name} directory", MessageType.INFO)
                except Exception as e:
                    formatter.print(f"Warning: Could not clean up {dir_path.name} directory: {e}", MessageType.WARNING)
            else:
                formatter.print(f"{dir_path.name} directory does not exist, skipping cleanup", MessageType.INFO)
    
    def display_run_history(self) -> None:
        """Display history of previous runs."""
        if not self.run_history_file.exists():
            formatter.print("No previous runs found", MessageType.INFO)
            return
        
        try:
            with open(self.run_history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            formatter.print("Could not read run history", MessageType.WARNING)
            return
        
        if not history:
            formatter.print("No previous runs found", MessageType.INFO)
            return
        
        formatter.print("PREVIOUS RUNS HISTORY", MessageType.SECTION)
        formatter.print(f"{'Run ID':<8} {'Date':<20} {'Success':<8} {'Steps':<10} {'PDF':<5} {'Time(s)':<8}", MessageType.INFO)
        formatter.print("-" * 70, MessageType.INFO)
        
        for run in history:
            timestamp = datetime.fromisoformat(run['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            success_str = "✓" if run.get('success', False) else "✗"
            steps_str = f"{run.get('steps_completed', 0)}/{run.get('total_steps', 0)}"
            pdf_str = "✓" if run.get('pdf_generated', False) else "✗"
            exec_time = f"{run.get('execution_time', 0):.1f}"
            
            formatter.print(f"{run['run_id']:<8} {timestamp:<20} {success_str:<8} {steps_str:<10} {pdf_str:<5} {exec_time:<8}", MessageType.INFO)
