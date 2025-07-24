"""WVS Wave 7 USA Data Analysis Pipeline - Agent-based version."""

import pandas as pd
import numpy as np
import yaml
import json
from pathlib import Path
from typing import Dict, List, Any
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import os
from claude_code_sdk import query, ClaudeCodeOptions
import asyncio
from terminal_formatter import formatter, MessageType

# Set ANTHROPIC_API_KEY from ANTHROPIC_API_KEY_SSA for claude_code_sdk
if os.getenv("ANTHROPIC_API_KEY_SSA"):
    os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY_SSA")

warnings.filterwarnings("ignore")


class AgentBasedWVSAnalysis:
    """Agent-based analysis pipeline for WVS Wave 7 USA data."""

    def __init__(self):
        """Initialize analysis with data and configurations."""
        self.data_path = Path("data/processed/usa_w7.csv")
        self.research_path = Path("spec/research.yaml")
        self.codebook_path = Path("data/code-maps/codebook_map.json")
        self.output_path = Path("outputs")
        self.output_path.mkdir(exist_ok=True)

        # Load data and configurations
        self.df = pd.read_csv(self.data_path)
        with open(self.research_path, "r") as f:
            self.research_config = yaml.safe_load(f)
        with open(self.codebook_path, "r") as f:
            self.codebook = json.load(f)

    def _classify_message_type(self, message_str: str) -> str:
        """Classify the type of message for appropriate formatting."""
        # Check for ResultMessage
        if "ResultMessage(" in message_str:
            return "result"
        
        # Check for SystemMessage
        if "SystemMessage(" in message_str:
            return "system"
        
        # Check for tool-related messages
        system_prefixes = [
            "Called the",
            "Result of calling",
            "Tool:",
            "Function:",
        ]
        
        for prefix in system_prefixes:
            if message_str.startswith(prefix):
                return "system"
        
        # Default to agent message
        return "agent"

    def _format_result_message(self, message, message_str: str):
        """Format and display ResultMessage."""
        try:
            # Extract result details from the message
            if hasattr(message, 'subtype'):
                subtype = getattr(message, 'subtype', 'unknown')
                duration_ms = getattr(message, 'duration_ms', 0)
                num_turns = getattr(message, 'num_turns', 0)
                total_cost_usd = getattr(message, 'total_cost_usd', 0.0)
                
                result_text = f"Analysis completed ({subtype})"
                details = {
                    'Duration': f"{duration_ms/1000:.1f}s",
                    'Turns': str(num_turns),
                    'Cost': f"${total_cost_usd:.4f}"
                }
                
                formatter.print_result_message(result_text, details)
            else:
                # Fallback to string representation
                formatter.print_result_message(message_str[:100] + "..." if len(message_str) > 100 else message_str)
        except Exception as e:
            formatter.print_result_message(f"Result: {str(e)}")

    def _format_system_message(self, message, message_str: str):
        """Format and display SystemMessage."""
        try:
            # Extract content from system message
            if hasattr(message, 'content'):
                content = str(message.content)
                formatter.print_system_message(content)
            else:
                # Extract meaningful parts from string representation
                if "SystemMessage(" in message_str:
                    # Try to extract the content between parentheses
                    start = message_str.find("content='") + 9
                    end = message_str.find("'", start)
                    if start > 8 and end > start:
                        content = message_str[start:end]
                        formatter.print_system_message(content)
                    else:
                        formatter.print_system_message(message_str[:100] + "..." if len(message_str) > 100 else message_str)
                else:
                    formatter.print_system_message(message_str)
        except Exception as e:
            formatter.print_system_message(f"System: {str(e)}")

    def _format_agent_message(self, message, message_str: str):
        """Format and display regular agent messages."""
        try:
            # Extract content from message
            if hasattr(message, 'content'):
                # Handle TextBlock or similar structures
                if isinstance(message.content, list):
                    for block in message.content:
                        if hasattr(block, 'text'):
                            text = block.text.strip()
                            if text:
                                formatter.print_agent_message(text)
                else:
                    text = str(message.content).strip()
                    if text:
                        formatter.print_agent_message(text)
            else:
                # Fallback to string representation if no content attribute
                text = message_str.strip()
                if text:
                    formatter.print_agent_message(text)
        except Exception as e:
            formatter.print_agent_message(f"Agent message error: {str(e)}")

    def preprocess_data(self, variables: List[str]) -> pd.DataFrame:
        """Preprocess data for analysis."""
        # Always include weight variable
        all_vars = list(set(variables + ["W_WEIGHT"]))

        # Filter to only variables that exist in the data
        existing_vars = [var for var in all_vars if var in self.df.columns]
        missing_vars = [var for var in all_vars if var not in self.df.columns]

        if missing_vars:
            formatter.print(f"Variables not found in data: {missing_vars}", MessageType.WARNING)

        df_subset = self.df[existing_vars].copy()

        # Handle missing values based on WVS conventions
        # -1 to -5 are typically missing value codes
        for col in df_subset.columns:
            if col != "W_WEIGHT":  # Don't modify weights
                df_subset.loc[:, col] = df_subset[col].replace([-5, -4, -3, -2, -1], np.nan)

        # Apply population weight
        if "W_WEIGHT" in df_subset.columns:
            df_subset.loc[:, "weight"] = df_subset["W_WEIGHT"]
        else:
            df_subset.loc[:, "weight"] = 1.0

        return df_subset

    async def generate_and_execute_analysis(
        self, research: Dict[str, Any], research_id: int
    ) -> Dict[str, Any]:
        """Generate and execute analysis using Claude Code SDK agent."""
        # Create a comprehensive prompt for the agent
        var_info = []
        for var_type, var_list in research["variables"].items():
            if var_list:
                var_info.append(f"{var_type}: {', '.join(var_list)}")

        # Get all variables for preprocessing
        all_vars = []
        for var_type, var_list in research["variables"].items():
            if isinstance(var_list, list):
                all_vars.extend(var_list)

        # Preprocess data
        df = self.preprocess_data(all_vars)
        df_clean = df.dropna()

        if len(df_clean) < 100:
            return {
                "error": f"Insufficient data after preprocessing: {len(df_clean)} records",
                "statistics": {"error": "Analysis failed"},
                "visualizations": [],
                "findings": f"Analysis failed due to insufficient data: {len(df_clean)} records",
            }

        # Save preprocessed data to CSV for agent access
        data_file = self.output_path / f"research_{research_id + 1}_data.csv"
        df_clean.to_csv(data_file, index=False)

        prompt = f"""You are tasked with analyzing WVS (World Values Survey) data for the following research proposal:

Title: {research["title"]}
Objective: {research["objective"]}
Hypotheses: {json.dumps(research["hypotheses"], indent=2)}
Variables: {chr(10).join(var_info)}
Analytical Approach: {research["analytical_approach"]}

The preprocessed data is available at: {data_file}

Your task is to:
1. Load and examine the data
2. Create appropriate indices/scales from the variables
3. Implement the specified analytical approach
4. Test each hypothesis with appropriate statistical methods
5. Generate meaningful visualizations and save them to the outputs/ directory
6. Create a Python script that generates results as a dictionary with keys: 'statistics', 'visualizations', 'findings'
7. Handle any errors or issues that arise during analysis
8. Save all results and visualizations to the outputs/ directory

Use weighted statistics where appropriate (using the 'weight' column).
The data includes all specified variables plus a 'weight' column for population weighting.

Please create and execute a complete analysis pipeline. If you encounter any errors, debug and fix them automatically.
Save your final analysis script as outputs/research_{research_id + 1}_analysis.py
"""

        analysis_log = []
        final_result = None

        formatter.print("Starting agent-based analysis...", MessageType.PROGRESS)
        
        # Configure Claude Code SDK options with full tool permissions
        options = ClaudeCodeOptions(
            max_turns=10,
            system_prompt="You are a data scientist and statistician specialized in social science research. You have access to read and write files, execute bash commands, and use all available tools.",
            cwd=Path.cwd(),
            allowed_tools=["Read", "Write", "Bash", "Editor", "Computer", "Browser", "Terminal"],
            permission_mode="acceptEdits"
        )
        
        try:
            async for message in query(prompt=prompt, options=options):
                # Convert message to string for logging
                message_str = str(message)
                analysis_log.append(message_str)
                
                # Classify message type for appropriate formatting
                msg_type = self._classify_message_type(message_str)
                
                # Handle different message types
                if msg_type == "result":
                    # Handle ResultMessage
                    self._format_result_message(message, message_str)
                elif msg_type == "system":
                    # Handle SystemMessage
                    self._format_system_message(message, message_str)
                else:
                    # Handle regular agent messages
                    self._format_agent_message(message, message_str)

            # Try to load results from the analysis script if it was created
            analysis_script_path = (
                self.output_path / f"research_{research_id + 1}_analysis.py"
            )
            if analysis_script_path.exists():
                # Execute the generated analysis script
                exec_globals = {
                    "pd": pd,
                    "np": np,
                    "sm": sm,
                    "smf": smf,
                    "stats": stats,
                    "plt": plt,
                    "sns": sns,
                    "StandardScaler": StandardScaler,
                    "SimpleImputer": SimpleImputer,
                    "Path": Path,
                    "output_path": self.output_path,
                    "research_id": research_id,
                }

                with open(analysis_script_path, "r") as f:
                    analysis_code = f.read()

                exec(analysis_code, exec_globals)

                if "results" in exec_globals:
                    final_result = exec_globals["results"]
                elif "analyze_data" in exec_globals:
                    final_result = exec_globals["analyze_data"](df_clean)

            # If no results from script, create basic results from log
            if final_result is None:
                final_result = {
                    "statistics": {
                        "analysis_completed": True,
                        "sample_size": len(df_clean),
                    },
                    "visualizations": [],
                    "findings": " ".join(analysis_log[-5:])
                    if analysis_log
                    else "Analysis completed by agent",
                }

        except Exception as e:
            formatter.print(f"Error in agent-based analysis: {e}", MessageType.ERROR)
            final_result = {
                "error": str(e),
                "statistics": {"error": "Agent analysis failed"},
                "visualizations": [],
                "findings": f"Agent-based analysis failed due to: {str(e)}",
            }

        # Save analysis log
        log_file = self.output_path / f"research_{research_id + 1}_agent_log.txt"
        with open(log_file, "w") as f:
            f.write("\n".join(analysis_log))

        return final_result

    def execute_agent_analysis(self, research_id: int) -> Dict[str, Any]:
        """Execute agent-based analysis for a specific research proposal."""
        research = self.research_config["research_proposals"][research_id]
        results = {
            "research_id": research_id + 1,
            "title": research["title"],
            "hypotheses": research["hypotheses"],
        }

        try:
            # Run the agent-based analysis
            agent_results = asyncio.run(
                self.generate_and_execute_analysis(research, research_id)
            )
            results.update(agent_results)

        except Exception as e:
            formatter.print(f"Error in agent-based analysis for research {research_id + 1}: {e}", MessageType.ERROR)
            results["error"] = str(e)
            results["statistics"] = {"error": "Agent analysis failed"}
            results["visualizations"] = []
            results["findings"] = f"Agent-based analysis failed due to: {str(e)}"

        return results

    def generate_summary_statistics(
        self, df: pd.DataFrame, research: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary statistics for the research variables."""
        stats_dict = {}

        # Get all variables
        all_vars = []
        for var_list in research["variables"].values():
            if isinstance(var_list, list):
                all_vars.extend(var_list)

        # Calculate basic statistics
        for var in all_vars:
            if var in df.columns and var != "W_WEIGHT":
                stats_dict[var] = {
                    "mean": np.average(
                        df[var].dropna(), weights=df.loc[df[var].notna(), "weight"]
                    ),
                    "std": np.sqrt(
                        np.average(
                            (
                                df[var].dropna()
                                - np.average(
                                    df[var].dropna(),
                                    weights=df.loc[df[var].notna(), "weight"],
                                )
                            )
                            ** 2,
                            weights=df.loc[df[var].notna(), "weight"],
                        )
                    ),
                    "n": df[var].notna().sum(),
                    "missing": df[var].isna().sum(),
                }

        return stats_dict

    def generate_report(self, results: List[Dict[str, Any]]):
        """Generate analysis report."""
        report_path = self.output_path / "dynamic_analysis_report.md"

        with open(report_path, "w") as f:
            f.write("# WVS Wave 7 USA Data Analysis Results (Dynamic)\n\n")
            f.write(f"Dataset: {self.research_config['metadata']['dataset']}\n")
            f.write(f"Country: {self.research_config['metadata']['country']}\n")
            f.write(
                f"Sample Weight: {self.research_config['metadata']['sample_weight']}\n\n"
            )

            for result in results:
                f.write(f"## Research {result['research_id']}: {result['title']}\n\n")

                if "error" in result:
                    f.write(f"### Error:\n{result['error']}\n\n")
                    continue

                # Write hypotheses
                f.write("### Hypotheses:\n")
                for i, hyp in enumerate(result.get("hypotheses", []), 1):
                    f.write(f"{i}. {hyp}\n")
                f.write("\n")

                # Write statistics
                if "statistics" in result:
                    f.write("### Key Statistics:\n")
                    stats = result["statistics"]
                    if isinstance(stats, dict):
                        for key, value in stats.items():
                            if isinstance(value, (int, float)):
                                f.write(f"- {key}: {value:.3f}\n")
                            else:
                                f.write(f"- {key}: {value}\n")
                    f.write("\n")

                # Write findings
                if "findings" in result:
                    f.write("### Findings:\n")
                    findings = result["findings"]
                    if isinstance(findings, dict):
                        for key, value in findings.items():
                            f.write(f"**{key}:** {value}\n\n")
                    else:
                        f.write(str(findings))
                    f.write("\n\n")

                # Reference visualizations
                if "visualizations" in result and result["visualizations"]:
                    f.write("### Visualizations:\n")
                    for viz in result["visualizations"]:
                        f.write(f"![{viz}]({viz})\n")
                    f.write("\n")

        formatter.print(f"Report generated: {report_path}", MessageType.SUCCESS)

    def run_all_analyses(self):
        """Run all research analyses using agent-based approach."""
        results = []

        formatter.print("WVS Analysis Pipeline", MessageType.SECTION)
        total_proposals = len(self.research_config["research_proposals"])
        
        for i, research in enumerate(self.research_config["research_proposals"]):
            formatter.print_progress(i, total_proposals, "Processing research proposals")
            formatter.print(f"Research {i + 1}: {research['title']}", MessageType.SUBSECTION)
            
            try:
                result = self.execute_agent_analysis(i)
                results.append(result)
                formatter.print(f"Completed Research {i + 1}", MessageType.SUCCESS)
            except Exception as e:
                formatter.print(f"Error in Research {i + 1}: {e}", MessageType.ERROR)
                results.append(
                    {"research_id": i + 1, "title": research["title"], "error": str(e)}
                )

        formatter.print_progress(total_proposals, total_proposals, "Processing research proposals")
        formatter.clear_line()
        
        # Generate report
        self.generate_report(results)

        return results


if __name__ == "__main__":
    analysis = AgentBasedWVSAnalysis()
    results = analysis.run_all_analyses()
    formatter.print("\nAgent-based analysis complete! Check the outputs/ directory for results.", MessageType.SUCCESS)
