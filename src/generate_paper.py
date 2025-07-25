"""Generate LaTeX paper from analysis results - Dynamic version."""
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import anthropic
import os
import subprocess
from claude_code_sdk import query, ClaudeCodeOptions
import asyncio
from src.terminal_formatter import formatter, MessageType
import re


class DynamicPaperGenerator:
    """Generate academic paper in LaTeX format dynamically based on analysis results."""
    
    def __init__(self):
        """Initialize paper generator with paths and data."""
        self.research_path = Path("spec/research.yaml")
        self.report_path = Path("outputs/dynamic_analysis_report.md")
        self.output_path = Path("outputs")
        
        # Load configurations
        with open(self.research_path, 'r') as f:
            self.research_config = yaml.safe_load(f)
            
        # Initialize Anthropic client
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY_SSA"))
        
        # Set ANTHROPIC_API_KEY from ANTHROPIC_API_KEY_SSA for claude_code_sdk
        if os.getenv("ANTHROPIC_API_KEY_SSA"):
            os.environ["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY_SSA")
    
    def load_analysis_results(self) -> str:
        """Load analysis results from the report."""
        if self.report_path.exists():
            with open(self.report_path, 'r') as f:
                return f.read()
        else:
            # Fallback to original report if dynamic one doesn't exist
            fallback_path = self.output_path / 'analysis_report.md'
            if fallback_path.exists():
                with open(fallback_path, 'r') as f:
                    return f.read()
            return ""
    
    def _extract_figure_descriptions(self, analysis_results: str) -> Dict[str, str]:
        """Extract figure descriptions from analysis results."""
        figure_descriptions = {}
        
        # Common patterns for different figure types
        patterns = {
            "correlation_matrix": "correlation matrix|correlation heatmap|variable correlations",
            "descriptive_distributions": "distribution|histogram|density plot|descriptive statistics",
            "hypothesis_1": "hypothesis 1|h1:|first hypothesis|digital engagement.*trust",
            "hypothesis_2": "hypothesis 2|h2:|second hypothesis|moderation|digital capability",
            "hypothesis_3": "hypothesis 3|h3:|third hypothesis|mediation|online social capital",
            "hypothesis_4": "hypothesis 4|h4:|fourth hypothesis|platform type|platform differences"
        }
        
        # Search for figure mentions in the analysis results
        lines = analysis_results.lower().split('\n')
        for i, line in enumerate(lines):
            for fig_type, pattern in patterns.items():
                if re.search(pattern, line, re.IGNORECASE):
                    # Get context from surrounding lines
                    context_start = max(0, i - 2)
                    context_end = min(len(lines), i + 3)
                    context = ' '.join(lines[context_start:context_end])
                    figure_descriptions[fig_type] = context[:200]  # Limit length
        
        return figure_descriptions
    
    def generate_paper_content(self, compile_error_feedback: Optional[str] = None) -> str:
        """Generate paper content dynamically using LLM, with optional compile error feedback."""
        # Load analysis results
        analysis_results = self.load_analysis_results()
        
        # Get list of actual figure files
        figure_files = sorted(self.output_path.glob("*.png"))
        figure_names = [f.name for f in figure_files]
        
        # Create a mapping of figure descriptions from the analysis results
        figure_mapping = self._extract_figure_descriptions(analysis_results)
        
        # Create prompt
        prompt = f"""Generate a complete academic paper in LaTeX format based on the following research and analysis results.

Research Configuration:
{json.dumps(self.research_config, indent=2)}

Analysis Results:
{analysis_results}

Available Figure Files:
{json.dumps(figure_names, indent=2)}

Figure Descriptions from Analysis:
{json.dumps(figure_mapping, indent=2)}
"""
        if compile_error_feedback:
            prompt += f"\n\nThe previous LaTeX file failed to compile with the following error(s):\n{compile_error_feedback}\nPlease fix these errors and regenerate the LaTeX code. Only output the corrected LaTeX code."
        else:
            prompt += """
The paper should include:
1. Title that reflects all three research topics
2. Abstract (150-200 words) summarizing all findings
3. Introduction with theoretical motivation
4. Literature Review covering relevant theories
5. Methods section describing data and analytical approach
6. Results section with subsections for each research topic
7. Discussion integrating findings across all studies
8. Limitations
9. Conclusion
10. References (in bibtex format)

Use professional academic writing style. Include:
- Proper citations (Author, Year)
- Tables for regression results
- Include all figures listed above in appropriate sections using \includegraphics{filename} (without path)
- Reference figures appropriately in the text (e.g., "As shown in Figure 1...")
- Statistical notation (p-values, coefficients, R-squared)

IMPORTANT FIGURE INSERTION REQUIREMENTS:
1. Include ALL the PNG figures listed above in the paper at appropriate locations
2. Each figure should have:
   - A descriptive caption that explains what the figure shows
   - A reference in the main text before the figure appears
   - Proper LaTeX figure environment with label for cross-referencing
3. Match figures to their content:
   - correlation_matrix.png: Use in methodology or initial results section
   - descriptive_distributions.png: Use in descriptive statistics section
   - hypothesis_X_analysis.png: Use in the corresponding hypothesis testing section
4. Example figure insertion:
   \begin{figure}[htbp]
   \centering
   \includegraphics[width=0.8\textwidth]{hypothesis_1_analysis.png}
   \caption{Analysis results for Hypothesis 1 showing the relationship between digital engagement and social trust}
   \label{fig:hypothesis1}
   \end{figure}

Generate complete LaTeX code that compiles without errors.
"""
        
        response = self.client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=8000,
            temperature=0.5,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Extract LaTeX content
        # Anthropic APIのレスポンスからLaTeXテキストを抽出
        if isinstance(response.content[0], dict) and 'text' in response.content[0]:
            latex_content = response.content[0]['text']
        else:
            latex_content = str(response.content[0])
        # Clean up the content if needed
        if "```latex" in latex_content:
            latex_content = latex_content.split("```latex")[1].split("```", 1)[0]
        elif "```" in latex_content:
            latex_content = latex_content.split("```", 1)[1].split("```", 1)[0]
        return latex_content
    
    def extract_bibliography(self, latex_content: str) -> str:
        """Extract bibliography entries from LaTeX content."""
        # Look for bibliography in the LaTeX content
        bib_entries = []
        
        # Common patterns for extracting citations
        import re
        
        # Extract from \bibitem commands if present
        bibitem_pattern = r'\\bibitem\{([^}]+)\}([^\\]+(?:\\\\[^\\]+)*)'
        bibitems = re.findall(bibitem_pattern, latex_content)
        
        # Convert to BibTeX format if needed
        if bibitems:
            for key, content in bibitems:
                # Simple conversion - in practice would need more sophisticated parsing
                bib_entries.append(f"@article{{{key},\n  title={{{content.strip()}}},\n  year={{2020}}\n}}\n")
        
        # If no bibitems found, create a default bibliography
        if not bib_entries:
            bib_entries = [
                """@book{putnam2000,
  title={Bowling Alone: The Collapse and Revival of American Community},
  author={Putnam, Robert D.},
  year={2000},
  publisher={Simon and Schuster}
}""",
                """@article{inglehart2018,
  title={Cultural Evolution: People's Motivations are Changing, and Reshaping the World},
  author={Inglehart, Ronald},
  year={2018},
  publisher={Cambridge University Press}
}""",
                """@book{norris2019,
  title={Cultural Backlash: Trump, Brexit, and Authoritarian Populism},
  author={Norris, Pippa and Inglehart, Ronald},
  year={2019},
  publisher={Cambridge University Press}
}""",
                """@article{diener1999,
  title={Subjective Well-being: Three Decades of Progress},
  author={Diener, Ed and Suh, Eunkook M. and Lucas, Richard E. and Smith, Heidi L.},
  journal={Psychological Bulletin},
  volume={125},
  number={2},
  pages={276--302},
  year={1999}
}""",
                """@book{haidt2012,
  title={The Righteous Mind: Why Good People are Divided by Politics and Religion},
  author={Haidt, Jonathan},
  year={2012},
  publisher={Pantheon Books}
}"""
            ]
        
        return "\n\n".join(bib_entries)
    
    def try_compile_latex(self, latex_path: Path):
        """Try to compile LaTeX file to PDF with BibTeX. Returns (success: bool, error_message: str)."""
        try:
            subprocess.run(['pdflatex', '--version'], capture_output=True, check=True)
        except Exception as e:
            return False, "pdflatex not found. Skipping PDF compilation."
        # Get the base name without extension
        base_name = latex_path.stem
        # LaTeX compilation sequence for BibTeX
        commands = [
            ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"],
            ["bibtex", base_name],
            ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"],
            ["pdflatex", "-interaction=nonstopmode", f"{base_name}.tex"],
        ]
        for i, command in enumerate(commands):
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=self.output_path,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                error_msg = result.stderr[-2000:] if result.stderr else result.stdout[-2000:]
                return False, f"Command {' '.join(command)} failed: {error_msg}"
        pdf_path = latex_path.with_suffix('.pdf')
        if pdf_path.exists():
            return True, ""
        else:
            return False, "PDF generation failed."
    
    async def debug_latex_with_agent(self, latex_path: Path, error_message: str) -> bool:
        """Use Claude Code agent to debug and fix LaTeX compilation errors."""
        prompt = f"""You are tasked with debugging and fixing LaTeX compilation errors for an academic paper.

The LaTeX file is located at: {latex_path}
The bibliography file is at: {self.output_path / "dynamic_references.bib"}

Compilation error:
{error_message}

Your task is to:
1. Read and analyze the LaTeX file
2. Identify the compilation errors
3. Fix the LaTeX syntax, package conflicts, or other issues
4. Ensure the paper compiles successfully to PDF
5. Make minimal changes to preserve the content and structure
6. Test the compilation after making fixes

Please debug and fix all compilation errors automatically."""

        formatter.print("Starting agent-based LaTeX debugging...", MessageType.PROGRESS)
        
        # Configure Claude Code SDK options
        options = ClaudeCodeOptions(
            max_turns=8,
            system_prompt="You are a LaTeX expert specializing in academic paper compilation. You have access to read and write files, execute bash commands, and use all available tools.",
            cwd=Path.cwd(),
            allowed_tools=["Read", "Write", "Bash", "Edit", "MultiEdit"],
            permission_mode="acceptEdits"
        )
        
        try:
            async for message in query(prompt=prompt, options=options):
                # Format agent messages similar to generate_and_execute_analysis.py
                message_str = str(message)
                
                if "ResultMessage(" in message_str:
                    formatter.print("LaTeX debugging completed", MessageType.SUCCESS)
                elif "SystemMessage(" in message_str:
                    # Extract meaningful system messages
                    content_found = False
                    
                    # Try multiple patterns to extract content
                    import re
                    patterns = [
                        (r"content='([^']*)'", 1),  # content='...'
                        (r'content="([^"]*)"', 1),  # content="..."
                        (r"data={'[^']*':\s*'([^']*)'", 1),  # data={'type': '...'}
                        (r"subtype='([^']*)'", 1),  # subtype='...'
                    ]
                    
                    for pattern, group in patterns:
                        match = re.search(pattern, message_str)
                        if match:
                            content = match.group(group)
                            if content and content.strip():
                                formatter.print(content, MessageType.SYSTEM)
                                content_found = True
                                break
                    
                    if not content_found:
                        formatter.print("System message received", MessageType.SYSTEM)
                else:
                    # Handle regular agent messages
                    if hasattr(message, 'content'):
                        if isinstance(message.content, list):
                            for block in message.content:
                                if hasattr(block, 'text'):
                                    text = block.text.strip()
                                    if text:
                                        formatter.print(text, MessageType.INFO)
                        else:
                            text = str(message.content).strip()
                            if text:
                                formatter.print(text, MessageType.INFO)

            # Test if the LaTeX file compiles now
            success, _ = self.try_compile_latex(latex_path)
            return success
            
        except Exception as e:
            formatter.print(f"Error in agent-based LaTeX debugging: {e}", MessageType.ERROR)
            return False

    def save_paper_with_agent_debugging(self):
        """Generate paper and use Claude Code agent for debugging if compilation fails."""
        formatter.print("Generating initial paper draft using LLM...", MessageType.PROGRESS)
        
        # Generate initial draft
        latex_content = self.generate_paper_content()
        latex_path = self.output_path / "dynamic_paper.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
            
        bib_content = self.extract_bibliography(latex_content)
        bib_path = self.output_path / "dynamic_references.bib"
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(bib_content)
            
        formatter.print(f"LaTeX paper generated: {latex_path}", MessageType.SUCCESS)
        formatter.print(f"Bibliography file created: {bib_path}", MessageType.SUCCESS)
        
        # Try initial compilation
        success, error_message = self.try_compile_latex(latex_path)
        if success:
            formatter.print(f"PDF generated successfully: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
            return latex_path
        else:
            formatter.print(f"LaTeX compilation failed. Starting agent-based debugging...", MessageType.WARNING)
            formatter.print(f"Compilation error: {error_message[:500]}...", MessageType.ERROR)
            
            # Use agent to debug and fix
            try:
                debug_success = asyncio.run(self.debug_latex_with_agent(latex_path, error_message))
                if debug_success:
                    formatter.print(f"PDF generated successfully after debugging: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
                    return latex_path
                else:
                    formatter.print("Agent-based debugging could not resolve all compilation errors.", MessageType.ERROR)
                    return None
                    
            except Exception as e:
                formatter.print(f"Error during agent-based debugging: {e}", MessageType.ERROR)
                return None


if __name__ == "__main__":
    generator = DynamicPaperGenerator()
    generator.save_paper_with_agent_debugging()