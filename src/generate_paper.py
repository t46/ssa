"""Generate LaTeX paper from analysis results."""
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


class PaperGenerator:
    """Generate academic paper in LaTeX format based on analysis results."""
    
    def __init__(self):
        """Initialize paper generator with paths and data."""
        self.research_path = Path("spec/research.yaml")
        self.report_path = Path("outputs/analysis_report.md")
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
            # Fallback to original report if main one doesn't exist
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
        """Generate paper content using LLM, with optional compile error feedback."""
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
            prompt += f"""
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
- Include all figures listed above in appropriate sections using \includegraphics{{filename}} (without path)
- Reference figures appropriately in the text (e.g., "As shown in Figure 1...")
- Statistical notation (p-values, coefficients, R-squared)

CRITICAL REFERENCE REQUIREMENTS:
1. ONLY use figure files that actually exist: {', '.join(figure_names)}
2. DO NOT reference non-existent figures or create fictional figure names
3. Every \includegraphics{{}} must reference an actual file from the list above
4. Every \cite{{}} must have a corresponding entry in the bibliography
5. Every \ref{{}} must have a corresponding \label{{}} in the document
6. Every figure must have both \includegraphics and \label{{fig:...}} in the same figure environment
7. Figure labels should match their content (e.g., \label{{fig:hypothesis1}} for hypothesis_1_analysis.png)

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
   \begin{{figure}}[htbp]
   \centering
   \includegraphics[width=0.8\textwidth]{{hypothesis_1_analysis.png}}
   \caption{{Analysis results for Hypothesis 1 showing the relationship between digital engagement and social trust}}
   \label{{fig:hypothesis1}}
   \end{{figure}}

IMPORTANT WIDTH CONTROL REQUIREMENTS TO PREVENT OVERFLOW:
1. FIGURES:
   - Always use width=0.8\textwidth or smaller (never exceed 0.9\textwidth)
   - For small figures, consider width=0.6\textwidth
   - Use \centering to center figures
   - Example: \includegraphics[width=0.8\textwidth,keepaspectratio]{{filename.png}}

2. TABLES:
   - Use \resizebox{{\textwidth}}{{!}}{{...}} for wide tables
   - Alternative: use \footnotesize or \small for table text
   - Use tabularx package for auto-adjusting column widths
   - Example for wide tables:
     \begin{{table}}[htbp]
     \centering
     \footnotesize
     \resizebox{{\textwidth}}{{!}}{{%
     \begin{{tabular}}{{lccccc}}
     ... table content ...
     \end{{tabular}}
     }}
     \caption{{Table caption}}
     \label{{tab:example}}
     \end{{table}}

3. EQUATIONS:
   - Break long equations across multiple lines using align environment
   - Use \\\\ for line breaks in equations
   - Use proper indentation with &= for alignment
   - Example:
     \begin{{align}}
     Y_i &= \beta_0 + \sum_{{j=1}}^{{k}} \beta_j X_{{ij}} + \varepsilon_i \\\\
     &\quad + \text{{additional terms if needed}}
     \end{{align}}

4. GENERAL WIDTH MANAGEMENT:
   - Never let any element exceed page margins
   - Use \linewidth instead of \textwidth in nested environments
   - Consider landscape orientation for very wide tables: \begin{{landscape}}...\end{{landscape}}
   - Use \raggedright for text that might overflow"""
        
        # Use streaming for long requests
        response_text = ""
        stream = self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=24000,
            temperature=0.5,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Collect streamed response
        with stream as stream:
            for text in stream.text_stream:
                response_text += text
        
        latex_content = response_text
        # Clean up the content if needed
        if "```latex" in latex_content:
            latex_content = latex_content.split("```latex")[1].split("```", 1)[0]
        elif "```" in latex_content:
            latex_content = latex_content.split("```", 1)[1].split("```", 1)[0]
        return latex_content
    
    def generate_paper_outline(self) -> Dict[str, Any]:
        """Generate the overall structure and outline of the paper."""
        # Load analysis results
        analysis_results = self.load_analysis_results()
        
        # Get list of actual figure files
        figure_files = sorted(self.output_path.glob("*.png"))
        figure_names = [f.name for f in figure_files]
        
        # Create a mapping of figure descriptions from the analysis results
        figure_mapping = self._extract_figure_descriptions(analysis_results)
        
        prompt = f"""Based on the research configuration and analysis results, generate a detailed outline for an academic paper.

Research Configuration:
{json.dumps(self.research_config, indent=2)}

Analysis Results:
{analysis_results}

Available Figures:
{json.dumps(figure_names, indent=2)}

Figure Descriptions:
{json.dumps(figure_mapping, indent=2)}

Please generate a comprehensive outline that includes:

1. **Title**: A compelling title that reflects all research topics
2. **Abstract Structure**: Key points to cover (150-200 words)
3. **Section Outlines**: For each section, provide:
   - Main objectives
   - Key points to cover
   - Figures/tables to include
   - Approximate word count

Sections to outline:
- Introduction
- Literature Review
- Methods
- Results (with subsections for each hypothesis)
- Discussion
- Limitations
- Conclusion

Return the outline as a structured JSON format with clear section breakdowns, key arguments, and figure placements.
"""

        # Use streaming for long requests
        response_text = ""
        stream = self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=16000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Collect streamed response
        with stream as stream:
            for text in stream.text_stream:
                response_text += text
        
        outline_text = response_text
        
        # Try to parse as JSON, fallback to text structure
        try:
            # Look for JSON in the response
            import re
            json_match = re.search(r'\{.*\}', outline_text, re.DOTALL)
            if json_match:
                outline = json.loads(json_match.group())
            else:
                # Create a structured outline from text
                outline = {
                    "title": "Multi-faceted Analysis of Digital Engagement and Social Trust",
                    "outline_text": outline_text
                }
        except:
            outline = {
                "title": "Multi-faceted Analysis of Digital Engagement and Social Trust", 
                "outline_text": outline_text
            }
        
        return outline
    
    def generate_section_content(self, section_name: str, outline: Dict[str, Any]) -> str:
        """Generate detailed content for a specific section based on the outline."""
        # Load analysis results
        analysis_results = self.load_analysis_results()
        
        # Get list of actual figure files
        figure_files = sorted(self.output_path.glob("*.png"))
        figure_names = [f.name for f in figure_files]
        
        # Create a mapping of figure descriptions from the analysis results
        figure_mapping = self._extract_figure_descriptions(analysis_results)
        
        prompt = f"""Generate detailed LaTeX content for the {section_name} section of an academic paper.

Paper Outline:
{json.dumps(outline, indent=2)}

Research Configuration:
{json.dumps(self.research_config, indent=2)}

Analysis Results:
{analysis_results}

Available Figures:
{json.dumps(figure_names, indent=2)}

Figure Descriptions:
{json.dumps(figure_mapping, indent=2)}

Requirements for {section_name} section:
1. Write in professional academic style
2. Include proper citations (Author, Year format)
3. Reference figures and tables appropriately
4. Use LaTeX formatting (subsections, equations, etc.)
5. Ensure content flows logically from the outline
6. Be comprehensive and detailed (aim for substantial content)
7. PREVENT WIDTH OVERFLOW:
   - Figures: use width=0.8\textwidth or smaller
   - Tables: use \resizebox{{\textwidth}}{{!}}{{...}} for wide tables or \footnotesize
   - Equations: break long equations with align environment and \\
   - Never exceed page margins

For Results section: Include statistical notation, p-values, coefficients, R-squared values
For Methods section: Describe data collection, variables, analytical approach
For Discussion section: Interpret findings, compare with literature, discuss implications

Generate ONLY the LaTeX content for this section (no \\documentclass, \\begin{{document}}, etc.).
Include appropriate LaTeX section commands (\\section{{}}, \\subsection{{}}).
"""

        # Use streaming for long requests
        response_text = ""
        stream = self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=20000,
            temperature=0.4,
            messages=[{
                "role": "user", 
                "content": prompt
            }]
        )
        
        # Collect streamed response
        with stream as stream:
            for text in stream.text_stream:
                response_text += text
        
        section_content = response_text
            
        # Clean up content if needed
        if "```latex" in section_content:
            section_content = section_content.split("```latex")[1].split("```", 1)[0]
        elif "```" in section_content:
            section_content = section_content.split("```", 1)[1].split("```", 1)[0]
            
        return section_content.strip()
    
    def refine_complete_paper(self, sections: Dict[str, str], outline: Dict[str, Any]) -> str:
        """Refine and integrate all sections into a complete, coherent paper."""
        # Load analysis results for context
        analysis_results = self.load_analysis_results()
        
        # Get list of actual figure files
        figure_files = sorted(self.output_path.glob("*.png"))
        figure_names = [f.name for f in figure_files]
        
        prompt = f"""Refine and integrate the following sections into a complete, coherent academic paper in LaTeX format.

Original Outline:
{json.dumps(outline, indent=2)}

Generated Sections:
{json.dumps(sections, indent=2)}

Research Configuration:
{json.dumps(self.research_config, indent=2)}

Available Figures:
{json.dumps(figure_names, indent=2)}

Your task:
1. Create a complete LaTeX document with proper preamble and document structure
2. Integrate all sections smoothly with proper transitions
3. Ensure consistent terminology and citation style throughout
4. Check that all figures are properly referenced and placed
5. Add any missing elements (abstract, references, etc.)
6. Ensure proper LaTeX formatting and compilation compatibility
7. Maintain academic writing standards and logical flow
8. ENSURE WIDTH CONTROL throughout the document:
   - All figures must use width=0.8\textwidth or smaller
   - Wide tables must use \resizebox{{\textwidth}}{{!}}{{...}} or smaller fonts
   - Long equations must be broken across lines with align environment
   - No element should exceed page margins
   - Add necessary packages: graphicx, tabularx, amsmath, pdflscape for layout control

The final paper should include:
- Complete LaTeX document structure (\\documentclass to \\end{{document}})
- Title page with appropriate title from outline
- Abstract (150-200 words)
- All sections integrated coherently  
- Proper figure placements with captions and labels
- Consistent academic style
- Bibliography section

Generate the complete, publication-ready LaTeX document.
"""

        # Use streaming for long requests
        response_text = ""
        stream = self.client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=32000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        # Collect streamed response
        with stream as stream:
            for text in stream.text_stream:
                response_text += text
        
        refined_content = response_text
            
        # Clean up content if needed
        if "```latex" in refined_content:
            refined_content = refined_content.split("```latex")[1].split("```", 1)[0]
        elif "```" in refined_content:
            refined_content = refined_content.split("```", 1)[1].split("```", 1)[0]
            
        return refined_content.strip()
    
    def check_reference_issues(self, latex_path: Path) -> Dict[str, List[str]]:
        """Check for reference issues in LaTeX file (missing figures, citations, etc.)."""
        issues = {
            "missing_figures": [],
            "missing_citations": [],
            "unused_figures": [],
            "reference_warnings": []
        }
        
        try:
            with open(latex_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Get list of actual figure files
            figure_files = set(f.name for f in self.output_path.glob("*.png"))
            
            # Find all \includegraphics references
            import re
            graphics_matches = re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', content)
            referenced_figures = set(graphics_matches)
            
            # Find missing figures
            for fig in referenced_figures:
                if fig not in figure_files:
                    issues["missing_figures"].append(fig)
            
            # Find unused figures
            for fig in figure_files:
                if fig not in referenced_figures:
                    issues["unused_figures"].append(fig)
            
            # Find all citations
            citation_matches = re.findall(r'\\cite\{([^}]+)\}', content)
            all_citations = set()
            for citation_group in citation_matches:
                # Handle multiple citations like \cite{ref1,ref2,ref3}
                refs = [ref.strip() for ref in citation_group.split(',')]
                all_citations.update(refs)
            
            # Check if bibliography exists
            bib_path = self.output_path / "references.bib"
            available_refs = set()
            if bib_path.exists():
                with open(bib_path, 'r', encoding='utf-8') as f:
                    bib_content = f.read()
                # Find all @article{key}, @book{key}, etc.
                ref_matches = re.findall(r'@\w+\{([^,\s]+)', bib_content)
                available_refs = set(ref_matches)
            
            # Find missing citations
            for citation in all_citations:
                if citation not in available_refs:
                    issues["missing_citations"].append(citation)
            
            # Check for undefined references from previous compilation
            log_path = latex_path.with_suffix('.log')
            if log_path.exists():
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # Look for reference warnings
                ref_warnings = re.findall(r'LaTeX Warning: Reference `([^\']+)\' on page \d+ undefined', log_content)
                cite_warnings = re.findall(r'LaTeX Warning: Citation `([^\']+)\' on page \d+ undefined', log_content)
                
                issues["reference_warnings"].extend(ref_warnings)
                issues["missing_citations"].extend(cite_warnings)
            
            # Remove duplicates
            for key in issues:
                issues[key] = list(set(issues[key]))
            
            return issues
            
        except Exception as e:
            formatter.print(f"Error checking references: {e}", MessageType.ERROR)
            return issues

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
        
        final_error_message = ""
        critical_failure = False
        reference_warnings = []
        
        for i, command in enumerate(commands):
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=self.output_path,
                capture_output=True,
                text=True
            )
            
            # Collect reference warnings from output
            if result.stdout:
                import re
                ref_warnings = re.findall(r'LaTeX Warning: Reference `([^\']+)\' on page \d+ undefined', result.stdout)
                cite_warnings = re.findall(r'LaTeX Warning: Citation `([^\']+)\' on page \d+ undefined', result.stdout)
                reference_warnings.extend(ref_warnings + cite_warnings)
            
            if result.returncode != 0:
                error_msg = result.stderr[-1000:] if result.stderr else result.stdout[-1000:]
                final_error_message += f"Command {' '.join(command)} failed: {error_msg}\n"
                
                # BibTeX failure is often non-critical (especially with few references)
                if command[0] == "bibtex":
                    print(f"Warning: BibTeX failed, but continuing compilation...")
                    continue
                
                # First pdflatex failure is critical
                if i == 0:
                    critical_failure = True
                    break
                
                # For later pdflatex runs, check if PDF was already created
                pdf_path = latex_path.with_suffix('.pdf')
                if not pdf_path.exists():
                    critical_failure = True
                    break
        
        # Final success check: PDF must exist
        pdf_path = latex_path.with_suffix('.pdf')
        if pdf_path.exists():
            # Report reference warnings even if compilation succeeded
            if reference_warnings:
                unique_warnings = list(set(reference_warnings))
                warning_msg = f"PDF generated but with undefined references: {unique_warnings}"
                formatter.print(warning_msg, MessageType.WARNING)
                # Return warnings in error message for agent to fix
                return True, f"REFERENCE_WARNINGS: {warning_msg}"
            elif final_error_message:
                print(f"Warning: PDF generated with some non-critical errors: {final_error_message[:500]}")
            return True, ""
        else:
            return False, final_error_message or "PDF generation failed."
    
    async def debug_latex_with_agent(self, latex_path: Path, error_message: str) -> bool:
        """Use Claude Code agent to debug and fix LaTeX compilation errors and reference issues."""
        
        # Check for reference issues first
        reference_issues = self.check_reference_issues(latex_path)
        
        # Get list of actual figure files
        figure_files = sorted(self.output_path.glob("*.png"))
        figure_names = [f.name for f in figure_files]
        
        # Build comprehensive issue report
        issue_summary = []
        if reference_issues["missing_figures"]:
            issue_summary.append(f"Missing figure files: {reference_issues['missing_figures']}")
        if reference_issues["unused_figures"]:
            issue_summary.append(f"Unused figure files available: {reference_issues['unused_figures']}")
        if reference_issues["missing_citations"]:
            issue_summary.append(f"Missing bibliography entries: {reference_issues['missing_citations']}")
        if reference_issues["reference_warnings"]:
            issue_summary.append(f"Undefined references: {reference_issues['reference_warnings']}")
        
        prompt = f"""You are tasked with debugging and fixing LaTeX compilation errors and reference issues for an academic paper.

The LaTeX file is located at: {latex_path}
The bibliography file is at: {self.output_path / "references.bib"}

AVAILABLE FIGURE FILES:
{json.dumps(figure_names, indent=2)}

COMPILATION ERROR:
{error_message}

REFERENCE ISSUES DETECTED:
{chr(10).join(issue_summary) if issue_summary else "No reference issues detected"}

Your task is to:
1. Read and analyze the LaTeX file and bibliography file
2. Fix LaTeX syntax, package conflicts, or other compilation issues
3. CRITICAL: Fix all reference issues:
   - Replace missing figure references with existing figure files
   - Remove references to non-existent figures or replace with available ones
   - Add missing bibliography entries for cited references
   - Remove citations to non-existent references or add proper bibliography entries
   - Ensure all \\ref{{}} and \\cite{{}} commands point to existing targets

4. REFERENCE FIXING GUIDELINES:
   - For missing figures: Replace with similar available figures from the list above
   - For missing citations: Either remove the citation or add a proper bibliography entry
   - Use only the figure files that actually exist: {', '.join(figure_names)}
   - Ensure all figure labels (\\label{{fig:...}}) match figure references (\\ref{{fig:...}})
   - Ensure all citation keys in \\cite{{}} exist in references.bib

5. Make minimal changes to preserve content and structure
6. Test the compilation after making fixes
7. Verify that all references resolve correctly

The goal is a paper that compiles successfully with NO undefined references or missing figures."""

        formatter.print("Starting agent-based LaTeX debugging with reference fixing...", MessageType.PROGRESS)
        
        # Show detected issues to user
        if issue_summary:
            formatter.print("Reference issues detected:", MessageType.WARNING)
            for issue in issue_summary:
                formatter.print(f"  - {issue}", MessageType.WARNING)
        
        # Configure Claude Code SDK options
        options = ClaudeCodeOptions(
            max_turns=15,  # Increased for reference fixing
            system_prompt="You are a LaTeX expert specializing in academic paper compilation and reference management. You have access to read and write files, execute bash commands, and use all available tools. Pay special attention to fixing undefined references and missing figures.",
            cwd=Path.cwd(),
            allowed_tools=["Read", "Write", "Bash", "Edit", "MultiEdit"],
            permission_mode="acceptEdits"
        )
        
        try:
            async for message in query(prompt=prompt, options=options):
                # Format agent messages similar to generate_and_execute_analysis.py
                message_str = str(message)
                
                if "ResultMessage(" in message_str:
                    formatter.print("LaTeX debugging and reference fixing completed", MessageType.SUCCESS)
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

            # Test if the LaTeX file compiles now and check references
            success, _ = self.try_compile_latex(latex_path)
            
            if success:
                # Double-check references after fixing
                final_issues = self.check_reference_issues(latex_path)
                remaining_issues = []
                for issue_type, issues in final_issues.items():
                    if issues:
                        remaining_issues.extend(issues)
                
                if remaining_issues:
                    formatter.print(f"Warning: Some reference issues may remain: {remaining_issues}", MessageType.WARNING)
                else:
                    formatter.print("All reference issues resolved successfully", MessageType.SUCCESS)
            
            return success
            
        except Exception as e:
            formatter.print(f"Error in agent-based LaTeX debugging: {e}", MessageType.ERROR)
            return False

    def cleanup_latex_file(self, latex_path: Path) -> bool:
        """Clean up common LaTeX formatting issues and prevent width overflow."""
        try:
            with open(latex_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Remove literal \n characters (should be actual newlines)
            content = content.replace('\\n', '\n')
            
            # Fix double backslashes that aren't meant to be line breaks
            content = re.sub(r'\\\\(?![a-zA-Z*])', r'\\', content)
            
            # Remove trailing \n at the very end
            content = content.rstrip('\n') + '\n'
            
            # Ensure proper line endings
            content = content.replace('\r\n', '\n').replace('\r', '\n')
            
            # FIX WIDTH OVERFLOW ISSUES:
            
            # 1. Fix figures without proper width constraints
            content = re.sub(
                r'\\includegraphics\{([^}]+)\}',
                r'\\includegraphics[width=0.8\\textwidth,keepaspectratio]{\1}',
                content
            )
            # Don't double-apply width settings
            content = re.sub(
                r'\\includegraphics\[[^\]]*width[^\]]*\]\[width=0\.8\\textwidth,keepaspectratio\]',
                lambda m: m.group(0).replace('[width=0.8\\textwidth,keepaspectratio]', ''),
                content
            )
            
            # 2. Wrap wide tables with resizebox if not already wrapped
            # Look for tabular environments not inside resizebox
            def wrap_wide_tables(match):
                table_content = match.group(0)
                if 'resizebox' not in table_content:
                    # Always apply resizebox for width control, regardless of footnotesize
                    if 'footnotesize' not in table_content:
                        # Add footnotesize if not present
                        table_content = table_content.replace(
                            '\\begin{tabular}', 
                            '\\footnotesize\n\\begin{tabular}'
                        )
                    # Wrap entire tabular with resizebox
                    return table_content.replace(
                        '\\begin{tabular}', 
                        '\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}'
                    ).replace(
                        '\\end{tabular}',
                        '\\end{tabular}\n}'
                    )
                return table_content
            
            content = re.sub(
                r'\\begin\{table\}.*?\\end\{table\}',
                wrap_wide_tables,
                content,
                flags=re.DOTALL
            )
            
            # 3. Break long equations that might overflow
            # This is a simple heuristic - look for very long equation content
            def fix_long_equations(match):
                eq_content = match.group(1)
                # If equation line is very long, suggest breaking it
                if len(eq_content) > 80 and '\\\\' not in eq_content and 'align' not in eq_content:
                    # Try to break at + or = signs
                    if ' + ' in eq_content:
                        eq_content = eq_content.replace(' + ', ' \\\\\n    &+ ')
                    elif ' = ' in eq_content and eq_content.count(' = ') == 1:
                        eq_content = eq_content.replace(' = ', ' &= ')
                return f'\\begin{{equation}}\n{eq_content}\n\\end{{equation}}'
            
            content = re.sub(
                r'\\begin\{equation\}\s*\n?(.*?)\n?\s*\\end\{equation\}',
                fix_long_equations,
                content,
                flags=re.DOTALL
            )
            
            # 4. Ensure necessary packages are included
            if '\\usepackage{graphicx}' not in content and '\\includegraphics' in content:
                content = content.replace(
                    '\\begin{document}',
                    '\\usepackage{graphicx}\n\\begin{document}'
                )
            
            if '\\usepackage{amsmath}' not in content and ('\\begin{align}' in content or '\\begin{equation}' in content):
                content = content.replace(
                    '\\begin{document}',
                    '\\usepackage{amsmath}\n\\begin{document}'
                )
            
            if '\\usepackage{tabularx}' not in content and '\\begin{tabularx}' in content:
                content = content.replace(
                    '\\begin{document}',
                    '\\usepackage{tabularx}\n\\begin{document}'
                )
            
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            formatter.print("Applied width overflow prevention fixes", MessageType.INFO)
            return True
        except Exception as e:
            formatter.print(f"Error cleaning LaTeX file: {e}", MessageType.ERROR)
            return False

    def save_paper_with_agent_debugging(self):
        """Generate paper and use Claude Code agent for debugging if compilation fails."""
        formatter.print("Generating initial paper draft using LLM...", MessageType.PROGRESS)
        
        # Generate initial draft
        latex_content = self.generate_paper_content()
        latex_path = self.output_path / "paper.tex"
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)
        
        # Clean up the LaTeX file
        self.cleanup_latex_file(latex_path)
            
        bib_content = self.extract_bibliography(latex_content)
        bib_path = self.output_path / "references.bib"
        with open(bib_path, 'w', encoding='utf-8') as f:
            f.write(bib_content)
            
        formatter.print(f"LaTeX paper generated: {latex_path}", MessageType.SUCCESS)
        formatter.print(f"Bibliography file created: {bib_path}", MessageType.SUCCESS)
        
        # Try initial compilation
        success, error_message = self.try_compile_latex(latex_path)
        if success and not error_message.startswith("REFERENCE_WARNINGS"):
            formatter.print(f"PDF generated successfully: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
            return latex_path
        else:
            if error_message.startswith("REFERENCE_WARNINGS"):
                formatter.print("PDF generated but reference issues detected. Starting agent-based fixing...", MessageType.WARNING)
            else:
                formatter.print(f"LaTeX compilation failed. Starting agent-based debugging...", MessageType.WARNING)
                
            formatter.print(f"Issues to fix: {error_message[:500]}...", MessageType.ERROR)
            
            # Use agent to debug and fix
            try:
                debug_success = asyncio.run(self.debug_latex_with_agent(latex_path, error_message))
                if debug_success:
                    # Try compilation again after debugging
                    success, final_error = self.try_compile_latex(latex_path)
                    if success and not final_error.startswith("REFERENCE_WARNINGS"):
                        formatter.print(f"PDF generated successfully after debugging: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
                        return latex_path
                    else:
                        if final_error.startswith("REFERENCE_WARNINGS"):
                            formatter.print("Agent reduced but did not eliminate all reference issues.", MessageType.WARNING)
                        else:
                            formatter.print("Agent fixed some issues but compilation still failed.", MessageType.WARNING)
                        # Check if PDF exists anyway (sometimes it's generated despite errors)
                        pdf_path = latex_path.with_suffix('.pdf')
                        if pdf_path.exists():
                            formatter.print(f"PDF was generated despite issues: {pdf_path}", MessageType.SUCCESS)
                            return latex_path
                        return None
                else:
                    formatter.print("Agent-based debugging could not resolve all issues.", MessageType.ERROR)
                    # Check if PDF exists anyway
                    pdf_path = latex_path.with_suffix('.pdf')
                    if pdf_path.exists():
                        formatter.print(f"PDF was generated despite debugging failure: {pdf_path}", MessageType.SUCCESS)
                        return latex_path
                    return None
                    
            except Exception as e:
                formatter.print(f"Error during agent-based debugging: {e}", MessageType.ERROR)
                # Check if PDF exists anyway
                pdf_path = latex_path.with_suffix('.pdf')
                if pdf_path.exists():
                    formatter.print(f"PDF was generated despite error: {pdf_path}", MessageType.SUCCESS)
                    return latex_path
                return None

    def save_paper_with_multi_step_generation(self):
        """Generate paper using 3-step process: outline → sections → refinement."""
        formatter.print("Starting 3-step paper generation process...", MessageType.PROGRESS)
        
        # Step 1: Generate outline
        formatter.print("Step 1: Generating paper outline...", MessageType.PROGRESS)
        outline = self.generate_paper_outline()
        
        # Save outline for debugging
        outline_path = self.output_path / "paper_outline.json"
        with open(outline_path, 'w', encoding='utf-8') as f:
            json.dump(outline, f, indent=2, ensure_ascii=False)
        formatter.print(f"Outline saved: {outline_path}", MessageType.SUCCESS)
        
        # Step 2: Generate individual sections
        formatter.print("Step 2: Generating detailed sections...", MessageType.PROGRESS)
        sections = {}
        section_names = [
            "Introduction",
            "Literature Review", 
            "Methods",
            "Results",
            "Discussion",
            "Limitations",
            "Conclusion"
        ]
        
        for section_name in section_names:
            formatter.print(f"  Generating {section_name}...", MessageType.INFO)
            try:
                section_content = self.generate_section_content(section_name, outline)
                sections[section_name] = section_content
                
                # Save individual section for debugging
                section_file = self.output_path / f"section_{section_name.lower().replace(' ', '_')}.tex"
                with open(section_file, 'w', encoding='utf-8') as f:
                    f.write(section_content)
                    
            except Exception as e:
                formatter.print(f"Error generating {section_name}: {e}", MessageType.ERROR)
                sections[section_name] = f"% Error generating {section_name}: {e}"
        
        formatter.print(f"Generated {len(sections)} sections", MessageType.SUCCESS)
        
        # Step 3: Refine and integrate complete paper
        formatter.print("Step 3: Refining and integrating complete paper...", MessageType.PROGRESS)
        try:
            refined_latex = self.refine_complete_paper(sections, outline)
            
            # Save refined paper
            latex_path = self.output_path / "paper.tex"
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(refined_latex)
            
            # Clean up the LaTeX file
            self.cleanup_latex_file(latex_path)
            
            # Generate bibliography
            bib_content = self.extract_bibliography(refined_latex)
            bib_path = self.output_path / "references.bib"
            with open(bib_path, 'w', encoding='utf-8') as f:
                f.write(bib_content)
                
            formatter.print(f"Multi-step LaTeX paper generated: {latex_path}", MessageType.SUCCESS)
            formatter.print(f"Bibliography file created: {bib_path}", MessageType.SUCCESS)
            
            # Try compilation
            success, error_message = self.try_compile_latex(latex_path)
            if success and not error_message.startswith("REFERENCE_WARNINGS"):
                formatter.print(f"PDF generated successfully: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
                return latex_path
            else:
                if error_message.startswith("REFERENCE_WARNINGS"):
                    formatter.print("PDF generated but reference issues detected. Starting agent-based fixing...", MessageType.WARNING)
                else:
                    formatter.print(f"LaTeX compilation failed. Starting agent-based debugging...", MessageType.WARNING)
                    
                formatter.print(f"Issues to fix: {error_message[:500]}...", MessageType.ERROR)
                
                # Use agent to debug and fix
                try:
                    debug_success = asyncio.run(self.debug_latex_with_agent(latex_path, error_message))
                    if debug_success:
                        # Try compilation again after debugging
                        success, final_error = self.try_compile_latex(latex_path)
                        if success and not final_error.startswith("REFERENCE_WARNINGS"):
                            formatter.print(f"PDF generated successfully after debugging: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
                            return latex_path
                        else:
                            if final_error.startswith("REFERENCE_WARNINGS"):
                                formatter.print("Agent reduced but did not eliminate all reference issues.", MessageType.WARNING)
                            else:
                                formatter.print("Agent fixed some issues but compilation still failed.", MessageType.WARNING)
                            # Check if PDF exists anyway (sometimes it's generated despite errors)
                            pdf_path = latex_path.with_suffix('.pdf')
                            if pdf_path.exists():
                                formatter.print(f"PDF was generated despite issues: {pdf_path}", MessageType.SUCCESS)
                                return latex_path
                            return None
                    else:
                        formatter.print("Agent-based debugging could not resolve all issues.", MessageType.ERROR)
                        # Check if PDF exists anyway
                        pdf_path = latex_path.with_suffix('.pdf')
                        if pdf_path.exists():
                            formatter.print(f"PDF was generated despite debugging failure: {pdf_path}", MessageType.SUCCESS)
                            return latex_path
                        return None
                        
                except Exception as e:
                    formatter.print(f"Error during agent-based debugging: {e}", MessageType.ERROR)
                    # Check if PDF exists anyway
                    pdf_path = latex_path.with_suffix('.pdf')
                    if pdf_path.exists():
                        formatter.print(f"PDF was generated despite error: {pdf_path}", MessageType.SUCCESS)
                        return latex_path
                    return None
                    
        except Exception as e:
            formatter.print(f"Error in Step 3 (refinement): {e}", MessageType.ERROR)
            return None


if __name__ == "__main__":
    generator = PaperGenerator()
    generator.save_paper_with_multi_step_generation()