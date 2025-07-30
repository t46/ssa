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
        
        # Use streaming for long requests
        response_text = ""
        stream = self.client.messages.stream(
            model="claude-3-7-sonnet-20250219",
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
            model="claude-3-7-sonnet-20250219",
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

For Results section: Include statistical notation, p-values, coefficients, R-squared values
For Methods section: Describe data collection, variables, analytical approach
For Discussion section: Interpret findings, compare with literature, discuss implications

Generate ONLY the LaTeX content for this section (no \\documentclass, \\begin{{document}}, etc.).
Include appropriate LaTeX section commands (\\section{{}}, \\subsection{{}}).
"""

        # Use streaming for long requests
        response_text = ""
        stream = self.client.messages.stream(
            model="claude-3-7-sonnet-20250219",
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
            model="claude-3-7-sonnet-20250219",
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
        
        for i, command in enumerate(commands):
            print(f"Running: {' '.join(command)}")
            result = subprocess.run(
                command,
                cwd=self.output_path,
                capture_output=True,
                text=True
            )
            
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
            if final_error_message:
                print(f"Warning: PDF generated with some non-critical errors: {final_error_message[:500]}")
            return True, ""
        else:
            return False, final_error_message or "PDF generation failed."
    
    async def debug_latex_with_agent(self, latex_path: Path, error_message: str) -> bool:
        """Use Claude Code agent to debug and fix LaTeX compilation errors."""
        prompt = f"""You are tasked with debugging and fixing LaTeX compilation errors for an academic paper.

The LaTeX file is located at: {latex_path}
The bibliography file is at: {self.output_path / "references.bib"}

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
            max_turns=12, # Increased max_turns for more detailed debugging
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

    def cleanup_latex_file(self, latex_path: Path) -> bool:
        """Clean up common LaTeX formatting issues."""
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
            
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
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
                    # Try compilation again after debugging
                    success, _ = self.try_compile_latex(latex_path)
                    if success:
                        formatter.print(f"PDF generated successfully after debugging: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
                        return latex_path
                    else:
                        formatter.print("Agent fixed some issues but compilation still failed.", MessageType.WARNING)
                        # Check if PDF exists anyway (sometimes it's generated despite errors)
                        pdf_path = latex_path.with_suffix('.pdf')
                        if pdf_path.exists():
                            formatter.print(f"PDF was generated despite errors: {pdf_path}", MessageType.SUCCESS)
                            return latex_path
                        return None
                else:
                    formatter.print("Agent-based debugging could not resolve all compilation errors.", MessageType.ERROR)
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
                        # Try compilation again after debugging
                        success, _ = self.try_compile_latex(latex_path)
                        if success:
                            formatter.print(f"PDF generated successfully after debugging: {latex_path.with_suffix('.pdf')}", MessageType.SUCCESS)
                            return latex_path
                        else:
                            formatter.print("Agent fixed some issues but compilation still failed.", MessageType.WARNING)
                            # Check if PDF exists anyway (sometimes it's generated despite errors)
                            pdf_path = latex_path.with_suffix('.pdf')
                            if pdf_path.exists():
                                formatter.print(f"PDF was generated despite errors: {pdf_path}", MessageType.SUCCESS)
                                return latex_path
                            return None
                    else:
                        formatter.print("Agent-based debugging could not resolve all compilation errors.", MessageType.ERROR)
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