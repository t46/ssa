"""Generate LaTeX paper from analysis results - Dynamic version."""
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import anthropic
import os
import subprocess


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
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
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
    
    def generate_paper_content(self, compile_error_feedback: Optional[str] = None) -> str:
        """Generate paper content dynamically using LLM, with optional compile error feedback."""
        # Load analysis results
        analysis_results = self.load_analysis_results()
        
        # Create prompt
        prompt = f"""Generate a complete academic paper in LaTeX format based on the following research and analysis results.

Research Configuration:
{json.dumps(self.research_config, indent=2)}

Analysis Results:
{analysis_results}
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
- References to figures (stored as research1_*.png, research2_*.png, etc.)
- Statistical notation (p-values, coefficients, R-squared)

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
    
    def save_paper_with_feedback(self, max_attempts=3):
        """Generate and save the paper, retrying with LLM feedback if LaTeX fails to compile."""
        compile_error_feedback: Optional[str] = None
        for attempt in range(1, max_attempts + 1):
            print(f"\n[Attempt {attempt}] Generating paper content using LLM...")
            latex_content = self.generate_paper_content(compile_error_feedback)
            latex_path = self.output_path / "dynamic_paper.tex"
            with open(latex_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
            bib_content = self.extract_bibliography(latex_content)
            bib_path = self.output_path / "dynamic_references.bib"
            with open(bib_path, 'w', encoding='utf-8') as f:
                f.write(bib_content)
            print(f"LaTeX paper generated: {latex_path}")
            print(f"Bibliography file created: {bib_path}")
            success, error_message = self.try_compile_latex(latex_path)
            if success:
                print(f"PDF generated successfully: {latex_path.with_suffix('.pdf')}")
                return latex_path
            else:
                print(f"LaTeX compilation failed (attempt {attempt}):\n{error_message[:1000]}")
                compile_error_feedback = error_message
        print("\nFailed to generate a compilable LaTeX file after several attempts.")
        return None


if __name__ == "__main__":
    generator = DynamicPaperGenerator()
    generator.save_paper_with_feedback()