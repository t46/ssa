"""Generate research ideas based on WVS questionnaire."""
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import anthropic
import os
from tools.semantic_search import SemanticSearcher


def load_questionnaire_map() -> Dict[str, Any]:
    """Load questionnaire mapping from JSON file."""
    path = Path("data/code-maps/questionnaire_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_codebook_map() -> Dict[str, Any]:
    """Load codebook mapping from JSON file."""
    path = Path("data/code-maps/codebook_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pdf_content(pdf_path: Path, max_pages: int = 30) -> str:
    """Extract text content from PDF file."""
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        num_pages = min(len(pdf_reader.pages), max_pages)
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n"
    return text


def map_variables_to_questions(proposals: List[Dict[str, Any]], searcher: SemanticSearcher) -> List[Dict[str, Any]]:
    """Map variable descriptions to actual WVS variable codes."""
    for proposal in proposals:
        mapped_vars = {}
        for var_type in ['dependent', 'independent', 'controls', 'mediators']:
            if var_type in proposal.get('variables', {}):
                mapped_vars[var_type] = []
                for var_desc in proposal['variables'][var_type]:
                    results = searcher.search(var_desc, top_k=1)
                    if results:
                        var_code, label, score, _ = results[0]
                        if score > 0.3:  # Threshold for relevance
                            mapped_vars[var_type].append(var_code)
                            print(f"Mapped '{var_desc}' to {var_code} ({label})")
                        else:
                            print(f"No good match for '{var_desc}'")
        proposal['variables'] = mapped_vars
    return proposals


def generate_research_proposals() -> List[Dict[str, Any]]:
    """Generate research proposals based on WVS data using LLM."""
    # Load questionnaire PDF content
    pdf_path = Path("data/raw/F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf")
    pdf_content = extract_pdf_content(pdf_path)
    
    # Load codebook for variable mapping
    codebook = load_codebook_map()
    
    # Create prompt for LLM
    prompt = f"""Based on the World Values Survey Wave 7 questionnaire content below, generate 1 innovative research proposal for analyzing American values and attitudes.

For the proposal, provide:
1. A compelling title
2. Clear research objective 
3. Theoretical background (2-3 sentences referencing relevant theories)
4. One testable hypothesis
5. Variables needed:
   - Dependent variables (describe what you want to measure)
   - Independent variables (describe predictors)
   - Control variables (describe demographics/confounds)
   - Mediators (if applicable)
6. Analytical approach

Questionnaire excerpt:
{pdf_content[:5000]}

Format the output as a JSON array with the structure shown in the example.

Example structure:
[{{
  "id": 1,
  "title": "Title here",
  "objective": "Research objective",
  "theoretical_background": "Theory explanation",
  "hypotheses": ["H1", "H2", "H3"],
  "variables": {{
    "dependent": ["description of dependent variables"],
    "independent": ["description of independent variables"],
    "controls": ["age", "gender", "education", "income"]
  }},
  "analytical_approach": "Statistical method"
}}]
"""
    
    # Initialize Anthropic client
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY_SSA"))
    
    # Generate proposals using Claude
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=4000,
        temperature=0.7,
        messages=[{
            "role": "user",
            "content": prompt
        }]
    )
    
    # Parse the response
    import re
    response_text = getattr(response, "content", None)
    if isinstance(response_text, list) and hasattr(response_text[0], "text"):
        response_text = response_text[0].text
    elif isinstance(response_text, str):
        pass  # already a string
    else:
        raise ValueError("Unexpected response format from LLM.")

    json_match = re.search(r'\[\s*{[\s\S]*}\s*\]', response_text)
    if json_match:
        proposals = json.loads(json_match.group())
    else:
        print("Error: Could not parse LLM response as JSON.")
        raise ValueError("Could not parse LLM response as JSON.")
    
    # Map variable descriptions to actual WVS codes
    searcher = SemanticSearcher()
    searcher.load_and_index_file(Path("data/code-maps/codebook_map.json"))
    proposals = map_variables_to_questions(proposals, searcher)
    
    return proposals


def save_research_proposals():
    """Generate and save research proposals to YAML file."""
    try:
        proposals = generate_research_proposals()
        
        output = {
            "research_proposals": proposals,
            "metadata": {
                "dataset": "WVS Wave 7 (2017-2022)",
                "country": "United States (B_COUNTRY = 840)",
                "sample_weight": "W_WEIGHT",
                "note": "All analyses should apply population weights and handle missing values appropriately"
            }
        }
        
        output_path = Path("spec/research.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"Research proposals saved to {output_path}")
        return proposals
    except Exception as e:
        print(f"Error generating research proposals: {e}")
        raise


if __name__ == "__main__":
    save_research_proposals()