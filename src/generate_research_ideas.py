"""Generate research ideas based on WVS questionnaire."""
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import anthropic
import os
import re
from tools.semantic_search import SemanticSearcher
from tools.search_papers import search_papers, Paper


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


def generate_search_queries(proposal: Dict[str, Any]) -> List[str]:
    """Generate search queries from research proposal for literature search."""
    queries = []
    
    # Main query based on title and objective
    main_concepts = proposal.get('title', '').replace('"', '').split()[:5]
    if len(main_concepts) >= 2:
        queries.append(' '.join(main_concepts[:3]))
    
    # Query from theoretical background
    theoretical_bg = proposal.get('theoretical_background', '')
    if theoretical_bg:
        # Extract key theoretical terms
        theory_words = [word for word in theoretical_bg.split() 
                       if len(word) > 5 and word.lower() not in 
                       ['theory', 'theories', 'research', 'studies', 'analysis']]
        if theory_words:
            queries.append(' '.join(theory_words[:3]))
    
    # Query from dependent variables
    dependent_vars = proposal.get('variables', {}).get('dependent', [])
    if dependent_vars:
        # Use first dependent variable as main concept
        dep_var = dependent_vars[0].replace('_', ' ').replace('-', ' ')
        queries.append(dep_var)
    
    # Query from independent variables
    independent_vars = proposal.get('variables', {}).get('independent', [])
    if independent_vars and dependent_vars:
        indep_var = independent_vars[0].replace('_', ' ').replace('-', ' ')
        dep_var = dependent_vars[0].replace('_', ' ').replace('-', ' ')
        queries.append(f"{dep_var} {indep_var}")
    
    return queries[:3]  # Limit to top 3 queries


def reformulate_query(original_query: str) -> List[str]:
    """Reformulate query to broader terms if original search yields few results."""
    reformulated = []
    
    # Split into individual words and create simpler queries
    words = original_query.split()
    if len(words) > 1:
        # Try individual words
        for word in words:
            if len(word) > 4:  # Only use meaningful words
                reformulated.append(word)
        
        # Try pairs of words
        for i in range(len(words) - 1):
            reformulated.append(f"{words[i]} {words[i+1]}")
    
    # Add more general terms based on common research areas
    general_terms = [
        "social psychology",
        "behavioral economics", 
        "political psychology",
        "social behavior",
        "attitudes values",
        "cross-cultural psychology"
    ]
    
    # Select relevant general terms based on original query
    for term in general_terms:
        if any(word in original_query.lower() for word in term.split()):
            reformulated.append(term)
    
    return reformulated[:5]  # Limit reformulations


def search_relevant_papers(proposal: Dict[str, Any], min_papers: int = 5) -> List[Paper]:
    """Search for relevant papers using multiple query strategies."""
    all_papers = []
    used_queries = set()
    
    # Generate initial queries
    queries = generate_search_queries(proposal)
    
    for query in queries:
        if query in used_queries:
            continue
        used_queries.add(query)
        
        print(f"Searching papers with query: '{query}'")
        # Try with domain filter first
        papers = search_papers(query, limit=10, domain_filter=True)
        
        if papers:
            all_papers.extend(papers)
            print(f"Found {len(papers)} papers with Social Sciences filter")
        else:
            print(f"No papers found with Social Sciences filter, trying without filter...")
            # Try without domain filter
            papers = search_papers(query, limit=10, domain_filter=False)
            if papers:
                all_papers.extend(papers)
                print(f"Found {len(papers)} papers without domain filter")
            else:
                print(f"No papers found for '{query}', trying reformulation...")
                # Try reformulated queries
                reformulated = reformulate_query(query)
                for ref_query in reformulated:
                    if ref_query in used_queries:
                        continue
                    used_queries.add(ref_query)
                    
                    print(f"Trying reformulated query: '{ref_query}'")
                    papers = search_papers(ref_query, limit=8, domain_filter=False)
                    if papers:
                        all_papers.extend(papers)
                        print(f"Found {len(papers)} papers with reformulated query")
                        break  # Stop after first successful reformulation
        
        # Stop if we have enough papers
        if len(all_papers) >= min_papers:
            break
    
    # Remove duplicates based on DOI or title
    unique_papers = []
    seen_dois = set()
    seen_titles = set()
    
    for paper in all_papers:
        identifier = paper.doi if paper.doi else paper.title.lower().strip()
        if paper.doi and paper.doi in seen_dois:
            continue
        if paper.title.lower().strip() in seen_titles:
            continue
        
        if paper.doi:
            seen_dois.add(paper.doi)
        seen_titles.add(paper.title.lower().strip())
        unique_papers.append(paper)
    
    return unique_papers[:15]  # Limit to top 15 papers


def refine_research_proposal(proposal: Dict[str, Any], papers: List[Paper]) -> Dict[str, Any]:
    """Refine research proposal based on literature search results."""
    if not papers:
        print("No papers found for refinement, returning original proposal")
        return proposal
    
    # Prepare literature context
    literature_context = []
    for paper in papers[:8]:  # Use top 8 papers
        paper_info = f"Title: {paper.title}\n"
        if paper.authors:
            paper_info += f"Authors: {', '.join(paper.authors[:3])}\n"
        if paper.publication_year:
            paper_info += f"Year: {paper.publication_year}\n"
        if paper.abstract:
            paper_info += f"Abstract: {paper.abstract[:300]}...\n"
        literature_context.append(paper_info)
    
    # Create refinement prompt
    original_proposal_str = json.dumps(proposal, indent=2)
    literature_str = "\n\n".join(literature_context)
    
    prompt = f"""Based on the following literature search results, please refine and improve the original research proposal. The refinement should:

1. Incorporate insights from the recent literature
2. Refine hypotheses to be more specific and testable
3. Add relevant theoretical perspectives found in the literature
4. Suggest additional variables or methodological improvements
5. Identify research gaps that justify the study

Original Research Proposal:
{original_proposal_str}

Relevant Literature:
{literature_str}

Please provide the refined proposal in the same JSON format, with improved:
- Theoretical background (incorporating findings from the literature)
- More precise hypotheses
- Enhanced variables section
- Updated analytical approach if needed
- A new "literature_gaps" field explaining what gap this study fills

Return only the JSON object without additional text."""
    
    # Get refinement from LLM
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY_SSA"))
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=3000,
            temperature=0.3,
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )
        
        response_text = getattr(response, "content", None)
        if isinstance(response_text, list) and hasattr(response_text[0], "text"):
            response_text = response_text[0].text
        
        # Extract JSON from response
        json_match = re.search(r'\{[\s\S]*\}', response_text)
        if json_match:
            refined_proposal = json.loads(json_match.group())
            print(f"Successfully refined proposal: {refined_proposal.get('title', 'Unknown')}")
            return refined_proposal
        else:
            print("Could not parse refinement response, returning original proposal")
            return proposal
            
    except Exception as e:
        print(f"Error during proposal refinement: {e}")
        return proposal


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
    
    # Literature-informed refinement for each proposal
    print("\n=== Starting Literature Search and Proposal Refinement ===")
    refined_proposals = []
    for i, proposal in enumerate(proposals):
        print(f"\nProcessing proposal {i+1}: {proposal.get('title', 'Unknown')}")
        
        # Search relevant papers
        papers = search_relevant_papers(proposal)
        print(f"Found {len(papers)} relevant papers")
        
        # Refine proposal based on literature
        if papers:
            refined_proposal = refine_research_proposal(proposal, papers)
            # Add paper references to proposal
            refined_proposal['referenced_papers'] = [
                {
                    'title': paper.title,
                    'authors': paper.authors[:3],
                    'year': paper.publication_year,
                    'doi': paper.doi,
                    'citations': paper.cited_by_count
                }
                for paper in papers[:5]  # Top 5 papers
            ]
        else:
            print("No papers found, keeping original proposal")
            refined_proposal = proposal
        
        refined_proposals.append(refined_proposal)
    
    # Map variable descriptions to actual WVS codes
    searcher = SemanticSearcher()
    searcher.load_and_index_file(Path("data/code-maps/codebook_map.json"))
    refined_proposals = map_variables_to_questions(refined_proposals, searcher)
    
    return refined_proposals


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
                "note": "All analyses should apply population weights and handle missing values appropriately",
                "literature_informed": True,
                "search_strategy": "Multi-query search with reformulation, literature-based refinement"
            }
        }
        
        output_path = Path("spec/research.yaml")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            yaml.dump(output, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        print(f"\nResearch proposals saved to {output_path}")
        print("Proposals have been refined based on current literature search")
        return proposals
    except Exception as e:
        print(f"Error generating research proposals: {e}")
        raise


if __name__ == "__main__":
    save_research_proposals()