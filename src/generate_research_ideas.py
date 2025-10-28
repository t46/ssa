"""Generate research ideas based on WVS questionnaire."""
import json
import yaml
from pathlib import Path
from typing import List, Dict, Any
import PyPDF2
import os
import re
import random
from tools.semantic_search import SemanticSearcher
from tools.search_papers import search_papers, Paper
from llm_client import LLMClientFactory


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


def extract_pdf_content(pdf_path: Path, max_pages: int = 30, fallback_to_sampling: bool = True) -> str:
    """Extract text content from PDF file.
    
    Strategy: First attempt to extract all pages. If that fails (timeout, memory error, etc.),
    fall back to random sampling of pages.
    
    Args:
        pdf_path: Path to PDF file
        max_pages: Maximum number of pages to extract when using fallback sampling
        fallback_to_sampling: If True, fall back to random sampling on error
    
    Returns:
        Extracted text content
    """
    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        total_pages = len(pdf_reader.pages)
        
        # First, try to extract all pages
        try:
            print(f"Attempting to extract all {total_pages} pages from PDF")
            text = ""
            for page_num in range(total_pages):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text() + "\n"
            
            print(f"Successfully extracted all {total_pages} pages")
            return text
            
        except Exception as e:
            # If extraction fails and fallback is enabled, try random sampling
            if fallback_to_sampling:
                print(f"Warning: Failed to extract all pages ({type(e).__name__}: {e})")
                print(f"Falling back to random sampling of {max_pages} pages")
                
                try:
                    # Randomly sample pages
                    page_indices = sorted(random.sample(range(total_pages), min(max_pages, total_pages)))
                    
                    # Show page numbers (1-indexed for user readability)
                    sampled_page_nums = [p + 1 for p in page_indices]
                    if len(sampled_page_nums) > 10:
                        print(f"Sampled pages: {sampled_page_nums[:5]} ... {sampled_page_nums[-5:]}")
                    else:
                        print(f"Sampled pages: {sampled_page_nums}")
                    
                    text = ""
                    for page_num in page_indices:
                        page = pdf_reader.pages[page_num]
                        text += f"\n--- Page {page_num + 1} ---\n"
                        text += page.extract_text() + "\n"
                    
                    print(f"Successfully extracted {len(page_indices)} sampled pages")
                    return text
                    
                except Exception as sampling_error:
                    print(f"Error: Random sampling also failed: {sampling_error}")
                    raise
            else:
                # Re-raise the original error if fallback is disabled
                raise


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
    
    # Get refinement from LLM using factory
    try:
        factory = LLMClientFactory()
        client = factory.create_client()
        
        messages = [{"role": "user", "content": prompt}]
        response_text = client.generate_response(messages)
        
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


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = Path("config/llm_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_research_proposals() -> List[Dict[str, Any]]:
    """Generate research proposals based on WVS data using LLM."""
    # Load configuration
    config = load_config()
    num_ideas = config.get('research', {}).get('num_ideas', 1)
    
    # Load questionnaire PDF content
    pdf_path = Path("data/raw/F00010738-WVS-7_Master_Questionnaire_2017-2020_English.pdf")
    pdf_content = extract_pdf_content(pdf_path)
    
    # Load codebook for variable mapping
    codebook = load_codebook_map()
    
    # Create prompt for LLM
    diversity_instruction = ""
    if num_ideas > 1:
        diversity_instruction = f"""
IMPORTANT: Generate {num_ideas} DISTINCT and DIVERSE research proposals covering DIFFERENT topics and research areas. 
Each proposal should explore a UNIQUE aspect of American values and attitudes. 
Avoid generating similar proposals - ensure variety in:
- Research topics (e.g., political attitudes, social trust, family values, religious beliefs, economic attitudes, environmental concerns, etc.)
- Theoretical frameworks (different theories for each proposal)
- Methodological approaches
- Variables of interest

"""
    
    prompt = f"""Based on the World Values Survey Wave 7 questionnaire content below, generate {num_ideas} innovative research proposal(s) for analyzing American values and attitudes.

CRITICAL CONSTRAINT: Your research proposals MUST use ONLY variables and questions that actually exist in the WVS Wave 7 questionnaire provided below. Do NOT propose research on topics that are not covered in this specific questionnaire (e.g., if there are no questions about digital technology, internet use, or social media, do not propose research on those topics).

Carefully review the questionnaire content and base your proposals strictly on the questions that are actually present.
{diversity_instruction}
For each proposal, provide:
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

Full questionnaire content:
{pdf_content}

Format the output as a JSON array with the structure shown in the example below.
{"ENSURE EACH PROPOSAL HAS A DIFFERENT TOPIC AND RESEARCH FOCUS." if num_ideas > 1 else ""}

Example structure (based on ACTUAL WVS questions):
[{{
  "id": 1,
  "title": "Life Satisfaction and Religious Values in America",
  "objective": "Examine how religious importance relates to life satisfaction and happiness",
  "theoretical_background": "Religious coping theory suggests religious beliefs provide meaning and social support that enhance well-being...",
  "hypotheses": ["H1: Higher importance of religion predicts greater life satisfaction"],
  "variables": {{
    "dependent": ["life satisfaction", "feeling of happiness"],
    "independent": ["importance of religion in life", "frequency of prayer"],
    "controls": ["age", "gender", "education", "income"]
  }},
  "analytical_approach": "Multiple regression analysis with interaction effects"
}}{", {{...}}" if num_ideas > 1 else ""}]

NOTE: The example above uses ACTUAL questions from the WVS questionnaire (e.g., 'Important in life: Religion', 'Satisfaction with your life', 'Feeling of happiness'). 
Generate {num_ideas} {"DIFFERENT proposals on DISTINCT topics, each using ONLY variables that exist in the provided questionnaire" if num_ideas > 1 else "proposal using ONLY variables from the provided questionnaire"}.
"""
    
    # Initialize LLM client using factory
    factory = LLMClientFactory()
    client = factory.create_client()
    
    # Generate proposals using LLM
    messages = [{"role": "user", "content": prompt}]
    response_text = client.generate_response(messages)
    
    # Ensure response_text is a string
    if isinstance(response_text, str):
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
    """Generate and save research proposals to YAML file. 
    
    Note: This function always generates fresh proposals and overwrites any existing research.yaml file.
    """
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
        
        # Always create fresh research proposals (overwrite existing file)
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