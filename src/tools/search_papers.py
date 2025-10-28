#!/usr/bin/env python3
"""Search for related papers using OpenAlex or Semantic Scholar API."""

import json
import sys
import time
from pathlib import Path
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field


APIProvider = Literal["openalex", "semantic-scholar"]


class Paper(BaseModel):
    """Paper information from search APIs."""

    id: str
    title: str
    publication_year: int | None = None
    cited_by_count: int = 0
    doi: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    open_access_url: str | None = None


def fetch_with_retry(
    url: str,
    params: dict[str, Any],
    api_name: str,
    max_retries: int = 3,
    timeout: int = 30,
) -> dict[str, Any] | None:
    """Fetch data from API with retry logic and exponential backoff.

    Args:
        url: API endpoint URL
        params: Query parameters
        api_name: Name of the API (for logging)
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds

    Returns:
        JSON response data or None if all retries failed
    """
    for attempt in range(max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None
            
            # Check if we should retry
            if status_code == 429:  # Rate limit
                if attempt < max_retries:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    print(
                        f"[{api_name}] Rate limit exceeded (429). "
                        f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(
                        f"[{api_name}] Rate limit exceeded. Max retries reached.",
                        file=sys.stderr,
                    )
                    return None
            elif status_code and 500 <= status_code < 600:  # Server errors
                if attempt < max_retries:
                    wait_time = 2 ** attempt
                    print(
                        f"[{api_name}] Server error ({status_code}). "
                        f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})",
                        file=sys.stderr,
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    print(
                        f"[{api_name}] Server error ({status_code}). Max retries reached.",
                        file=sys.stderr,
                    )
                    return None
            else:
                # Client error (4xx except 429) - don't retry
                print(f"[{api_name}] HTTP error: {e}", file=sys.stderr)
                return None
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(
                    f"[{api_name}] Request timeout. "
                    f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                continue
            else:
                print(f"[{api_name}] Request timeout. Max retries reached.", file=sys.stderr)
                return None
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(
                    f"[{api_name}] Connection error. "
                    f"Retrying in {wait_time}s... (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait_time)
                continue
            else:
                print(f"[{api_name}] Connection error. Max retries reached.", file=sys.stderr)
                return None
        except requests.RequestException as e:
            print(f"[{api_name}] Request failed: {e}", file=sys.stderr)
            return None
    
    return None


def search_papers(
    query: str,
    limit: int = 10,
    domain_filter: bool = True,
    api: APIProvider = "openalex",
) -> list[Paper]:
    """Search for papers using specified API.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        domain_filter: If True, filter to Social Sciences domain only (OpenAlex only)
        api: Which API to use ('openalex' or 'semantic-scholar')

    Returns:
        List of Paper objects
    """
    if api == "semantic-scholar":
        return search_papers_semantic_scholar(query, limit)
    else:
        return search_papers_openalex(query, limit, domain_filter)


def search_papers_openalex(query: str, limit: int = 10, domain_filter: bool = True) -> list[Paper]:
    """Search for papers using OpenAlex API.

    Args:
        query: Search query string
        limit: Maximum number of results to return
        domain_filter: If True, filter to Social Sciences domain only

    Returns:
        List of Paper objects
    """
    base_url = "https://api.openalex.org/works"
    params = {
        "search": query,
        "per-page": limit,
        "select": "id,title,publication_year,cited_by_count,doi,abstract_inverted_index,authorships,primary_location,open_access,topics",
    }
    
    if domain_filter:
        # Filter to Social Sciences domain (domain.id:https://openalex.org/domains/2)
        params["filter"] = "topics.domain.id:https://openalex.org/domains/2"

    data = fetch_with_retry(base_url, params, "OpenAlex")
    if not data:
        return []

    papers = []
    for work in data.get("results", []):
        # Extract author names
        authors = []
        for authorship in work.get("authorships", []):
            author = authorship.get("author", {})
            if author and author.get("display_name"):
                authors.append(author["display_name"])

        # Extract venue name
        venue = None
        primary_location = work.get("primary_location", {})
        if primary_location:
            source = primary_location.get("source", {})
            if source:
                venue = source.get("display_name")

        # Extract open access URL
        open_access_url = None
        open_access = work.get("open_access", {})
        if open_access and open_access.get("is_oa"):
            open_access_url = open_access.get("oa_url")

        # Convert inverted abstract to plain text
        abstract = None
        abstract_inverted = work.get("abstract_inverted_index")
        if abstract_inverted:
            abstract = _convert_inverted_abstract(abstract_inverted)

        paper = Paper(
            id=work["id"],
            title=work.get("title", ""),
            publication_year=work.get("publication_year"),
            cited_by_count=work.get("cited_by_count", 0),
            doi=work.get("doi"),
            abstract=abstract,
            authors=authors[:5],  # Limit to first 5 authors
            venue=venue,
            open_access_url=open_access_url,
        )
        papers.append(paper)

    return papers


def _convert_inverted_abstract(inverted_abstract: dict[str, list[int]]) -> str:
    """Convert inverted abstract index to plain text.

    Args:
        inverted_abstract: Dictionary mapping words to their positions

    Returns:
        Plain text abstract
    """
    if not inverted_abstract:
        return ""

    # Create position to word mapping
    position_to_word: dict[int, str] = {}
    for word, positions in inverted_abstract.items():
        for pos in positions:
            position_to_word[pos] = word

    # Sort by position and join words
    sorted_positions = sorted(position_to_word.keys())
    words = [position_to_word[pos] for pos in sorted_positions]
    return " ".join(words)


def search_papers_semantic_scholar(query: str, limit: int = 10) -> list[Paper]:
    """Search for papers using Semantic Scholar API.

    Args:
        query: Search query string
        limit: Maximum number of results to return

    Returns:
        List of Paper objects
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,year,citationCount,authors,abstract,venue,openAccessPdf,externalIds",
    }

    data = fetch_with_retry(base_url, params, "Semantic Scholar")
    if not data:
        return []

    papers = []
    for work in data.get("data", []):
        # Extract author names
        authors = []
        for author in work.get("authors", []):
            if author.get("name"):
                authors.append(author["name"])

        # Extract DOI from externalIds
        doi = None
        external_ids = work.get("externalIds", {})
        if external_ids:
            doi_value = external_ids.get("DOI")
            if doi_value:
                doi = f"https://doi.org/{doi_value}"

        # Extract open access URL
        open_access_url = None
        open_access_pdf = work.get("openAccessPdf")
        if open_access_pdf and isinstance(open_access_pdf, dict):
            open_access_url = open_access_pdf.get("url")

        paper = Paper(
            id=work.get("paperId", ""),
            title=work.get("title", ""),
            publication_year=work.get("year"),
            cited_by_count=work.get("citationCount", 0),
            doi=doi,
            abstract=work.get("abstract"),
            authors=authors[:5],  # Limit to first 5 authors
            venue=work.get("venue"),
            open_access_url=open_access_url,
        )
        papers.append(paper)

    return papers


def save_results(papers: list[Paper], output_path: Path) -> None:
    """Save search results to JSON file.

    Args:
        papers: List of Paper objects
        output_path: Path to save the results
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = [paper.model_dump() for paper in papers]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    """Main function to run paper search."""
    if len(sys.argv) < 2:
        print("Usage: python search_papers.py <query> [limit] [--no-filter] [--api openalex|semantic-scholar]")
        print("Examples:")
        print("  python search_papers.py 'happiness well-being' 20")
        print("  python search_papers.py 'happiness' 10 --no-filter")
        print("  python search_papers.py 'happiness' 10 --api semantic-scholar")
        sys.exit(1)

    query = sys.argv[1]
    
    # Parse arguments
    args_without_flags = [arg for arg in sys.argv[2:] if not arg.startswith("--")]
    limit = int(args_without_flags[0]) if args_without_flags else 10
    
    # Check flags
    domain_filter = "--no-filter" not in sys.argv
    
    # Determine API provider
    api: APIProvider = "openalex"  # default
    for i, arg in enumerate(sys.argv):
        if arg == "--api" and i + 1 < len(sys.argv):
            api_value = sys.argv[i + 1]
            if api_value in ("openalex", "semantic-scholar"):
                api = api_value  # type: ignore
            else:
                print(f"Error: Unknown API '{api_value}'. Use 'openalex' or 'semantic-scholar'.")
                sys.exit(1)
    
    print(f"Searching for papers with query: '{query}'")
    print(f"API: {api}")
    print(f"Limit: {limit} papers")
    if api == "openalex":
        print(f"Domain filter: {'Social Sciences only' if domain_filter else 'All domains'}")
    print()

    papers = search_papers(query, limit, domain_filter, api)

    if not papers:
        print("No papers found.")
        return

    # Display results
    print(f"Found {len(papers)} papers:\n")
    for i, paper in enumerate(papers, 1):
        print(f"{i}. {paper.title}")
        if paper.authors:
            print(f"   Authors: {', '.join(paper.authors)}")
        if paper.venue:
            print(f"   Venue: {paper.venue}")
        if paper.publication_year:
            print(f"   Year: {paper.publication_year}")
        print(f"   Citations: {paper.cited_by_count}")
        if paper.doi:
            print(f"   DOI: {paper.doi}")
        if paper.open_access_url:
            print(f"   Open Access: {paper.open_access_url}")
        if paper.abstract:
            print(f"   Abstract: {paper.abstract[:200]}...")
        print()

    # Save results
    output_path = Path("outputs/paper_search_results.json")
    save_results(papers, output_path)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()