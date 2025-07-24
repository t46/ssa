#!/usr/bin/env python3
"""Search for related papers using OpenAlex API."""

import json
import sys
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field


class Paper(BaseModel):
    """Paper information from OpenAlex."""

    id: str
    title: str
    publication_year: int | None = None
    cited_by_count: int = 0
    doi: str | None = None
    abstract: str | None = None
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    open_access_url: str | None = None


def search_papers(query: str, limit: int = 10, domain_filter: bool = True) -> list[Paper]:
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

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from OpenAlex: {e}", file=sys.stderr)
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
        print("Usage: python search_papers.py <query> [limit] [--no-filter]")
        print("Example: python search_papers.py 'happiness well-being' 20")
        print("         python search_papers.py 'happiness' 10 --no-filter")
        sys.exit(1)

    query = sys.argv[1]
    # Parse limit, skipping --no-filter if present
    args = [arg for arg in sys.argv[2:] if arg != "--no-filter"]
    limit = int(args[0]) if args else 10

    domain_filter = "--no-filter" not in sys.argv
    print(f"Searching for papers with query: '{query}'")
    print(f"Limit: {limit} papers")
    print(f"Domain filter: {'Social Sciences only' if domain_filter else 'All domains'}\n")

    # Check if --no-filter flag is present
    domain_filter = "--no-filter" not in sys.argv
    papers = search_papers(query, limit, domain_filter)

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