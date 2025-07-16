"""
Example usage of the semantic search tool.
"""

import json
from pathlib import Path
from semantic_search import SemanticSearcher, create_searcher_from_file


def demo_semantic_search():
    """Demonstrate semantic search functionality."""
    
    # Sample WVS-like data structure
    sample_data = {
        "version": {
            "label": "Version of Data File",
            "description": "Version of the data-file",
            "value": "6-0-0 (2022-05-31)"
        },
        "doi": {
            "label": "Digital Object Identifier",
            "description": "Digital Object Identifier (DOI)",
            "value": "doi.org/10.14281/18241.24"
        },
        "Q1": {
            "label": "Feeling of happiness",
            "description": "Taking all things together, would you say you are happy?",
            "value": "1-4 scale"
        },
        "Q2": {
            "label": "State of health",
            "description": "All in all, how would you describe your state of health?",
            "value": "1-5 scale"
        },
        "B_COUNTRY": {
            "label": "Country/Society",
            "description": "Country code",
            "value": "840 (USA)"
        },
        "S017": {
            "label": "Original weight",
            "description": "Original survey weight",
            "value": "numeric"
        }
    }
    
    # Initialize searcher
    searcher = SemanticSearcher()
    searcher.index_json_data(sample_data)
    
    # Test queries
    test_queries = [
        "happiness feeling",
        "health condition",
        "country code",
        "survey weight",
        "file version",
        "DOI"
    ]
    
    print("Semantic Search Demo")
    print("=" * 80)
    
    for query in test_queries:
        print(f"\n🔍 Searching for: '{query}'")
        print("-" * 60)
        
        results = searcher.search(query, top_k=3)
        
        for i, (variable, label, score, metadata) in enumerate(results, 1):
            print(f"{i}. Variable: {variable}")
            print(f"   Label: {label}")
            print(f"   Similarity Score: {score:.4f}")
            print(f"   Description: {metadata.get('description', 'N/A')}")
            print(f"   Value: {metadata.get('value', 'N/A')}")
            print()


def demo_file_loading():
    """Demonstrate loading from a JSON file."""
    
    # Create a sample JSON file
    sample_file = Path("sample_codebook.json")
    
    sample_data = {
        "A001": {
            "label": "Important in life: Family",
            "description": "How important is family in your life?",
            "value": "1=Very important, 4=Not at all important"
        },
        "A002": {
            "label": "Important in life: Friends",
            "description": "How important are friends in your life?",
            "value": "1=Very important, 4=Not at all important"
        },
        "A003": {
            "label": "Important in life: Leisure time",
            "description": "How important is leisure time in your life?",
            "value": "1=Very important, 4=Not at all important"
        }
    }
    
    # Save sample data
    with open(sample_file, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, indent=2)
    
    print("\n\nFile-based Search Demo")
    print("=" * 80)
    
    # Create searcher from file
    searcher = create_searcher_from_file(sample_file)
    
    # Search
    query = "importance of friendship"
    print(f"\n🔍 Searching for: '{query}'")
    print("-" * 60)
    
    results = searcher.search(query, top_k=2)
    for variable, label, score, metadata in results:
        print(f"Variable: {variable}")
        print(f"Label: {label}")
        print(f"Similarity: {score:.4f}")
        print()
    
    # Clean up
    sample_file.unlink()


if __name__ == "__main__":
    demo_semantic_search()
    demo_file_loading()