"""
Semantic search tool for finding variables based on label similarity.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


class SemanticSearcher:
    """Semantic search for JSON data structures based on label fields."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the semantic searcher with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatL2] = None
        self.labels: List[str] = []
        self.variables: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        
    def index_json_data(self, data: Dict[str, Any]) -> None:
        """
        Index JSON data structure by extracting labels and creating embeddings.
        
        Args:
            data: JSON data with structure {variable: {"label": ..., ...}}
        """
        self.labels = []
        self.variables = []
        self.metadata = []
        
        # Extract labels and variables
        for variable, info in data.items():
            if isinstance(info, dict) and "label" in info:
                self.labels.append(info["label"])
                self.variables.append(variable)
                self.metadata.append(info)
        
        if not self.labels:
            raise ValueError("No labels found in the data")
        
        # Create embeddings
        embeddings = self.model.encode(self.labels)
        
        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings.astype(np.float32))
        
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, str, float, Dict[str, Any]]]:
        """
        Search for variables based on label similarity.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of tuples (variable, label, similarity_score, full_metadata)
        """
        if self.index is None:
            raise ValueError("Index not built. Call index_json_data first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        
        # Search
        distances, indices = self.index.search(
            query_embedding.astype(np.float32), 
            min(top_k, len(self.labels))
        )
        
        # Prepare results
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.variables):  # Ensure valid index
                similarity = 1 / (1 + distance)  # Convert distance to similarity
                results.append((
                    self.variables[idx],
                    self.labels[idx],
                    similarity,
                    self.metadata[idx]
                ))
        
        return results
    
    def load_and_index_file(self, file_path: Path) -> None:
        """
        Load JSON file and index its contents.
        
        Args:
            file_path: Path to JSON file
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.index_json_data(data)


def create_searcher_from_file(json_path: Path) -> SemanticSearcher:
    """
    Convenience function to create and initialize a searcher from a JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Initialized SemanticSearcher instance
    """
    searcher = SemanticSearcher()
    searcher.load_and_index_file(json_path)
    return searcher


if __name__ == "__main__":
    import sys
    # デフォルトのJSONファイルパス
    default_json_path = Path("data/code-maps/codebook_map.json")
    # コマンドライン引数でパス指定があればそれを使う
    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        json_path = default_json_path

    # ファイル存在チェック
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        sys.exit(1)

    # 検索クエリを標準入力から受け取る
    print("\nSemantic Search Tool")
    print("====================")
    print(f"Using JSON file: {json_path}")
    print("Type your query and press Enter (Ctrl+C to exit).\n")

    # 検索器の作成とインデックス化
    searcher = SemanticSearcher()
    searcher.load_and_index_file(json_path)

    try:
        while True:
            query = input("Query: ").strip()
            if not query:
                continue
            results = searcher.search(query, top_k=5)
            print("\nResults:")
            print("-" * 40)
            for var, label, score, metadata in results:
                print(f"Variable: {var}")
                print(f"Label: {label}")
                print(f"Similarity: {score:.3f}")
                print(f"Value: {metadata.get('value', 'N/A')}")
                print()
    except KeyboardInterrupt:
        print("\nExiting.")