#!/usr/bin/env python3
"""Validate citations and bibliography entries for existence and accuracy."""

import re
import time
import requests
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
from urllib.parse import quote
from src.tools.search_papers import search_papers, Paper
from src.terminal_formatter import formatter, MessageType


class CitationValidator:
    """Validate the existence and accuracy of citations and bibliography entries."""
    
    def __init__(self):
        """Initialize the citation validator."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'SSA-Research-Tool/1.0 (mailto:research@example.com)'
        })
        
    def verify_doi(self, doi: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Verify if a DOI exists and return metadata if available.
        
        Args:
            doi: DOI string to verify
            
        Returns:
            Tuple of (exists: bool, metadata: Optional[Dict])
        """
        if not doi:
            return False, None
            
        # Clean DOI format
        clean_doi = doi.replace('https://doi.org/', '').replace('http://dx.doi.org/', '')
        clean_doi = clean_doi.strip()
        
        try:
            # Use DOI.org content negotiation to get metadata
            url = f"https://doi.org/{clean_doi}"
            headers = {
                'Accept': 'application/vnd.citationstyles.csl+json',
                'User-Agent': 'SSA-Research-Tool/1.0 (mailto:research@example.com)'
            }
            
            response = self.session.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                try:
                    metadata = response.json()
                    return True, metadata
                except json.JSONDecodeError:
                    # DOI exists but no JSON metadata available
                    return True, None
            else:
                return False, None
                
        except requests.RequestException as e:
            formatter.print(f"Error verifying DOI {clean_doi}: {e}", MessageType.WARNING)
            return False, None
    
    def search_paper_by_title(self, title: str, authors: Optional[List[str]] = None, 
                             year: Optional[int] = None) -> Tuple[bool, Optional[Paper]]:
        """Search for a paper by title and optionally authors/year.
        
        Args:
            title: Paper title to search for
            authors: Optional list of author names
            year: Optional publication year
            
        Returns:
            Tuple of (found: bool, paper: Optional[Paper])
        """
        if not title or len(title.strip()) < 5:
            return False, None
            
        try:
            # Build search query
            query = f'"{title.strip()}"'
            
            # Add author constraint if provided
            if authors and len(authors) > 0:
                # Use first author for search
                first_author = authors[0].split()[-1]  # Get last name
                query += f' author:"{first_author}"'
            
            # Search using OpenAlex
            papers = search_papers(query, limit=5, domain_filter=False)
            
            if not papers:
                # Try without quotes for more flexible matching
                query = title.strip()
                if authors and len(authors) > 0:
                    first_author = authors[0].split()[-1]
                    query += f' author:"{first_author}"'
                papers = search_papers(query, limit=3, domain_filter=False)
            
            # Find best match
            for paper in papers:
                # Check title similarity
                if self._titles_match(title, paper.title):
                    # Check year if provided
                    if year and paper.publication_year and abs(paper.publication_year - year) > 2:
                        continue
                    # Check authors if provided
                    if authors and not self._authors_match(authors, paper.authors):
                        continue
                    return True, paper
                    
            return False, None
            
        except Exception as e:
            formatter.print(f"Error searching paper by title '{title}': {e}", MessageType.WARNING)
            return False, None
    
    def _titles_match(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """Check if two titles match with some tolerance for differences."""
        if not title1 or not title2:
            return False
            
        # Normalize titles
        t1 = re.sub(r'[^\w\s]', '', title1.lower().strip())
        t2 = re.sub(r'[^\w\s]', '', title2.lower().strip())
        
        # Remove common stopwords and articles
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        t1_words = set(word for word in t1.split() if word not in stopwords and len(word) > 2)
        t2_words = set(word for word in t2.split() if word not in stopwords and len(word) > 2)
        
        if not t1_words or not t2_words:
            return False
            
        # Calculate Jaccard similarity
        intersection = len(t1_words.intersection(t2_words))
        union = len(t1_words.union(t2_words))
        
        similarity = intersection / union if union > 0 else 0
        return similarity >= threshold
    
    def _authors_match(self, authors1: List[str], authors2: List[str]) -> bool:
        """Check if author lists have reasonable overlap."""
        if not authors1 or not authors2:
            return True  # If no authors provided, don't filter on this
            
        # Extract last names
        lastnames1 = set()
        for author in authors1[:3]:  # Check first 3 authors
            parts = author.strip().split()
            if parts:
                lastnames1.add(parts[-1].lower())
        
        lastnames2 = set()
        for author in authors2[:3]:
            parts = author.strip().split()
            if parts:
                lastnames2.add(parts[-1].lower())
        
        # Check for any overlap
        return len(lastnames1.intersection(lastnames2)) > 0
    
    def validate_bibliography_entry(self, bib_entry: Dict[str, Any]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """Validate a single bibliography entry.
        
        Args:
            bib_entry: Dictionary containing bibliography entry fields
            
        Returns:
            Tuple of (valid: bool, validated_metadata: Optional[Dict])
        """
        title = bib_entry.get('title', '')
        authors = bib_entry.get('authors', [])
        year = bib_entry.get('year')
        doi = bib_entry.get('doi')
        
        # Try DOI verification first if available
        if doi:
            doi_valid, doi_metadata = self.verify_doi(doi)
            if doi_valid:
                formatter.print(f"✅ DOI verified: {doi}", MessageType.SUCCESS)
                return True, doi_metadata
            else:
                formatter.print(f"❌ DOI not found: {doi}", MessageType.WARNING)
        
        # Try title-based search
        if title:
            found, paper = self.search_paper_by_title(title, authors, year)
            if found and paper:
                formatter.print(f"✅ Paper found by title: {title}", MessageType.SUCCESS)
                # Convert Paper object to metadata dict
                metadata = {
                    'title': paper.title,
                    'authors': paper.authors,
                    'year': paper.publication_year,
                    'doi': paper.doi,
                    'citations': paper.cited_by_count
                }
                return True, metadata
            else:
                formatter.print(f"❌ Paper not found by title: {title}", MessageType.WARNING)
        
        return False, None
    
    def extract_citations_from_latex(self, latex_content: str) -> List[str]:
        """Extract all citation keys from LaTeX content.
        
        Args:
            latex_content: LaTeX document content
            
        Returns:
            List of citation keys found in the document
        """
        # Find all \cite{...} commands
        citation_pattern = r'\\cite(?:p|t|author|year)?\{([^}]+)\}'
        matches = re.findall(citation_pattern, latex_content)
        
        # Extract individual citation keys (handle multiple citations like \cite{key1,key2})
        citation_keys = set()
        for match in matches:
            keys = [key.strip() for key in match.split(',')]
            citation_keys.update(keys)
        
        return list(citation_keys)
    
    def parse_bibtex_entries(self, bibtex_content: str) -> Dict[str, Dict[str, Any]]:
        """Parse BibTeX content and extract entry information.
        
        Args:
            bibtex_content: BibTeX file content
            
        Returns:
            Dictionary mapping citation keys to entry metadata
        """
        entries = {}
        
        # Pattern to match BibTeX entries
        entry_pattern = r'@(\w+)\s*\{\s*([^,\s]+)\s*,([^}]+)\}'
        
        for match in re.finditer(entry_pattern, bibtex_content, re.DOTALL):
            entry_type = match.group(1).lower()
            citation_key = match.group(2)
            fields_text = match.group(3)
            
            # Parse fields
            fields = {}
            field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
            for field_match in re.finditer(field_pattern, fields_text):
                field_name = field_match.group(1).lower()
                field_value = field_match.group(2).strip()
                fields[field_name] = field_value
            
            # Convert to standardized format
            entry_data = {
                'type': entry_type,
                'title': fields.get('title', ''),
                'authors': self._parse_authors(fields.get('author', '')),
                'year': self._parse_year(fields.get('year', '')),
                'doi': fields.get('doi', ''),
                'journal': fields.get('journal', ''),
                'publisher': fields.get('publisher', ''),
                'original_fields': fields
            }
            
            entries[citation_key] = entry_data
        
        return entries
    
    def _parse_authors(self, author_string: str) -> List[str]:
        """Parse author string into list of individual authors."""
        if not author_string:
            return []
        
        # Split by 'and' keyword
        authors = re.split(r'\s+and\s+', author_string)
        return [author.strip() for author in authors if author.strip()]
    
    def _parse_year(self, year_string: str) -> Optional[int]:
        """Parse year string to integer."""
        if not year_string:
            return None
        
        # Extract 4-digit year
        year_match = re.search(r'\b(19|20)\d{2}\b', year_string)
        if year_match:
            return int(year_match.group())
        
        return None
    
    def validate_bibliography_file(self, bibtex_path: Path) -> Tuple[Dict[str, bool], Dict[str, Dict[str, Any]]]:
        """Validate all entries in a BibTeX file.
        
        Args:
            bibtex_path: Path to the BibTeX file
            
        Returns:
            Tuple of (validation_results: Dict[key, valid], validated_metadata: Dict[key, metadata])
        """
        if not bibtex_path.exists():
            formatter.print(f"Bibliography file not found: {bibtex_path}", MessageType.ERROR)
            return {}, {}
        
        # Read and parse BibTeX file
        with open(bibtex_path, 'r', encoding='utf-8') as f:
            bibtex_content = f.read()
        
        entries = self.parse_bibtex_entries(bibtex_content)
        formatter.print(f"Found {len(entries)} bibliography entries to validate", MessageType.INFO)
        
        validation_results = {}
        validated_metadata = {}
        
        for citation_key, entry_data in entries.items():
            formatter.print(f"Validating citation: {citation_key}", MessageType.INFO)
            
            # Add rate limiting to avoid overwhelming APIs
            time.sleep(0.5)
            
            valid, metadata = self.validate_bibliography_entry(entry_data)
            validation_results[citation_key] = valid
            
            if valid and metadata:
                validated_metadata[citation_key] = metadata
        
        return validation_results, validated_metadata
    
    def clean_invalid_citations(self, latex_path: Path, bibtex_path: Path, 
                               validation_results: Dict[str, bool]) -> Tuple[int, int]:
        """Remove invalid citations from LaTeX and BibTeX files.
        
        Args:
            latex_path: Path to LaTeX file
            bibtex_path: Path to BibTeX file
            validation_results: Dictionary mapping citation keys to validity
            
        Returns:
            Tuple of (citations_removed_from_latex, entries_removed_from_bibtex)
        """
        invalid_keys = {key for key, valid in validation_results.items() if not valid}
        
        if not invalid_keys:
            formatter.print("No invalid citations found to remove", MessageType.SUCCESS)
            return 0, 0
        
        formatter.print(f"Removing {len(invalid_keys)} invalid citations: {invalid_keys}", MessageType.WARNING)
        
        # Clean LaTeX file
        latex_removals = 0
        if latex_path.exists():
            with open(latex_path, 'r', encoding='utf-8') as f:
                latex_content = f.read()
            
            original_content = latex_content
            
            # Remove or replace citations
            for invalid_key in invalid_keys:
                # Pattern for single citation
                single_pattern = rf'\\cite(?:p|t|author|year)?\{{{re.escape(invalid_key)}\}}'
                latex_content = re.sub(single_pattern, '', latex_content)
                
                # Pattern for multiple citations (remove just the invalid key)
                multi_pattern = rf'\\cite(?:p|t|author|year)?\{{([^}}]*){re.escape(invalid_key)}([^}}]*)\}}'
                
                def replace_multi_cite(match):
                    content = match.group(1) + match.group(2)
                    # Clean up commas
                    content = re.sub(r',\s*,', ',', content)
                    content = content.strip(',').strip()
                    if content:
                        return f'\\cite{{{content}}}'
                    else:
                        return ''
                
                latex_content = re.sub(multi_pattern, replace_multi_cite, latex_content)
            
            # Count changes
            if latex_content != original_content:
                with open(latex_path, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                latex_removals = len(invalid_keys)
        
        # Clean BibTeX file
        bibtex_removals = 0
        if bibtex_path.exists():
            with open(bibtex_path, 'r', encoding='utf-8') as f:
                bibtex_content = f.read()
            
            original_entries = len(self.parse_bibtex_entries(bibtex_content))
            
            # Remove invalid entries
            for invalid_key in invalid_keys:
                entry_pattern = rf'@\w+\s*\{{\s*{re.escape(invalid_key)}\s*,.*?\n\}}'
                bibtex_content = re.sub(entry_pattern, '', bibtex_content, flags=re.DOTALL)
            
            # Clean up extra whitespace
            bibtex_content = re.sub(r'\n\s*\n\s*\n', '\n\n', bibtex_content)
            bibtex_content = bibtex_content.strip()
            
            new_entries = len(self.parse_bibtex_entries(bibtex_content))
            bibtex_removals = original_entries - new_entries
            
            with open(bibtex_path, 'w', encoding='utf-8') as f:
                f.write(bibtex_content)
        
        return latex_removals, bibtex_removals


def validate_paper_citations(latex_path: Path, bibtex_path: Path) -> bool:
    """Validate all citations in a paper and clean invalid ones.
    
    Args:
        latex_path: Path to LaTeX file
        bibtex_path: Path to BibTeX file
        
    Returns:
        True if validation completed successfully (regardless of found issues)
    """
    validator = CitationValidator()
    
    try:
        formatter.print("Starting citation validation process...", MessageType.PROGRESS)
        
        # Validate bibliography entries
        validation_results, validated_metadata = validator.validate_bibliography_file(bibtex_path)
        
        if not validation_results:
            formatter.print("No citations to validate", MessageType.INFO)
            return True
        
        # Report results
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        invalid_count = total_count - valid_count
        
        formatter.print(f"Citation validation results: {valid_count}/{total_count} valid citations", MessageType.INFO)
        
        if invalid_count > 0:
            formatter.print(f"Found {invalid_count} invalid citations", MessageType.WARNING)
            
            # Clean invalid citations
            latex_removals, bibtex_removals = validator.clean_invalid_citations(
                latex_path, bibtex_path, validation_results
            )
            
            formatter.print(f"Removed {latex_removals} citation references from LaTeX", MessageType.INFO)
            formatter.print(f"Removed {bibtex_removals} entries from bibliography", MessageType.INFO)
        else:
            formatter.print("All citations are valid!", MessageType.SUCCESS)
        
        return True
        
    except Exception as e:
        formatter.print(f"Error during citation validation: {e}", MessageType.ERROR)
        return False


if __name__ == "__main__":
    # Test the validator
    import sys
    if len(sys.argv) != 3:
        print("Usage: python citation_validator.py <latex_file> <bibtex_file>")
        sys.exit(1)
    
    latex_file = Path(sys.argv[1])
    bibtex_file = Path(sys.argv[2])
    
    success = validate_paper_citations(latex_file, bibtex_file)
    sys.exit(0 if success else 1)
