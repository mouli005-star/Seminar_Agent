"""
Web Search Tools for SeminarAgent

This module provides tools for searching various academic sources including
Google Scholar, ResearchGate, arXiv, and other academic databases.
"""

import logging
import time
import random
from typing import List, Dict, Any, Optional
from urllib.parse import quote_plus, urljoin
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebSearchTools:
    """
    Collection of tools for searching academic sources on the web.
    
    This class provides methods to search various academic databases and
    extract relevant information from search results.
    """
    
    def __init__(self):
        """Initialize the Web Search Tools"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Rate limiting settings
        self.min_delay = 1.0
        self.max_delay = 3.0
        
    def _rate_limit(self):
        """Apply rate limiting to avoid being blocked"""
        delay = random.uniform(self.min_delay, self.max_delay)
        time.sleep(delay)
    
    def search_google_scholar(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search Google Scholar for academic publications.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        logger.info(f"Searching Google Scholar for: {query}")
        
        try:
            # Use DuckDuckGo to search Google Scholar
            with DDGS() as ddgs:
                search_query = f"site:scholar.google.com {query}"
                results = list(ddgs.text(search_query, max_results=max_results))
            
            processed_results = []
            for result in results:
                processed_result = self._process_google_scholar_result(result)
                if processed_result:
                    processed_results.append(processed_result)
                
                if len(processed_results) >= max_results:
                    break
            
            logger.info(f"Google Scholar search completed. Found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            return []
    
    def _process_google_scholar_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a Google Scholar search result"""
        try:
            # Extract basic information
            title = result.get('title', '')
            url = result.get('link', '')
            snippet = result.get('body', '')
            
            # Try to extract more detailed information
            authors, publication_date, journal = self._extract_metadata_from_snippet(snippet)
            
            return {
                'title': title,
                'authors': authors,
                'publication_date': publication_date,
                'journal': journal,
                'abstract': snippet,
                'url': url,
                'source': 'Google Scholar',
                'content_type': 'research_paper',
                'relevance_score': 0.8  # Default relevance score
            }
            
        except Exception as e:
            logger.warning(f"Error processing Google Scholar result: {e}")
            return None
    
    def search_researchgate(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search ResearchGate for academic publications.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        logger.info(f"Searching ResearchGate for: {query}")
        
        try:
            # Use DuckDuckGo to search ResearchGate
            with DDGS() as ddgs:
                search_query = f"site:researchgate.net {query}"
                results = list(ddgs.text(search_query, max_results=max_results))
            
            processed_results = []
            for result in results:
                processed_result = self._process_researchgate_result(result)
                if processed_result:
                    processed_results.append(processed_result)
                
                if len(processed_results) >= max_results:
                    break
            
            logger.info(f"ResearchGate search completed. Found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching ResearchGate: {e}")
            return []
    
    def _process_researchgate_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a ResearchGate search result"""
        try:
            title = result.get('title', '')
            url = result.get('link', '')
            snippet = result.get('body', '')
            
            # Extract metadata from ResearchGate format
            authors, publication_date, journal = self._extract_metadata_from_snippet(snippet)
            
            return {
                'title': title,
                'authors': authors,
                'publication_date': publication_date,
                'journal': journal,
                'abstract': snippet,
                'url': url,
                'source': 'ResearchGate',
                'content_type': 'research_paper',
                'relevance_score': 0.8
            }
            
        except Exception as e:
            logger.warning(f"Error processing ResearchGate result: {e}")
            return None
    
    def search_arxiv(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search arXiv for preprints and research papers.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        logger.info(f"Searching arXiv for: {query}")
        
        try:
            # Use DuckDuckGo to search arXiv
            with DDGS() as ddgs:
                search_query = f"site:arxiv.org {query}"
                results = list(ddgs.text(search_query, max_results=max_results))
            
            processed_results = []
            for result in results:
                processed_result = self._process_arxiv_result(result)
                if processed_result:
                    processed_results.append(processed_result)
                
                if len(processed_results) >= max_results:
                    break
            
            logger.info(f"arXiv search completed. Found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching arXiv: {e}")
            return []
    
    def _process_arxiv_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process an arXiv search result"""
        try:
            title = result.get('title', '')
            url = result.get('link', '')
            snippet = result.get('body', '')
            
            # Extract arXiv-specific metadata
            authors, publication_date, journal = self._extract_metadata_from_snippet(snippet)
            
            return {
                'title': title,
                'authors': authors,
                'publication_date': publication_date,
                'journal': 'arXiv',
                'abstract': snippet,
                'url': url,
                'source': 'arXiv',
                'content_type': 'preprint',
                'relevance_score': 0.7  # Preprints might be slightly less relevant
            }
            
        except Exception as e:
            logger.warning(f"Error processing arXiv result: {e}")
            return None
    
    def search_pubmed(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Search PubMed for medical and life sciences publications.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            
        Returns:
            List of dictionaries containing search results
        """
        logger.info(f"Searching PubMed for: {query}")
        
        try:
            # Use DuckDuckGo to search PubMed
            with DDGS() as ddgs:
                search_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
                results = list(ddgs.text(search_query, max_results=max_results))
            
            processed_results = []
            for result in results:
                processed_result = self._process_pubmed_result(result)
                if processed_result:
                    processed_results.append(processed_result)
                
                if len(processed_results) >= max_results:
                    break
            
            logger.info(f"PubMed search completed. Found {len(processed_results)} results")
            return processed_results
            
        except Exception as e:
            logger.error(f"Error searching PubMed: {e}")
            return []
    
    def _process_pubmed_result(self, result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a PubMed search result"""
        try:
            title = result.get('title', '')
            url = result.get('link', '')
            snippet = result.get('body', '')
            
            # Extract PubMed-specific metadata
            authors, publication_date, journal = self._extract_metadata_from_snippet(snippet)
            
            return {
                'title': title,
                'authors': authors,
                'publication_date': publication_date,
                'journal': journal,
                'abstract': snippet,
                'url': url,
                'source': 'PubMed',
                'content_type': 'research_paper',
                'relevance_score': 0.9  # PubMed articles are highly relevant
            }
            
        except Exception as e:
            logger.warning(f"Error processing PubMed result: {e}")
            return None
    
    def _extract_metadata_from_snippet(self, snippet: str) -> tuple:
        """
        Extract metadata (authors, publication date, journal) from a text snippet.
        
        Args:
            snippet: Text snippet to extract metadata from
            
        Returns:
            Tuple of (authors, publication_date, journal)
        """
        authors = []
        publication_date = None
        journal = None
        
        try:
            # This is a simplified extraction - in practice, you'd want more sophisticated parsing
            # based on the specific format of each source
            
            # Try to extract publication date (common formats)
            import re
            date_patterns = [
                r'(\d{4})',  # Year only
                r'(\d{4}-\d{2})',  # Year-Month
                r'(\d{4}-\d{2}-\d{2})',  # Full date
            ]
            
            for pattern in date_patterns:
                match = re.search(pattern, snippet)
                if match:
                    publication_date = match.group(1)
                    break
            
            # Try to extract journal name (usually in italics or quotes)
            journal_patterns = [
                r'Journal of ([^,]+)',
                r'([A-Z][a-z]+ [A-Z][a-z]+)',
                r'"([^"]+)"',
            ]
            
            for pattern in journal_patterns:
                match = re.search(pattern, snippet)
                if match:
                    journal = match.group(1)
                    break
            
            # Try to extract authors (usually in format "Author1, Author2")
            author_patterns = [
                r'([A-Z][a-z]+ [A-Z][a-z]+)',
                r'([A-Z][a-z]+ et al\.)',
            ]
            
            for pattern in author_patterns:
                matches = re.findall(pattern, snippet)
                if matches:
                    authors = matches[:3]  # Limit to first 3 authors
                    break
            
        except Exception as e:
            logger.debug(f"Error extracting metadata: {e}")
        
        return authors, publication_date, journal
    
    def search_multiple_sources(self, query: str, max_results: int = 20) -> List[Dict[str, Any]]:
        """
        Search multiple academic sources simultaneously.
        
        Args:
            query: Search query string
            max_results: Maximum number of total results to return
            
        Returns:
            List of dictionaries containing search results from all sources
        """
        logger.info(f"Searching multiple sources for: {query}")
        
        all_results = []
        sources = [
            ('Google Scholar', self.search_google_scholar),
            ('ResearchGate', self.search_researchgate),
            ('arXiv', self.search_arxiv),
            ('PubMed', self.search_pubmed)
        ]
        
        results_per_source = max_results // len(sources)
        
        for source_name, search_func in sources:
            try:
                logger.info(f"Searching {source_name}...")
                results = search_func(query, results_per_source)
                all_results.extend(results)
                
                # Apply rate limiting between sources
                self._rate_limit()
                
            except Exception as e:
                logger.error(f"Error searching {source_name}: {e}")
                continue
        
        # Remove duplicates and sort by relevance
        unique_results = self._remove_duplicates(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Multi-source search completed. Found {len(sorted_results)} unique results")
        return sorted_results[:max_results]
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on title similarity"""
        unique_results = []
        seen_titles = set()
        
        for result in results:
            title = result.get('title', '').lower().strip()
            
            # Check if this title is similar to any we've seen
            is_duplicate = False
            for seen_title in seen_titles:
                if self._similar_titles(title, seen_title):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_results.append(result)
                seen_titles.add(title)
        
        return unique_results
    
    def _similar_titles(self, title1: str, title2: str, threshold: float = 0.8) -> bool:
        """
        Check if two titles are similar using simple string comparison.
        
        Args:
            title1: First title
            title2: Second title
            threshold: Similarity threshold (0.0 to 1.0)
            
        Returns:
            True if titles are similar, False otherwise
        """
        # Simple similarity check - in practice, you'd want more sophisticated methods
        # like Levenshtein distance or semantic similarity
        
        # Remove common words and punctuation
        common_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words1 = set(word.lower() for word in title1.split() if word.lower() not in common_words)
        words2 = set(word.lower() for word in title2.split() if word.lower() not in common_words)
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return False
        
        similarity = intersection / union
        return similarity >= threshold


def main():
    """Test the Web Search Tools"""
    try:
        tools = WebSearchTools()
        query = "machine learning healthcare applications"
        
        print(f"Testing web search tools with query: {query}")
        
        # Test individual sources
        print("\n1. Testing Google Scholar...")
        google_results = tools.search_google_scholar(query, max_results=3)
        print(f"Found {len(google_results)} Google Scholar results")
        
        print("\n2. Testing ResearchGate...")
        researchgate_results = tools.search_researchgate(query, max_results=3)
        print(f"Found {len(researchgate_results)} ResearchGate results")
        
        print("\n3. Testing arXiv...")
        arxiv_results = tools.search_arxiv(query, max_results=3)
        print(f"Found {len(arxiv_results)} arXiv results")
        
        # Test multi-source search
        print("\n4. Testing multi-source search...")
        all_results = tools.search_multiple_sources(query, max_results=10)
        print(f"Found {len(all_results)} total results from all sources")
        
        # Display some results
        print("\nSample results:")
        for i, result in enumerate(all_results[:3], 1):
            print(f"\n{i}. {result.get('title', 'No title')}")
            print(f"   Source: {result.get('source', 'Unknown')}")
            print(f"   Authors: {', '.join(result.get('authors', []))}")
            print(f"   Date: {result.get('publication_date', 'Unknown')}")
        
    except Exception as e:
        print(f"Error testing Web Search Tools: {e}")


if __name__ == "__main__":
    main()

