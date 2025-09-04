"""
Content Parser for SeminarAgent

This module provides tools for parsing and extracting content from various
academic sources and web pages.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ContentParser:
    """
    Tool for parsing and extracting content from academic sources.
    
    This class provides methods to extract structured information from
    various types of academic content including abstracts, metadata,
    and full-text content when available.
    """
    
    def __init__(self):
        """Initialize the Content Parser"""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # Common academic content patterns
        self.abstract_patterns = [
            r'Abstract[:\s]*([^.]*(?:\.{3}|\.))',
            r'ABSTRACT[:\s]*([^.]*(?:\.{3}|\.))',
            r'Summary[:\s]*([^.]*(?:\.{3}|\.))',
            r'SUMMARY[:\s]*([^.]*(?:\.{3}|\.))'
        ]
        
        self.author_patterns = [
            r'Authors?[:\s]*([^.]*(?:\.{3}|\.))',
            r'By[:\s]*([^.]*(?:\.{3}|\.))',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*)'
        ]
        
        self.date_patterns = [
            r'(\d{4})',  # Year only
            r'(\d{4}-\d{2})',  # Year-Month
            r'(\d{4}-\d{2}-\d{2})',  # Full date
            r'Published[:\s]*(\d{4})',
            r'Date[:\s]*(\d{4})'
        ]
        
        self.journal_patterns = [
            r'Journal[:\s]*([^,.]*(?:\.{3}|\.))',
            r'Published in[:\s]*([^,.]*(?:\.{3}|\.))',
            r'([A-Z][a-z]+ [A-Z][a-z]+ Journal)',
            r'([A-Z][a-z]+ [A-Z][a-z]+)'
        ]
    
    def extract_content_from_url(self, url: str, max_length: int = 5000) -> Dict[str, Any]:
        """
        Extract content from a given URL.
        
        Args:
            url: URL to extract content from
            max_length: Maximum length of extracted content
            
        Returns:
            Dictionary containing extracted content and metadata
        """
        logger.info(f"Extracting content from URL: {url}")
        
        try:
            # Fetch the webpage
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse HTML content
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content based on the source type
            source_type = self._identify_source_type(url)
            
            if source_type == 'google_scholar':
                return self._extract_google_scholar_content(soup, max_length)
            elif source_type == 'researchgate':
                return self._extract_researchgate_content(soup, max_length)
            elif source_type == 'arxiv':
                return self._extract_arxiv_content(soup, max_length)
            elif source_type == 'pubmed':
                return self._extract_pubmed_content(soup, max_length)
            else:
                return self._extract_generic_content(soup, max_length)
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return {
                'error': str(e),
                'url': url,
                'content': '',
                'metadata': {}
            }
    
    def _identify_source_type(self, url: str) -> str:
        """Identify the type of academic source from the URL"""
        domain = urlparse(url).netloc.lower()
        
        if 'scholar.google.com' in domain:
            return 'google_scholar'
        elif 'researchgate.net' in domain:
            return 'researchgate'
        elif 'arxiv.org' in domain:
            return 'arxiv'
        elif 'pubmed.ncbi.nlm.nih.gov' in domain:
            return 'pubmed'
        else:
            return 'generic'
    
    def _extract_google_scholar_content(self, soup: BeautifulSoup, max_length: int) -> Dict[str, Any]:
        """Extract content from Google Scholar pages"""
        content = {
            'title': '',
            'authors': [],
            'abstract': '',
            'publication_date': '',
            'journal': '',
            'citations': 0,
            'url': '',
            'content_type': 'research_paper'
        }
        
        try:
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                content['title'] = title_elem.get_text().strip()
            
            # Extract authors
            author_elems = soup.find_all('a', href=re.compile(r'/citations'))
            if author_elems:
                content['authors'] = [elem.get_text().strip() for elem in author_elems]
            
            # Extract abstract
            abstract_elem = soup.find('div', class_='gs_rs') or soup.find('div', string=re.compile(r'Abstract'))
            if abstract_elem:
                content['abstract'] = abstract_elem.get_text().strip()[:max_length]
            
            # Extract publication info
            pub_info = soup.find('div', class_='gs_a')
            if pub_info:
                pub_text = pub_info.get_text()
                content['journal'] = self._extract_journal(pub_text)
                content['publication_date'] = self._extract_date(pub_text)
            
            # Extract citation count
            citation_elem = soup.find('a', string=re.compile(r'Cited by'))
            if citation_elem:
                citation_text = citation_elem.get_text()
                citation_match = re.search(r'(\d+)', citation_text)
                if citation_match:
                    content['citations'] = int(citation_match.group(1))
            
        except Exception as e:
            logger.warning(f"Error extracting Google Scholar content: {e}")
        
        return content
    
    def _extract_researchgate_content(self, soup: BeautifulSoup, max_length: int) -> Dict[str, Any]:
        """Extract content from ResearchGate pages"""
        content = {
            'title': '',
            'authors': [],
            'abstract': '',
            'publication_date': '',
            'journal': '',
            'doi': '',
            'content_type': 'research_paper'
        }
        
        try:
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                content['title'] = title_elem.get_text().strip()
            
            # Extract authors
            author_elems = soup.find_all('a', href=re.compile(r'/profile/'))
            if author_elems:
                content['authors'] = [elem.get_text().strip() for elem in author_elems]
            
            # Extract abstract
            abstract_elem = soup.find('div', class_='abstract') or soup.find('div', string=re.compile(r'Abstract'))
            if abstract_elem:
                content['abstract'] = abstract_elem.get_text().strip()[:max_length]
            
            # Extract publication info
            pub_info = soup.find('div', class_='publication-info')
            if pub_info:
                pub_text = pub_info.get_text()
                content['journal'] = self._extract_journal(pub_text)
                content['publication_date'] = self._extract_date(pub_text)
            
            # Extract DOI
            doi_elem = soup.find('a', href=re.compile(r'doi\.org'))
            if doi_elem:
                content['doi'] = doi_elem.get_text().strip()
            
        except Exception as e:
            logger.warning(f"Error extracting ResearchGate content: {e}")
        
        return content
    
    def _extract_arxiv_content(self, soup: BeautifulSoup, max_length: int) -> Dict[str, Any]:
        """Extract content from arXiv pages"""
        content = {
            'title': '',
            'authors': [],
            'abstract': '',
            'publication_date': '',
            'journal': 'arXiv',
            'arxiv_id': '',
            'content_type': 'preprint'
        }
        
        try:
            # Extract title
            title_elem = soup.find('h1', class_='title') or soup.find('title')
            if title_elem:
                content['title'] = title_elem.get_text().replace('Title:', '').strip()
            
            # Extract authors
            author_elems = soup.find_all('div', class_='authors')
            if author_elems:
                for elem in author_elems:
                    author_links = elem.find_all('a')
                    if author_links:
                        content['authors'] = [link.get_text().strip() for link in author_links]
            
            # Extract abstract
            abstract_elem = soup.find('blockquote', class_='abstract')
            if abstract_elem:
                content['abstract'] = abstract_elem.get_text().replace('Abstract:', '').strip()[:max_length]
            
            # Extract arXiv ID
            id_elem = soup.find('div', class_='submission-history')
            if id_elem:
                id_text = id_elem.get_text()
                id_match = re.search(r'(arXiv:\d+\.\d+)', id_text)
                if id_match:
                    content['arxiv_id'] = id_match.group(1)
            
            # Extract submission date
            date_elem = soup.find('div', class_='submission-history')
            if date_elem:
                date_text = date_elem.get_text()
                content['publication_date'] = self._extract_date(date_text)
            
        except Exception as e:
            logger.warning(f"Error extracting arXiv content: {e}")
        
        return content
    
    def _extract_pubmed_content(self, soup: BeautifulSoup, max_length: int) -> Dict[str, Any]:
        """Extract content from PubMed pages"""
        content = {
            'title': '',
            'authors': [],
            'abstract': '',
            'publication_date': '',
            'journal': '',
            'pmid': '',
            'content_type': 'research_paper'
        }
        
        try:
            # Extract title
            title_elem = soup.find('h1', class_='title') or soup.find('title')
            if title_elem:
                content['title'] = title_elem.get_text().strip()
            
            # Extract authors
            author_elems = soup.find_all('a', class_='full-name')
            if author_elems:
                content['authors'] = [elem.get_text().strip() for elem in author_elems]
            
            # Extract abstract
            abstract_elem = soup.find('div', class_='abstract-content')
            if abstract_elem:
                content['abstract'] = abstract_elem.get_text().strip()[:max_length]
            
            # Extract journal info
            journal_elem = soup.find('div', class_='journal-citation')
            if journal_elem:
                journal_text = journal_elem.get_text()
                content['journal'] = self._extract_journal(journal_text)
                content['publication_date'] = self._extract_date(journal_text)
            
            # Extract PMID
            pmid_elem = soup.find('span', class_='pmid')
            if pmid_elem:
                content['pmid'] = pmid_elem.get_text().strip()
            
        except Exception as e:
            logger.warning(f"Error extracting PubMed content: {e}")
        
        return content
    
    def _extract_generic_content(self, soup: BeautifulSoup, max_length: int) -> Dict[str, Any]:
        """Extract content from generic academic pages"""
        content = {
            'title': '',
            'authors': [],
            'abstract': '',
            'publication_date': '',
            'journal': '',
            'content_type': 'research_paper'
        }
        
        try:
            # Extract title
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                content['title'] = title_elem.get_text().strip()
            
            # Try to extract abstract using common patterns
            abstract = self._extract_abstract_generic(soup)
            if abstract:
                content['abstract'] = abstract[:max_length]
            
            # Try to extract authors using common patterns
            authors = self._extract_authors_generic(soup)
            if authors:
                content['authors'] = authors
            
            # Try to extract publication date
            date = self._extract_date_generic(soup)
            if date:
                content['publication_date'] = date
            
            # Try to extract journal name
            journal = self._extract_journal_generic(soup)
            if journal:
                content['journal'] = journal
            
        except Exception as e:
            logger.warning(f"Error extracting generic content: {e}")
        
        return content
    
    def _extract_abstract_generic(self, soup: BeautifulSoup) -> str:
        """Extract abstract from generic academic pages"""
        # Look for common abstract patterns
        for pattern in self.abstract_patterns:
            elements = soup.find_all(string=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                match = re.search(pattern, element, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        # Look for divs with abstract-like content
        abstract_elements = soup.find_all(['div', 'p'], string=re.compile(r'abstract|summary', re.IGNORECASE))
        for elem in abstract_elements:
            next_elem = elem.find_next_sibling()
            if next_elem and next_elem.get_text().strip():
                return next_elem.get_text().strip()
        
        return ''
    
    def _extract_authors_generic(self, soup: BeautifulSoup) -> List[str]:
        """Extract authors from generic academic pages"""
        authors = []
        
        # Look for common author patterns
        for pattern in self.author_patterns:
            elements = soup.find_all(string=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                match = re.search(pattern, element, re.IGNORECASE)
                if match:
                    author_text = match.group(1).strip()
                    # Split multiple authors
                    if ',' in author_text:
                        authors.extend([author.strip() for author in author_text.split(',')])
                    else:
                        authors.append(author_text)
        
        # Remove duplicates and clean up
        unique_authors = []
        for author in authors:
            clean_author = re.sub(r'\s+', ' ', author.strip())
            if clean_author and clean_author not in unique_authors:
                unique_authors.append(clean_author)
        
        return unique_authors[:5]  # Limit to first 5 authors
    
    def _extract_date_generic(self, soup: BeautifulSoup) -> str:
        """Extract publication date from generic academic pages"""
        # Look for common date patterns
        for pattern in self.date_patterns:
            elements = soup.find_all(string=re.compile(pattern))
            for element in elements:
                match = re.search(pattern, element)
                if match:
                    return match.group(1).strip()
        
        return ''
    
    def _extract_journal_generic(self, soup: BeautifulSoup) -> str:
        """Extract journal name from generic academic pages"""
        # Look for common journal patterns
        for pattern in self.journal_patterns:
            elements = soup.find_all(string=re.compile(pattern, re.IGNORECASE))
            for element in elements:
                match = re.search(pattern, element, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
        
        return ''
    
    def _extract_journal(self, text: str) -> str:
        """Extract journal name from text"""
        for pattern in self.journal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return ''
    
    def _extract_date(self, text: str) -> str:
        """Extract publication date from text"""
        for pattern in self.date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip()
        return ''
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\-\:\;\(\)]', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace('–', '-').replace('—', '-')
        
        return text.strip()
    
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text using simple frequency analysis"""
        if not text:
            return []
        
        # Remove common words
        common_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those',
            'from', 'as', 'it', 'its', 'they', 'them', 'their', 'we', 'us', 'our', 'you', 'your'
        }
        
        # Extract words
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common words and count frequency
        word_count = {}
        for word in words:
            if word not in common_words:
                word_count[word] = word_count.get(word, 0) + 1
        
        # Sort by frequency and return top keywords
        sorted_words = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:max_keywords]]


def main():
    """Test the Content Parser"""
    try:
        parser = ContentParser()
        
        # Test text cleaning
        test_text = "This   is   a   test   text   with   extra   spaces."
        cleaned = parser.clean_text(test_text)
        print(f"Original: '{test_text}'")
        print(f"Cleaned: '{cleaned}'")
        
        # Test keyword extraction
        test_abstract = """
        Machine learning algorithms have shown remarkable success in healthcare applications.
        Deep learning models can analyze medical images with high accuracy.
        Natural language processing helps extract information from clinical notes.
        """
        keywords = parser.extract_keywords(test_abstract, max_keywords=5)
        print(f"\nExtracted keywords: {keywords}")
        
        # Test pattern matching
        test_date = "Published in 2023"
        extracted_date = parser._extract_date(test_date)
        print(f"\nExtracted date from '{test_date}': {extracted_date}")
        
        test_journal = "Journal of Medical AI"
        extracted_journal = parser._extract_journal(test_journal)
        print(f"Extracted journal from '{test_journal}': {extracted_journal}")
        
    except Exception as e:
        print(f"Error testing Content Parser: {e}")


if __name__ == "__main__":
    main()

