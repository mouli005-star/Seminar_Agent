"""
Web Research Agent for SeminarAgent

This agent is responsible for searching and gathering recent academic journals
and publications related to the given seminar topic.
"""

import logging
import yaml
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from crewai import Agent, Task
from langchain_openai import ChatOpenAI

from tools.web_search_tools import WebSearchTools
from tools.content_parser import ContentParser

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Data structure for search results"""
    title: str
    authors: List[str]
    publication_date: Optional[str]
    abstract: str
    source: str
    url: str
    doi: Optional[str]
    keywords: List[str]
    relevance_score: float
    content_type: str


class WebResearchAgent:
    """
    Agent responsible for conducting web research on academic topics.
    
    This agent searches multiple academic sources for recent publications
    and filters results based on relevance and recency.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """Initialize the Web Research Agent"""
        self.config = self._load_config(config_path)
        self.search_tools = WebSearchTools()
        self.content_parser = ContentParser()
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=self._get_openai_key()
        )
        
        # Initialize CrewAI agent
        self.agent = Agent(
            role=self.config['web_research_agent']['role'],
            goal=self.config['web_research_agent']['goal'],
            backstory="""You are an expert academic researcher with years of experience 
            in finding and evaluating scholarly sources. You have a deep understanding 
            of academic databases, search strategies, and source evaluation criteria.""",
            verbose=True,
            allow_delegation=False,
            tools=[self.search_tools.search_google_scholar, 
                   self.search_tools.search_researchgate,
                   self.search_tools.search_arxiv],
            llm=self.llm
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {config_path}")
            return self._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails"""
        return {
            'web_research_agent': {
                'role': 'Academic Research Specialist',
                'goal': 'Find relevant academic sources',
                'search': {
                    'max_results': 20,
                    'date_range': '2014-2024',
                    'sources': ['Google Scholar', 'ResearchGate', 'arXiv']
                }
            }
        }
    
    def _get_openai_key(self) -> str:
        """Get OpenAI API key from environment"""
        import os
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return api_key
    
    def research_topic(self, topic: str, max_results: int = None) -> List[SearchResult]:
        """
        Conduct comprehensive research on the given topic.
        
        Args:
            topic: The seminar topic to research
            max_results: Maximum number of results to return
            
        Returns:
            List of SearchResult objects with research findings
        """
        logger.info(f"Starting research on topic: {topic}")
        
        if max_results is None:
            max_results = self.config['web_research_agent']['search']['max_results']
        
        # Create research task
        research_task = Task(
            description=f"""
            Conduct comprehensive research on the topic: "{topic}"
            
            Requirements:
            1. Search for recent academic publications (2014-2024)
            2. Focus on peer-reviewed journal articles and research papers
            3. Find at least {max_results} relevant sources
            4. Prioritize high-impact factor journals
            5. Include systematic reviews and meta-analyses when available
            6. Ensure sources are directly relevant to the seminar topic
            
            Search Strategy:
            - Use multiple academic sources (Google Scholar, ResearchGate, arXiv)
            - Apply advanced search filters for recent publications
            - Evaluate source credibility and relevance
            - Collect comprehensive metadata for each source
            
            Output: Provide a detailed list of sources with:
            - Title, authors, publication date
            - Abstract and key findings
            - Source URL and DOI (if available)
            - Relevance score and content type
            """,
            agent=self.agent,
            expected_output=f"""
            A comprehensive list of {max_results} academic sources related to "{topic}",
            including all required metadata and relevance assessments.
            """
        )
        
        try:
            # Execute research task
            result = research_task.execute()
            logger.info("Research task completed successfully")
            
            # Parse and structure the results
            search_results = self._parse_research_results(result, topic)
            
            # Filter and rank results
            filtered_results = self._filter_and_rank_results(search_results, max_results)
            
            logger.info(f"Research completed. Found {len(filtered_results)} relevant sources")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during research execution: {e}")
            # Fallback to direct search
            return self._fallback_search(topic, max_results)
    
    def _parse_research_results(self, result: str, topic: str) -> List[SearchResult]:
        """Parse the research results from the agent's output"""
        try:
            # This is a simplified parser - in practice, you'd want more sophisticated parsing
            # based on the actual output format from your LLM
            search_results = []
            
            # For now, we'll create a basic structure
            # In a real implementation, you'd parse the actual LLM output
            logger.info("Parsing research results...")
            
            # Placeholder for parsed results
            # This would be replaced with actual parsing logic
            return search_results
            
        except Exception as e:
            logger.error(f"Error parsing research results: {e}")
            return []
    
    def _filter_and_rank_results(self, results: List[SearchResult], max_results: int) -> List[SearchResult]:
        """Filter and rank search results based on quality and relevance"""
        if not results:
            return []
        
        # Sort by relevance score (descending)
        sorted_results = sorted(results, key=lambda x: x.relevance_score, reverse=True)
        
        # Apply additional filtering
        filtered_results = []
        for result in sorted_results:
            if self._meets_quality_criteria(result):
                filtered_results.append(result)
                if len(filtered_results) >= max_results:
                    break
        
        return filtered_results
    
    def _meets_quality_criteria(self, result: SearchResult) -> bool:
        """Check if a search result meets quality criteria"""
        # Check publication date (should be within last 10 years)
        if result.publication_date:
            try:
                pub_date = datetime.strptime(result.publication_date, "%Y-%m-%d")
                if pub_date < datetime.now() - timedelta(days=3650):  # 10 years
                    return False
            except ValueError:
                pass
        
        # Check abstract length
        if len(result.abstract) < 50:
            return False
        
        # Check relevance score
        if result.relevance_score < 0.5:
            return False
        
        return True
    
    def _fallback_search(self, topic: str, max_results: int) -> List[SearchResult]:
        """Fallback search method if the main research fails"""
        logger.info("Using fallback search method")
        
        try:
            # Use direct search tools
            results = []
            
            # Search Google Scholar
            google_results = self.search_tools.search_google_scholar(
                query=topic,
                max_results=max_results // 3
            )
            results.extend(self._convert_to_search_results(google_results, "Google Scholar"))
            
            # Search ResearchGate
            researchgate_results = self.search_tools.search_researchgate(
                query=topic,
                max_results=max_results // 3
            )
            results.extend(self._convert_to_search_results(researchgate_results, "ResearchGate"))
            
            # Search arXiv
            arxiv_results = self.search_tools.search_arxiv(
                query=topic,
                max_results=max_results // 3
            )
            results.extend(self._convert_to_search_results(arxiv_results, "arXiv"))
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Fallback search also failed: {e}")
            return []
    
    def _convert_to_search_results(self, raw_results: List[Dict], source: str) -> List[SearchResult]:
        """Convert raw search results to SearchResult objects"""
        search_results = []
        
        for result in raw_results:
            try:
                search_result = SearchResult(
                    title=result.get('title', ''),
                    authors=result.get('authors', []),
                    publication_date=result.get('publication_date'),
                    abstract=result.get('abstract', ''),
                    source=source,
                    url=result.get('url', ''),
                    doi=result.get('doi'),
                    keywords=result.get('keywords', []),
                    relevance_score=result.get('relevance_score', 0.7),
                    content_type=result.get('content_type', 'research_paper')
                )
                search_results.append(search_result)
            except Exception as e:
                logger.warning(f"Error converting result: {e}")
                continue
        
        return search_results
    
    def get_research_summary(self, results: List[SearchResult]) -> Dict[str, Any]:
        """Generate a summary of the research findings"""
        if not results:
            return {"error": "No research results available"}
        
        summary = {
            "total_sources": len(results),
            "sources_by_year": {},
            "sources_by_type": {},
            "top_keywords": [],
            "research_gaps": [],
            "trends": []
        }
        
        # Analyze sources by year
        for result in results:
            if result.publication_date:
                year = result.publication_date[:4]
                summary["sources_by_year"][year] = summary["sources_by_year"].get(year, 0) + 1
        
        # Analyze sources by type
        for result in results:
            content_type = result.content_type
            summary["sources_by_type"][content_type] = summary["sources_by_type"].get(content_type, 0) + 1
        
        # Collect keywords
        all_keywords = []
        for result in results:
            all_keywords.extend(result.keywords)
        
        # Count keyword frequency
        keyword_count = {}
        for keyword in all_keywords:
            keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        
        # Get top keywords
        summary["top_keywords"] = sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return summary


def main():
    """Test the Web Research Agent"""
    try:
        agent = WebResearchAgent()
        topic = "Machine Learning in Healthcare"
        results = agent.research_topic(topic, max_results=10)
        
        print(f"Research completed for topic: {topic}")
        print(f"Found {len(results)} relevant sources")
        
        summary = agent.get_research_summary(results)
        print("\nResearch Summary:")
        print(f"Total sources: {summary['total_sources']}")
        print(f"Sources by year: {summary['sources_by_year']}")
        print(f"Top keywords: {summary['top_keywords'][:5]}")
        
    except Exception as e:
        print(f"Error testing Web Research Agent: {e}")


if __name__ == "__main__":
    main()
