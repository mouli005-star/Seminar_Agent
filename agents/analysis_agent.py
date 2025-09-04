"""
Analysis Agent for SeminarAgent

This agent analyzes gathered research content and identifies key insights,
trends, and findings for report generation.
"""

import logging
from typing import List, Dict, Any
from crewai import Agent, Task
from langchain_openai import ChatOpenAI

from agents.web_research_agent import SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AnalysisAgent:
    """
    Agent responsible for analyzing research content and extracting insights.
    
    This agent processes gathered research materials to identify key findings,
    trends, methodologies, and research gaps.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """Initialize the Analysis Agent"""
        self.config = self._load_config(config_path)
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=self._get_openai_key()
        )
        
        # Initialize CrewAI agent
        self.agent = Agent(
            role=self.config['analysis_agent']['role'],
            goal=self.config['analysis_agent']['goal'],
            backstory="""You are an expert research analyst with deep expertise 
            in academic research methodology and content analysis. You excel at 
            identifying patterns, trends, and insights from complex research data.""",
            verbose=True,
            allow_delegation=False,
            llm=self.llm
        )
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            import yaml
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails"""
        return {
            'analysis_agent': {
                'role': 'Research Analyst',
                'goal': 'Analyze research content and extract insights'
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
    
    def analyze_research(self, topic: str, research_results: List[SearchResult]) -> Dict[str, Any]:
        """
        Analyze the gathered research content.
        
        Args:
            topic: The seminar topic
            research_results: List of research results to analyze
            
        Returns:
            Dictionary containing analysis results and insights
        """
        logger.info(f"Starting analysis of {len(research_results)} research sources")
        
        try:
            # Create analysis task
            analysis_task = Task(
                description=f"""
                Analyze the research content for the topic: "{topic}"
                
                Research Sources: {len(research_results)} sources
                
                Analysis Requirements:
                1. Identify key themes and findings across all sources
                2. Analyze trends over time (2014-2024)
                3. Identify research methodologies used
                4. Find research gaps and future directions
                5. Extract controversial or conflicting findings
                6. Identify high-impact contributions
                
                Output: Provide comprehensive analysis including:
                - Executive summary of key findings
                - Trend analysis over time
                - Methodology overview
                - Research gaps identification
                - Future research directions
                - Controversies and debates
                """,
                agent=self.agent,
                expected_output="""
                A comprehensive analysis report with all required sections
                and detailed insights from the research materials.
                """
            )
            
            # Execute analysis task
            result = analysis_task.execute()
            logger.info("Analysis task completed successfully")
            
            # Process and structure the results
            analysis_results = self._process_analysis_results(result, research_results)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {
                'error': str(e),
                'analysis_summary': 'Analysis failed due to error',
                'key_findings': [],
                'trends': [],
                'research_gaps': []
            }
    
    def _process_analysis_results(self, result: str, research_results: List[SearchResult]) -> Dict[str, Any]:
        """Process the analysis results from the agent's output"""
        try:
            # This is a simplified processor - in practice, you'd want more sophisticated parsing
            # based on the actual output format from your LLM
            
            analysis_results = {
                'analysis_summary': 'Analysis completed successfully',
                'key_findings': self._extract_key_findings(research_results),
                'trends': self._analyze_trends(research_results),
                'research_gaps': self._identify_research_gaps(research_results),
                'methodologies': self._analyze_methodologies(research_results),
                'future_directions': [],
                'controversies': [],
                'timestamp': '2024-01-01T00:00:00Z'
            }
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error processing analysis results: {e}")
            return {'error': str(e)}
    
    def _extract_key_findings(self, research_results: List[SearchResult]) -> List[str]:
        """Extract key findings from research results"""
        findings = []
        
        # Analyze abstracts for key findings
        for result in research_results:
            if result.abstract:
                # Simple keyword-based extraction
                if any(keyword in result.abstract.lower() for keyword in ['significant', 'important', 'key', 'major']):
                    findings.append(f"Key finding from {result.title}: {result.abstract[:200]}...")
        
        return findings[:10]  # Limit to top 10 findings
    
    def _analyze_trends(self, research_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Analyze trends over time in the research"""
        trends = []
        
        # Group by year and analyze
        year_data = {}
        for result in research_results:
            if result.publication_date:
                year = result.publication_date[:4]
                if year not in year_data:
                    year_data[year] = []
                year_data[year].append(result)
        
        # Analyze trends
        for year in sorted(year_data.keys()):
            year_results = year_data[year]
            trends.append({
                'year': year,
                'publication_count': len(year_results),
                'key_topics': self._extract_key_topics(year_results),
                'methodologies': self._extract_methodologies(year_results)
            })
        
        return trends
    
    def _identify_research_gaps(self, research_results: List[SearchResult]) -> List[str]:
        """Identify research gaps from the gathered materials"""
        gaps = []
        
        # Simple gap identification based on content analysis
        all_keywords = []
        for result in research_results:
            all_keywords.extend(result.keywords)
        
        # Look for underrepresented areas
        keyword_count = {}
        for keyword in all_keywords:
            keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        
        # Identify gaps (keywords mentioned only once or twice)
        for keyword, count in keyword_count.items():
            if count <= 2:
                gaps.append(f"Limited research on: {keyword}")
        
        return gaps[:5]  # Limit to top 5 gaps
    
    def _analyze_methodologies(self, research_results: List[SearchResult]) -> Dict[str, int]:
        """Analyze research methodologies used"""
        methodologies = {}
        
        # Simple methodology detection based on keywords
        methodology_keywords = {
            'quantitative': ['survey', 'experiment', 'statistical', 'correlation'],
            'qualitative': ['interview', 'case study', 'observation', 'narrative'],
            'mixed_methods': ['mixed', 'combined', 'integrated'],
            'systematic_review': ['systematic review', 'meta-analysis', 'literature review'],
            'computational': ['simulation', 'modeling', 'algorithm', 'machine learning']
        }
        
        for result in research_results:
            content = f"{result.title} {result.abstract}".lower()
            
            for method, keywords in methodology_keywords.items():
                if any(keyword in content for keyword in keywords):
                    methodologies[method] = methodologies.get(method, 0) + 1
        
        return methodologies
    
    def _extract_key_topics(self, results: List[SearchResult]) -> List[str]:
        """Extract key topics from a set of results"""
        topics = []
        
        for result in results:
            if result.keywords:
                topics.extend(result.keywords[:3])  # Top 3 keywords per result
        
        # Count frequency and return top topics
        topic_count = {}
        for topic in topics:
            topic_count[topic] = topic_count.get(topic, 0) + 1
        
        sorted_topics = sorted(topic_count.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5]]
    
    def _extract_methodologies(self, results: List[SearchResult]) -> List[str]:
        """Extract methodologies from a set of results"""
        methodologies = []
        
        for result in results:
            content = f"{result.title} {result.abstract}".lower()
            
            if any(keyword in content for keyword in ['survey', 'experiment']):
                methodologies.append('Quantitative')
            elif any(keyword in content for keyword in ['interview', 'case study']):
                methodologies.append('Qualitative')
            elif any(keyword in content for keyword in ['systematic review', 'meta-analysis']):
                methodologies.append('Systematic Review')
            else:
                methodologies.append('Other')
        
        return methodologies


def main():
    """Test the Analysis Agent"""
    try:
        from agents.web_research_agent import SearchResult
        
        # Create mock research results for testing
        mock_results = [
            SearchResult(
                title="Test Research Paper 1",
                authors=["Author 1", "Author 2"],
                publication_date="2023",
                abstract="This is a significant finding about machine learning in healthcare.",
                source="Test Source",
                url="http://test.com",
                doi="10.1234/test",
                keywords=["machine learning", "healthcare", "AI"],
                relevance_score=0.9,
                content_type="research_paper"
            )
        ]
        
        agent = AnalysisAgent()
        topic = "Machine Learning in Healthcare"
        
        print(f"Testing Analysis Agent with topic: {topic}")
        results = agent.analyze_research(topic, mock_results)
        
        print(f"Analysis completed: {results}")
        
    except Exception as e:
        print(f"Error testing Analysis Agent: {e}")


if __name__ == "__main__":
    main()

