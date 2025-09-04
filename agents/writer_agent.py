"""
Writer Agent for SeminarAgent

This agent generates comprehensive seminar reports based on research findings
and analysis results, following specified formatting and citation rules.
"""

import logging
from typing import List, Dict, Any
from crewai import Agent, Task
from langchain_openai import ChatOpenAI

from agents.web_research_agent import SearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WriterAgent:
    """
    Agent responsible for generating comprehensive seminar reports.
    
    This agent takes research findings and analysis results to create
    well-structured, properly formatted academic reports.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """Initialize the Writer Agent"""
        self.config = self._load_config(config_path)
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=self._get_openai_key()
        )
        
        # Initialize CrewAI agent
        self.agent = Agent(
            role=self.config['report_writer_agent']['role'],
            goal=self.config['report_writer_agent']['goal'],
            backstory="""You are an expert academic writer with years of experience 
            in creating comprehensive research reports and seminar materials. You excel 
            at structuring complex information, maintaining academic writing standards, 
            and ensuring proper citations and formatting.""",
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
            'report_writer_agent': {
                'role': 'Academic Report Writer',
                'goal': 'Generate comprehensive seminar reports'
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
    
    def generate_report(self, topic: str, research_results: List[SearchResult], 
                       analysis_results: Dict[str, Any], report_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive seminar report.
        
        Args:
            topic: The seminar topic
            research_results: List of research sources
            analysis_results: Analysis findings and insights
            report_rules: Rules and requirements for the report
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        logger.info(f"Starting report generation for topic: {topic}")
        
        try:
            # Create report writing task
            report_task = Task(
                description=f"""
                Generate a comprehensive seminar report for the topic: "{topic}"
                
                Research Sources: {len(research_results)} sources
                Analysis Results: Available
                Report Rules: Specified
                
                Report Requirements:
                1. Follow the specified structure and format
                2. Include all required sections
                3. Use proper academic writing style
                4. Include proper citations and references
                5. Ensure content meets length requirements
                6. Maintain professional tone and language
                
                Structure Requirements:
                - Title page and table of contents
                - Executive summary
                - Introduction with clear objectives
                - Literature review based on research sources
                - Methodology and approach
                - Findings and analysis
                - Discussion and implications
                - Conclusion and recommendations
                - References and appendices
                
                Content Requirements:
                - Minimum word count: {report_rules.get('content_requirements', {}).get('length', {}).get('min_word_count', 5000)}
                - Maximum word count: {report_rules.get('content_requirements', {}).get('length', {}).get('max_word_count', 15000)}
                - Citation style: {report_rules.get('citations', {}).get('style', 'APA')}
                - Focus on recent sources (2014-2024)
                
                Output: A comprehensive, well-structured seminar report in markdown format
                with proper citations, references, and professional formatting.
                """,
                agent=self.agent,
                expected_output="""
                A complete seminar report in markdown format with all required sections,
                proper citations, and professional academic writing standards.
                """
            )
            
            # Execute report writing task
            result = report_task.execute()
            logger.info("Report writing task completed successfully")
            
            # Process and structure the report
            report_results = self._process_report_results(result, topic, research_results, report_rules)
            
            return report_results
            
        except Exception as e:
            logger.error(f"Error during report generation: {e}")
            return {
                'error': str(e),
                'content': 'Report generation failed due to error',
                'metadata': {
                    'topic': topic,
                    'word_count': 0,
                    'sections': [],
                    'citations': 0
                }
            }
    
    def _process_report_results(self, result: str, topic: str, 
                               research_results: List[SearchResult], 
                               report_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Process the report results from the agent's output"""
        try:
            # This is a simplified processor - in practice, you'd want more sophisticated parsing
            # based on the actual output format from your LLM
            
            # Extract content and metadata
            content = result if result else "Report content not available"
            word_count = len(content.split())
            
            # Identify sections
            sections = self._identify_sections(content)
            
            # Count citations
            citations = self._count_citations(content)
            
            # Generate references
            references = self._generate_references(research_results, report_rules)
            
            report_results = {
                'content': content,
                'metadata': {
                    'topic': topic,
                    'word_count': word_count,
                    'sections': sections,
                    'citations': citations,
                    'references_count': len(references),
                    'generation_timestamp': '2024-01-01T00:00:00Z'
                },
                'references': references,
                'format': report_rules.get('output_formats', {}).get('primary', 'markdown'),
                'citation_style': report_rules.get('citations', {}).get('style', 'APA')
            }
            
            return report_results
            
        except Exception as e:
            logger.error(f"Error processing report results: {e}")
            return {'error': str(e)}
    
    def _identify_sections(self, content: str) -> List[str]:
        """Identify sections in the report content"""
        sections = []
        
        # Common section headers
        section_patterns = [
            r'^#\s+(.+)$',  # Markdown H1
            r'^##\s+(.+)$',  # Markdown H2
            r'^###\s+(.+)$',  # Markdown H3
        ]
        
        import re
        lines = content.split('\n')
        
        for line in lines:
            for pattern in section_patterns:
                match = re.match(pattern, line)
                if match:
                    sections.append(match.group(1).strip())
                    break
        
        return sections[:10]  # Limit to first 10 sections
    
    def _count_citations(self, content: str) -> int:
        """Count citations in the report content"""
        import re
        
        # Citation patterns
        citation_patterns = [
            r'\([A-Za-z]+\s+\d{4}\)',  # (Author Year)
            r'\[[^\]]+\]',  # [Reference]
            r'[A-Za-z]+\s+et\s+al\.\s+\d{4}'  # Author et al. Year
        ]
        
        total_citations = 0
        for pattern in citation_patterns:
            matches = re.findall(pattern, content)
            total_citations += len(matches)
        
        return total_citations
    
    def _generate_references(self, research_results: List[SearchResult], 
                           report_rules: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate formatted references from research results"""
        references = []
        citation_style = report_rules.get('citations', {}).get('style', 'APA')
        
        for i, result in enumerate(research_results, 1):
            reference = self._format_reference(result, citation_style, i)
            references.append(reference)
        
        return references
    
    def _format_reference(self, result: SearchResult, citation_style: str, index: int) -> Dict[str, str]:
        """Format a single reference according to the specified citation style"""
        if citation_style == 'APA':
            # APA format: Author, A. A. (Year). Title. Journal, Volume(Issue), Pages.
            authors = ', '.join(result.authors[:2])  # First two authors
            if len(result.authors) > 2:
                authors += ' et al.'
            
            title = result.title
            year = result.publication_date[:4] if result.publication_date else 'n.d.'
            journal = result.source if result.source else 'Unknown Journal'
            
            reference_text = f"{authors} ({year}). {title}. {journal}."
            
        elif citation_style == 'MLA':
            # MLA format: Author, A. A. "Title." Journal, vol. Volume, no. Issue, Year, pp. Pages.
            authors = ', '.join(result.authors[:2])
            if len(result.authors) > 2:
                authors += ' et al.'
            
            title = result.title
            year = result.publication_date[:4] if result.publication_date else 'n.d.'
            journal = result.source if result.source else 'Unknown Journal'
            
            reference_text = f'{authors}. "{title}." {journal}, {year}.'
            
        else:
            # Default format
            reference_text = f"{result.title} - {', '.join(result.authors)} ({result.publication_date})"
        
        return {
            'index': index,
            'citation': f"({result.authors[0] if result.authors else 'Unknown'} {result.publication_date[:4] if result.publication_date else 'n.d.'})",
            'reference': reference_text,
            'url': result.url,
            'doi': result.doi
        }
    
    def validate_report(self, report_results: Dict[str, Any], report_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the generated report against requirements"""
        validation_results = {
            'passed': True,
            'issues': [],
            'warnings': [],
            'score': 0.0
        }
        
        try:
            content = report_results.get('content', '')
            metadata = report_results.get('metadata', {})
            
            # Check word count
            word_count = metadata.get('word_count', 0)
            min_words = report_rules.get('content_requirements', {}).get('length', {}).get('min_word_count', 5000)
            max_words = report_rules.get('content_requirements', {}).get('length', {}).get('max_word_count', 15000)
            
            if word_count < min_words:
                validation_results['issues'].append(f"Word count ({word_count}) below minimum ({min_words})")
                validation_results['passed'] = False
            elif word_count > max_words:
                validation_results['warnings'].append(f"Word count ({word_count}) above maximum ({max_words})")
            
            # Check sections
            required_sections = report_rules.get('report_structure', {}).get('required_sections', [])
            sections = metadata.get('sections', [])
            
            missing_sections = []
            for section in required_sections:
                if not any(section.lower() in s.lower() for s in sections):
                    missing_sections.append(section)
            
            if missing_sections:
                validation_results['issues'].append(f"Missing sections: {', '.join(missing_sections)}")
                validation_results['passed'] = False
            
            # Check citations
            citations = metadata.get('citations', 0)
            if citations == 0:
                validation_results['warnings'].append("No citations found in the report")
            
            # Calculate score
            total_checks = 3
            passed_checks = 0
            
            if word_count >= min_words:
                passed_checks += 1
            if not missing_sections:
                passed_checks += 1
            if citations > 0:
                passed_checks += 1
            
            validation_results['score'] = passed_checks / total_checks
            
        except Exception as e:
            validation_results['issues'].append(f"Validation error: {e}")
            validation_results['passed'] = False
            validation_results['score'] = 0.0
        
        return validation_results


def main():
    """Test the Writer Agent"""
    try:
        from agents.web_research_agent import SearchResult
        
        # Create mock research results for testing
        mock_results = [
            SearchResult(
                title="Test Research Paper 1",
                authors=["Author 1", "Author 2"],
                publication_date="2023",
                abstract="This is a test abstract about machine learning in healthcare.",
                source="Test Journal",
                url="http://test.com",
                doi="10.1234/test",
                keywords=["machine learning", "healthcare", "AI"],
                relevance_score=0.9,
                content_type="research_paper"
            )
        ]
        
        # Mock analysis results
        mock_analysis = {
            'key_findings': ['Finding 1', 'Finding 2'],
            'trends': [{'year': '2023', 'count': 1}],
            'research_gaps': ['Gap 1']
        }
        
        # Mock report rules
        mock_rules = {
            'content_requirements': {
                'length': {'min_word_count': 1000, 'max_word_count': 5000}
            },
            'citations': {'style': 'APA'},
            'output_formats': {'primary': 'markdown'}
        }
        
        agent = WriterAgent()
        topic = "Machine Learning in Healthcare"
        
        print(f"Testing Writer Agent with topic: {topic}")
        results = agent.generate_report(topic, mock_results, mock_analysis, mock_rules)
        
        print(f"Report generation completed: {results}")
        
        # Validate report
        validation = agent.validate_report(results, mock_rules)
        print(f"Validation results: {validation}")
        
    except Exception as e:
        print(f"Error testing Writer Agent: {e}")


if __name__ == "__main__":
    main()

