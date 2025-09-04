"""
Coordinator Agent for SeminarAgent

This agent orchestrates the entire research and report generation process,
coordinating between the research, analysis, and writing agents.
"""

import logging
import yaml
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

from agents.web_research_agent import WebResearchAgent, SearchResult
from agents.analysis_agent import AnalysisAgent
from agents.writer_agent import WriterAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Main coordinator agent that manages the entire seminar report generation process.
    
    This agent orchestrates the workflow between research, analysis, and writing
    agents to produce comprehensive seminar reports.
    """
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        """Initialize the Coordinator Agent"""
        self.config = self._load_config(config_path)
        self.report_rules = self._load_report_rules()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            api_key=self._get_openai_key()
        )
        
        # Initialize sub-agents
        self.research_agent = WebResearchAgent(config_path)
        self.analysis_agent = AnalysisAgent(config_path)
        self.writer_agent = WriterAgent(config_path)
        
        # Initialize coordinator agent
        self.coordinator = Agent(
            role=self.config['coordinator_agent']['role'],
            goal=self.config['coordinator_agent']['goal'],
            backstory="""You are an experienced project manager and research coordinator 
            with expertise in managing complex academic research projects. You excel at 
            coordinating teams, managing timelines, and ensuring quality deliverables.""",
            verbose=True,
            allow_delegation=True,
            llm=self.llm
        )
        
        # Create output directories
        self._create_output_directories()
    
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
    
    def _load_report_rules(self, rules_path: str = "config/report_rules.yaml") -> Dict[str, Any]:
        """Load report rules from YAML file"""
        try:
            with open(rules_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.error(f"Report rules file not found: {rules_path}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing report rules file: {e}")
            return {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file loading fails"""
        return {
            'coordinator_agent': {
                'role': 'Project Manager',
                'goal': 'Coordinate research and report generation',
                'workflow': {
                    'agent_sequence': ['web_research_agent', 'analysis_agent', 'writer_agent']
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
    
    def _create_output_directories(self):
        """Create necessary output directories"""
        directories = ['output', 'knowledge', 'logs']
        for directory in directories:
            Path(directory).mkdir(exist_ok=True)
    
    def generate_seminar_report(self, topic: str, custom_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive seminar report for the given topic.
        
        Args:
            topic: The seminar topic to research and report on
            custom_rules: Optional custom rules to override default report rules
            
        Returns:
            Dictionary containing the generated report and metadata
        """
        logger.info(f"Starting seminar report generation for topic: {topic}")
        
        start_time = datetime.now()
        
        try:
            # Step 1: Research Phase
            logger.info("Phase 1: Research Phase")
            research_results = self._conduct_research(topic)
            
            if not research_results:
                raise ValueError("No research results found. Cannot proceed with report generation.")
            
            # Step 2: Analysis Phase
            logger.info("Phase 2: Analysis Phase")
            analysis_results = self._conduct_analysis(topic, research_results)
            
            # Step 3: Report Writing Phase
            logger.info("Phase 3: Report Writing Phase")
            report_results = self._generate_report(topic, research_results, analysis_results, custom_rules)
            
            # Step 4: Quality Control
            logger.info("Phase 4: Quality Control")
            quality_report = self._conduct_quality_control(report_results, research_results)
            
            # Compile final results
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()
            
            final_results = {
                'topic': topic,
                'execution_time_seconds': execution_time,
                'start_time': start_time.isoformat(),
                'end_time': end_time.isoformat(),
                'research_summary': self._create_research_summary(research_results),
                'analysis_summary': analysis_results,
                'report': report_results,
                'quality_report': quality_report,
                'metadata': {
                    'total_sources': len(research_results),
                    'report_length_words': len(report_results.get('content', '').split()),
                    'sources_by_year': self._analyze_sources_by_year(research_results),
                    'sources_by_type': self._analyze_sources_by_type(research_results)
                }
            }
            
            # Save results
            self._save_results(final_results, topic)
            
            logger.info(f"Seminar report generation completed successfully in {execution_time:.2f} seconds")
            return final_results
            
        except Exception as e:
            logger.error(f"Error during report generation: {e}")
            return {
                'error': str(e),
                'topic': topic,
                'execution_time_seconds': (datetime.now() - start_time).total_seconds(),
                'start_time': start_time.isoformat(),
                'end_time': datetime.now().isoformat()
            }
    
    def _conduct_research(self, topic: str) -> List[SearchResult]:
        """Conduct research using the research agent"""
        try:
            max_results = self.config['web_research_agent']['search']['max_results']
            research_results = self.research_agent.research_topic(topic, max_results)
            
            # Save research results
            self._save_research_results(research_results, topic)
            
            return research_results
            
        except Exception as e:
            logger.error(f"Error during research phase: {e}")
            return []
    
    def _conduct_analysis(self, topic: str, research_results: List[SearchResult]) -> Dict[str, Any]:
        """Conduct analysis using the analysis agent"""
        try:
            analysis_results = self.analysis_agent.analyze_research(topic, research_results)
            
            # Save analysis results
            self._save_analysis_results(analysis_results, topic)
            
            return analysis_results
            
        except Exception as e:
            logger.error(f"Error during analysis phase: {e}")
            return {'error': str(e)}
    
    def _generate_report(self, topic: str, research_results: List[SearchResult], 
                        analysis_results: Dict[str, Any], custom_rules: Dict[str, Any] = None) -> Dict[str, Any]:
        """Generate the final report using the writer agent"""
        try:
            # Merge custom rules with default rules
            final_rules = self.report_rules.copy()
            if custom_rules:
                final_rules.update(custom_rules)
            
            report_results = self.writer_agent.generate_report(
                topic, research_results, analysis_results, final_rules
            )
            
            # Save report
            self._save_report(report_results, topic)
            
            return report_results
            
        except Exception as e:
            logger.error(f"Error during report writing phase: {e}")
            return {'error': str(e)}
    
    def _conduct_quality_control(self, report_results: Dict[str, Any], 
                                research_results: List[SearchResult]) -> Dict[str, Any]:
        """Conduct quality control checks on the generated report"""
        try:
            quality_report = {
                'timestamp': datetime.now().isoformat(),
                'checks_performed': [],
                'issues_found': [],
                'recommendations': [],
                'overall_score': 0.0
            }
            
            # Check report structure
            structure_check = self._check_report_structure(report_results)
            quality_report['checks_performed'].append(structure_check)
            
            # Check content quality
            content_check = self._check_content_quality(report_results, research_results)
            quality_report['checks_performed'].append(content_check)
            
            # Check citation quality
            citation_check = self._check_citation_quality(report_results, research_results)
            quality_report['checks_performed'].append(citation_check)
            
            # Calculate overall score
            quality_report['overall_score'] = self._calculate_quality_score(quality_report['checks_performed'])
            
            # Generate recommendations
            quality_report['recommendations'] = self._generate_quality_recommendations(quality_report)
            
            return quality_report
            
        except Exception as e:
            logger.error(f"Error during quality control: {e}")
            return {'error': str(e)}
    
    def _check_report_structure(self, report_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if the report follows the required structure"""
        check = {
            'name': 'Report Structure Check',
            'passed': False,
            'details': [],
            'score': 0.0
        }
        
        try:
            content = report_results.get('content', '')
            required_sections = self.report_rules.get('report_structure', {}).get('required_sections', [])
            
            missing_sections = []
            for section in required_sections:
                if section.lower() not in content.lower():
                    missing_sections.append(section)
            
            if not missing_sections:
                check['passed'] = True
                check['score'] = 1.0
                check['details'].append("All required sections are present")
            else:
                check['details'].append(f"Missing sections: {', '.join(missing_sections)}")
                check['score'] = max(0.0, 1.0 - (len(missing_sections) / len(required_sections)))
            
        except Exception as e:
            check['details'].append(f"Error during structure check: {e}")
            check['score'] = 0.0
        
        return check
    
    def _check_content_quality(self, report_results: Dict[str, Any], 
                              research_results: List[SearchResult]) -> Dict[str, Any]:
        """Check the quality of report content"""
        check = {
            'name': 'Content Quality Check',
            'passed': False,
            'details': [],
            'score': 0.0
        }
        
        try:
            content = report_results.get('content', '')
            word_count = len(content.split())
            
            # Check length requirements
            min_words = self.report_rules.get('content_requirements', {}).get('length', {}).get('min_word_count', 5000)
            max_words = self.report_rules.get('content_requirements', {}).get('length', {}).get('max_word_count', 15000)
            
            if min_words <= word_count <= max_words:
                check['details'].append(f"Word count ({word_count}) is within required range ({min_words}-{max_words})")
                length_score = 1.0
            else:
                check['details'].append(f"Word count ({word_count}) is outside required range ({min_words}-{max_words})")
                length_score = 0.5
            
            # Check source coverage
            min_sources = self.report_rules.get('content_requirements', {}).get('quality', {}).get('min_sources', 15)
            if len(research_results) >= min_sources:
                check['details'].append(f"Sufficient sources ({len(research_results)} >= {min_sources})")
                source_score = 1.0
            else:
                check['details'].append(f"Insufficient sources ({len(research_results)} < {min_sources})")
                source_score = 0.5
            
            # Calculate overall score
            check['score'] = (length_score + source_score) / 2
            check['passed'] = check['score'] >= 0.7
            
        except Exception as e:
            check['details'].append(f"Error during content quality check: {e}")
            check['score'] = 0.0
        
        return check
    
    def _check_citation_quality(self, report_results: Dict[str, Any], 
                               research_results: List[SearchResult]) -> Dict[str, Any]:
        """Check the quality of citations in the report"""
        check = {
            'name': 'Citation Quality Check',
            'passed': False,
            'details': [],
            'score': 0.0
        }
        
        try:
            content = report_results.get('content', '')
            
            # Check for citation patterns
            citation_patterns = [
                r'\([A-Za-z]+\s+\d{4}\)',  # (Author Year)
                r'\[[^\]]+\]',  # [Reference]
                r'[A-Za-z]+\s+et\s+al\.\s+\d{4}'  # Author et al. Year
            ]
            
            citations_found = 0
            for pattern in citation_patterns:
                import re
                matches = re.findall(pattern, content)
                citations_found += len(matches)
            
            if citations_found > 0:
                check['details'].append(f"Found {citations_found} citations")
                citation_score = min(1.0, citations_found / len(research_results))
            else:
                check['details'].append("No citations found")
                citation_score = 0.0
            
            # Check for reference list
            if 'references' in content.lower() or 'bibliography' in content.lower():
                check['details'].append("Reference list section found")
                reference_score = 1.0
            else:
                check['details'].append("Reference list section not found")
                reference_score = 0.0
            
            # Calculate overall score
            check['score'] = (citation_score + reference_score) / 2
            check['passed'] = check['score'] >= 0.7
            
        except Exception as e:
            check['details'].append(f"Error during citation quality check: {e}")
            check['score'] = 0.0
        
        return check
    
    def _calculate_quality_score(self, checks: List[Dict[str, Any]]) -> float:
        """Calculate overall quality score from individual checks"""
        if not checks:
            return 0.0
        
        total_score = sum(check.get('score', 0.0) for check in checks)
        return total_score / len(checks)
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality check results"""
        recommendations = []
        
        for check in quality_report.get('checks_performed', []):
            if not check.get('passed', False):
                check_name = check.get('name', 'Unknown check')
                recommendations.append(f"Improve {check_name}: {', '.join(check.get('details', []))}")
        
        if quality_report.get('overall_score', 0.0) < 0.8:
            recommendations.append("Overall quality is below target. Consider revising the report.")
        
        if not recommendations:
            recommendations.append("Report meets quality standards. No major improvements needed.")
        
        return recommendations
    
    def _create_research_summary(self, research_results: List[SearchResult]) -> Dict[str, Any]:
        """Create a summary of research findings"""
        if not research_results:
            return {"error": "No research results available"}
        
        summary = {
            "total_sources": len(research_results),
            "sources_by_year": self._analyze_sources_by_year(research_results),
            "sources_by_type": self._analyze_sources_by_type(research_results),
            "top_keywords": self._extract_top_keywords(research_results),
            "research_gaps": [],
            "trends": []
        }
        
        return summary
    
    def _analyze_sources_by_year(self, research_results: List[SearchResult]) -> Dict[str, int]:
        """Analyze sources by publication year"""
        year_counts = {}
        for result in research_results:
            if result.publication_date:
                year = result.publication_date[:4]
                year_counts[year] = year_counts.get(year, 0) + 1
        return year_counts
    
    def _analyze_sources_by_type(self, research_results: List[SearchResult]) -> Dict[str, int]:
        """Analyze sources by content type"""
        type_counts = {}
        for result in research_results:
            content_type = result.content_type
            type_counts[content_type] = type_counts.get(content_type, 0) + 1
        return type_counts
    
    def _extract_top_keywords(self, research_results: List[SearchResult]) -> List[tuple]:
        """Extract top keywords from research results"""
        all_keywords = []
        for result in research_results:
            all_keywords.extend(result.keywords)
        
        # Count keyword frequency
        keyword_count = {}
        for keyword in all_keywords:
            keyword_count[keyword] = keyword_count.get(keyword, 0) + 1
        
        # Return top keywords
        return sorted(keyword_count.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _save_results(self, results: Dict[str, Any], topic: str):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        
        # Save main results
        results_file = f"output/seminar_report_{safe_topic}_{timestamp}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Save report content separately
        if 'report' in results and 'content' in results['report']:
            report_file = f"output/seminar_report_{safe_topic}_{timestamp}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(results['report']['content'])
        
        logger.info(f"Results saved to {results_file} and {report_file}")
    
    def _save_research_results(self, results: List[SearchResult], topic: str):
        """Save research results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        
        research_file = f"knowledge/research_{safe_topic}_{timestamp}.json"
        
        # Convert SearchResult objects to dictionaries
        research_data = []
        for result in results:
            research_data.append({
                'title': result.title,
                'authors': result.authors,
                'publication_date': result.publication_date,
                'abstract': result.abstract,
                'source': result.source,
                'url': result.url,
                'doi': result.doi,
                'keywords': result.keywords,
                'relevance_score': result.relevance_score,
                'content_type': result.content_type
            })
        
        with open(research_file, 'w', encoding='utf-8') as f:
            json.dump(research_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Research results saved to {research_file}")
    
    def _save_analysis_results(self, results: Dict[str, Any], topic: str):
        """Save analysis results to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        
        analysis_file = f"knowledge/analysis_{safe_topic}_{timestamp}.json"
        
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Analysis results saved to {analysis_file}")
    
    def _save_report(self, results: Dict[str, Any], topic: str):
        """Save report to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')
        
        report_file = f"output/report_{safe_topic}_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"Report saved to {report_file}")


def main():
    """Test the Coordinator Agent"""
    try:
        coordinator = CoordinatorAgent()
        topic = "Machine Learning Applications in Healthcare"
        
        print(f"Testing Coordinator Agent with topic: {topic}")
        print("Starting seminar report generation...")
        
        results = coordinator.generate_seminar_report(topic)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"\nReport generation completed successfully!")
            print(f"Execution time: {results['execution_time_seconds']:.2f} seconds")
            print(f"Total sources: {results['metadata']['total_sources']}")
            print(f"Report length: {results['metadata']['report_length_words']} words")
            print(f"Quality score: {results['quality_report']['overall_score']:.2f}")
            
            print(f"\nReport saved to output directory")
            print(f"Research data saved to knowledge directory")
        
    except Exception as e:
        print(f"Error testing Coordinator Agent: {e}")


if __name__ == "__main__":
    main()

