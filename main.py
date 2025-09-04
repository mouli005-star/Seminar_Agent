#!/usr/bin/env python3
"""
Main execution script for SeminarAgent

This script provides a command-line interface for running the seminar report
generation system. It handles command-line arguments and orchestrates the
entire process.
"""

import argparse
import logging
import sys
import os
from pathlib import Path
from datetime import datetime

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent))

from agents.coordinator import CoordinatorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/seminar_agent.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_environment():
    """Set up the environment and check requirements"""
    logger.info("Setting up SeminarAgent environment...")
    
    # Check if .env file exists
    env_file = Path('.env')
    if not env_file.exists():
        logger.warning(".env file not found. Creating template...")
        create_env_template()
    
    # Check OpenAI API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        logger.error("Please set your OpenAI API key in the .env file")
        return False
    
    # Create necessary directories
    directories = ['output', 'knowledge', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    
    logger.info("Environment setup completed successfully")
    return True


def create_env_template():
    """Create a template .env file"""
    env_content = """# SeminarAgent Environment Configuration
# Add your OpenAI API key below
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Customize OpenAI model
# OPENAI_MODEL=gpt-4

# Optional: Set custom temperature for LLM responses
# OPENAI_TEMPERATURE=0.1

# Optional: Set maximum execution time (in seconds)
# MAX_EXECUTION_TIME=3600
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    logger.info("Created .env template file. Please add your OpenAI API key.")


def validate_topic(topic: str) -> bool:
    """Validate the seminar topic"""
    if not topic or len(topic.strip()) < 5:
        logger.error("Topic must be at least 5 characters long")
        return False
    
    if len(topic) > 200:
        logger.error("Topic is too long (maximum 200 characters)")
        return False
    
    return True


def run_seminar_agent(topic: str, custom_rules: dict = None, verbose: bool = False):
    """Run the seminar agent with the given topic"""
    try:
        logger.info(f"Starting SeminarAgent for topic: {topic}")
        
        # Initialize coordinator agent
        coordinator = CoordinatorAgent()
        
        # Generate report
        start_time = datetime.now()
        results = coordinator.generate_seminar_report(topic, custom_rules)
        end_time = datetime.now()
        
        # Display results
        display_results(results, start_time, end_time, verbose)
        
        return results
        
    except Exception as e:
        logger.error(f"Error running SeminarAgent: {e}")
        print(f"❌ Error: {e}")
        return None


def display_results(results: dict, start_time: datetime, end_time: datetime, verbose: bool):
    """Display the results of the seminar report generation"""
    if 'error' in results:
        print(f"\n❌ Report generation failed: {results['error']}")
        return
    
    execution_time = results.get('execution_time_seconds', 0)
    
    print("\n" + "="*60)
    print("🎉 SEMINAR REPORT GENERATION COMPLETED SUCCESSFULLY!")
    print("="*60)
    
    print(f"\n📚 Topic: {results['topic']}")
    print(f"⏱️  Execution Time: {execution_time:.2f} seconds")
    print(f"🕐 Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🕐 Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Research summary
    research_summary = results.get('research_summary', {})
    if research_summary:
        print(f"\n🔍 Research Summary:")
        print(f"   • Total Sources: {research_summary.get('total_sources', 0)}")
        
        sources_by_year = research_summary.get('sources_by_year', {})
        if sources_by_year:
            print(f"   • Sources by Year: {dict(sorted(sources_by_year.items()))}")
        
        sources_by_type = research_summary.get('sources_by_type', {})
        if sources_by_type:
            print(f"   • Sources by Type: {sources_by_type}")
    
    # Report metadata
    metadata = results.get('metadata', {})
    if metadata:
        print(f"\n📄 Report Details:")
        print(f"   • Word Count: {metadata.get('report_length_words', 0):,}")
        print(f"   • Total Sources: {metadata.get('total_sources', 0)}")
    
    # Quality report
    quality_report = results.get('quality_report', {})
    if quality_report:
        overall_score = quality_report.get('overall_score', 0)
        print(f"\n✅ Quality Assessment:")
        print(f"   • Overall Score: {overall_score:.2f}/1.00")
        
        if overall_score >= 0.8:
            print("   • Status: 🟢 Excellent Quality")
        elif overall_score >= 0.6:
            print("   • Status: 🟡 Good Quality")
        else:
            print("   • Status: 🔴 Needs Improvement")
        
        recommendations = quality_report.get('recommendations', [])
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
    
    # File locations
    print(f"\n📁 Output Files:")
    print(f"   • Report: output/seminar_report_*.md")
    print(f"   • Data: output/seminar_report_*.json")
    print(f"   • Research: knowledge/research_*.json")
    print(f"   • Analysis: knowledge/analysis_*.json")
    
    if verbose:
        print(f"\n🔍 Detailed Results:")
        print(f"   • Full results saved to: output/seminar_report_*.json")
        print(f"   • Logs available in: logs/seminar_agent.log")
    
    print("\n" + "="*60)


def main():
    """Main entry point for the SeminarAgent CLI"""
    parser = argparse.ArgumentParser(
        description="SeminarAgent - AI-powered seminar report generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py "Machine Learning in Healthcare"
  python main.py "Climate Change Impact on Agriculture" --verbose
  python main.py "Quantum Computing Applications" --custom-rules rules.json

For more information, visit: https://github.com/yourusername/seminar-agent
        """
    )
    
    parser.add_argument(
        'topic',
        help='The seminar topic to research and report on'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--custom-rules',
        help='Path to custom report rules JSON file'
    )
    
    parser.add_argument(
        '--max-sources',
        type=int,
        default=20,
        help='Maximum number of sources to research (default: 20)'
    )
    
    parser.add_argument(
        '--output-format',
        choices=['markdown', 'pdf', 'docx'],
        default='markdown',
        help='Output format for the report (default: markdown)'
    )
    
    parser.add_argument(
        '--citation-style',
        choices=['APA', 'MLA', 'Chicago', 'IEEE'],
        default='APA',
        help='Citation style for the report (default: APA)'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("🤖 SeminarAgent - AI-powered Seminar Report Generator")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed. Please check the configuration.")
        sys.exit(1)
    
    # Validate topic
    if not validate_topic(args.topic):
        print("❌ Invalid topic. Please provide a valid seminar topic.")
        sys.exit(1)
    
    # Prepare custom rules if provided
    custom_rules = {}
    if args.custom_rules:
        try:
            import json
            with open(args.custom_rules, 'r') as f:
                custom_rules = json.load(f)
            logger.info(f"Loaded custom rules from {args.custom_rules}")
        except Exception as e:
            logger.error(f"Error loading custom rules: {e}")
            print(f"❌ Error loading custom rules: {e}")
            sys.exit(1)
    
    # Add command-line overrides to custom rules
    if args.max_sources:
        custom_rules['max_sources'] = args.max_sources
    
    if args.output_format:
        custom_rules['output_format'] = args.output_format
    
    if args.citation_style:
        custom_rules['citation_style'] = args.citation_style
    
    # Run the seminar agent
    print(f"\n🚀 Starting research and report generation for topic:")
    print(f"   '{args.topic}'")
    print("\nThis may take several minutes depending on the complexity...")
    
    results = run_seminar_agent(args.topic, custom_rules, args.verbose)
    
    if results and 'error' not in results:
        print("\n🎯 Report generation completed successfully!")
        print("Check the output directory for your generated report.")
        sys.exit(0)
    else:
        print("\n❌ Report generation failed.")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)

