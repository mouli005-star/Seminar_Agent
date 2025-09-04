# SeminarAgent 

An intelligent AI agent system designed to generate comprehensive seminar reports by conducting web research on recent academic journals and publications.

##  What It Does

SeminarAgent automatically:
- Searches for recent academic journals (last 5-10 years) based on your seminar topic
- Extracts and analyzes relevant content from multiple sources
- Generates comprehensive reports following your specific rules and requirements
- Provides proper citations and references
- Creates structured, professional-quality output

## Architecture

The system consists of four specialized AI agents:

1. **Web Research Agent** 
   - Searches Google Scholar, ResearchGate, arXiv, and other academic sources
   - Filters results by date (2014-2024) and relevance
   - Collects metadata and search results

2. **Content Extractor Agent** 
   - Extracts abstracts and key content from search results
   - Processes different journal website formats
   - Structures data for analysis

3. **Analysis Agent** 
   - Analyzes gathered content for key insights
   - Identifies trends and themes
   - Synthesizes findings for report generation

4. **Report Writer Agent** ✍
   - Generates final reports following your rules
   - Ensures proper formatting and citations
   - Creates professional-quality output

##  Quick Start

### 1. Install Dependencies
```bash
pip install -e .
```

### 2. Set Up Environment
Create a `.env` file with your OpenAI API key:
```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Configure Report Rules
Edit `config/report_rules.yaml` with your specific requirements.

### 4. Run the Agent
```bash
python main.py --topic "Your Seminar Topic"
```

## 📁 Project Structure

```
SeminarAgent/
├── agents/           # AI agent implementations
├── tools/            # Utility tools and helpers
├── config/           # Configuration files
├── knowledge/        # Stored research data
├── output/           # Generated reports
├── main.py           # Main execution script
├── pyproject.toml    # Project dependencies
└── README.md         # This file
```

## ⚙️ Configuration

### Report Rules (`config/report_rules.yaml`)
Define your specific report requirements:
- Format and structure
- Citation style
- Content sections
- Length requirements
- Special formatting rules

### Agent Configuration (`config/agent_config.yaml`)
Configure agent behavior:
- Search parameters
- Content filtering
- Analysis depth
- Output preferences

## 🔧 Customization

### Adding New Search Sources
Extend the web research capabilities by adding new academic sources in `tools/web_search_tools.py`.

### Modifying Report Format
Customize the report structure and formatting in `agents/writer_agent.py`.

### Adjusting Analysis Parameters
Fine-tune content analysis in `agents/analysis_agent.py`.

## 📊 Output

The system generates:
- **Structured Reports**: Professional-formatted seminar reports
- **Citations**: Proper academic references
- **Research Summary**: Key findings and insights
- **Source Analysis**: Evaluation of gathered materials

## 🛠️ Requirements

- Python 3.9+
- OpenAI API key
- Internet connection for web research
- Sufficient storage for research data

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:
1. Check the troubleshooting section
2. Review the configuration files
3. Check the logs in the output directory
4. Open an issue on GitHub

## 🎉 Acknowledgments

Built with:
- [CrewAI](https://github.com/joaomdmoura/crewAI) - Multi-agent orchestration
- [LangChain](https://github.com/langchain-ai/langchain) - LLM integration
- [DuckDuckGo Search](https://github.com/deedy5/duckduckgo_search) - Web search capabilities

---

**Happy Report Writing! 📚✨**

