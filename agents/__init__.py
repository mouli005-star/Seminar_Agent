"""
SeminarAgent - AI-powered Seminar Report Generator

This package contains the core AI agents for research, analysis, and report generation.
"""

from .web_research_agent import WebResearchAgent, SearchResult
from .analysis_agent import AnalysisAgent
from .writer_agent import WriterAgent
from .coordinator import CoordinatorAgent

__version__ = "0.1.0"
__author__ = "SeminarAgent Team"

__all__ = [
    "WebResearchAgent",
    "SearchResult", 
    "AnalysisAgent",
    "WriterAgent",
    "CoordinatorAgent"
]
