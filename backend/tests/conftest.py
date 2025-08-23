import pytest
import sys
import os
from typing import Dict, Any, List
from unittest.mock import MagicMock, Mock

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import Course, Lesson, CourseChunk
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    return Course(
        title="Building Towards Computer Use with Anthropic",
        course_link="https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/",
        instructor="Colt Steele",
        lessons=[
            Lesson(lesson_number=0, title="Introduction", lesson_link="https://example.com/lesson0"),
            Lesson(lesson_number=1, title="Getting Started", lesson_link="https://example.com/lesson1"),
            Lesson(lesson_number=2, title="Advanced Topics", lesson_link=None)
        ]
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 0 content: Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=0,
            chunk_index=0
        ),
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 1 content: In this lesson, you'll learn about the API basics.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Course Building Towards Computer Use with Anthropic Lesson 2 content: Advanced topics include prompt engineering and tool use.",
            course_title="Building Towards Computer Use with Anthropic",
            lesson_number=2,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = MagicMock()
    
    # Default search results
    mock_store.search.return_value = SearchResults(
        documents=[
            "Welcome to Building Toward Computer Use with Anthropic. Built in partnership with Anthropic.",
            "In this lesson, you'll learn about the API basics."
        ],
        metadata=[
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 0},
            {"course_title": "Building Towards Computer Use with Anthropic", "lesson_number": 1}
        ],
        distances=[0.1, 0.2],
        error=None
    )
    
    # Course info response
    mock_store.get_course_info.return_value = {
        'title': 'Building Towards Computer Use with Anthropic',
        'link': 'https://www.deeplearning.ai/short-courses/building-toward-computer-use-with-anthropic/',
        'instructor': 'Colt Steele',
        'lessons': [
            {'number': 0, 'title': 'Introduction', 'link': 'https://example.com/lesson0'},
            {'number': 1, 'title': 'Getting Started', 'link': 'https://example.com/lesson1'}
        ]
    }
    
    # Lesson link response
    mock_store.get_lesson_link.return_value = "https://example.com/lesson0"
    
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = MagicMock()
    
    # Mock message response
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="This is a test response about the course content.")]
    mock_response.stop_reason = "end_turn"
    
    mock_client.messages.create.return_value = mock_response
    
    return mock_client


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing"""
    from dataclasses import dataclass
    
    @dataclass
    class MockConfig:
        ANTHROPIC_API_KEY: str = "test-api-key"
        ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
        EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
        CHUNK_SIZE: int = 800
        CHUNK_OVERLAP: int = 100
        MAX_RESULTS: int = 5
        MAX_HISTORY: int = 2
        CHROMA_PATH: str = "./test_chroma_db"
    
    return MockConfig()


@pytest.fixture
def search_results_empty():
    """Create empty search results"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error=None
    )


@pytest.fixture
def search_results_with_error():
    """Create search results with error"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[],
        error="Search error: Connection failed"
    )