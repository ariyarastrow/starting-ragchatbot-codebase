import os
import sys
from unittest.mock import MagicMock, mock_open, patch

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from vector_store import SearchResults


class TestRAGSystemIntegration:
    """Integration tests for RAG system"""

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_initialization(
        self, mock_session, mock_processor, mock_ai, mock_vector, mock_config
    ):
        """Test RAG system initialization"""
        rag = RAGSystem(mock_config)

        # Verify components were initialized
        mock_processor.assert_called_once_with(
            mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP
        )
        mock_vector.assert_called_once_with(
            mock_config.CHROMA_PATH,
            mock_config.EMBEDDING_MODEL,
            mock_config.MAX_RESULTS,
        )
        mock_ai.assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )
        mock_session.assert_called_once_with(mock_config.MAX_HISTORY)

        # Verify tools were registered
        assert len(rag.tool_manager.tools) == 2
        assert "search_course_content" in rag.tool_manager.tools
        assert "get_course_outline" in rag.tool_manager.tools

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_content_question(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_config,
    ):
        """Test querying with a content-related question"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup search results
        mock_vector.search.return_value = SearchResults(
            documents=[
                "Prompt caching retains processing results between invocations."
            ],
            metadata=[
                {"course_title": "Building Towards Computer Use", "lesson_number": 5}
            ],
            distances=[0.1],
            error=None,
        )

        mock_vector.get_lesson_link.return_value = "https://example.com/lesson5"

        # Setup AI response
        mock_ai.generate_response.return_value = (
            "Prompt caching is a technique that retains processing results."
        )

        # Create RAG system and query
        rag = RAGSystem(mock_config)
        response, sources = rag.query(
            "What is prompt caching?", session_id="test-session"
        )

        # Verify AI was called with tools
        mock_ai.generate_response.assert_called_once()
        call_args = mock_ai.generate_response.call_args

        assert "What is prompt caching?" in call_args[1]["query"]
        assert call_args[1]["tools"] is not None
        assert len(call_args[1]["tools"]) == 2  # Both search and outline tools
        assert call_args[1]["tool_manager"] is not None

        # Verify response
        assert "Prompt caching" in response

        # Verify session was updated
        mock_session.add_exchange.assert_called_once_with(
            "test-session", "What is prompt caching?", response
        )

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_outline_question(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_config,
    ):
        """Test querying for course outline"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup course info
        mock_vector.get_course_info.return_value = {
            "title": "Building Towards Computer Use",
            "link": "https://example.com/course",
            "instructor": "Colt Steele",
            "lessons": [
                {"number": 0, "title": "Introduction"},
                {"number": 1, "title": "Getting Started"},
            ],
        }

        # Setup AI response
        mock_ai.generate_response.return_value = (
            "Course outline: Introduction, Getting Started..."
        )

        # Create RAG system and query
        rag = RAGSystem(mock_config)
        response, sources = rag.query("Show me the course outline")

        # Verify tools were provided to AI
        call_args = mock_ai.generate_response.call_args[1]
        tool_defs = call_args["tools"]

        # Check that get_course_outline tool is available
        outline_tool = next(
            (t for t in tool_defs if t["name"] == "get_course_outline"), None
        )
        assert outline_tool is not None
        assert "course_name" in outline_tool["input_schema"]["properties"]

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_with_tool_execution(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_config,
    ):
        """Test full tool execution flow"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session
        mock_session.get_conversation_history.return_value = None

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Create RAG system
        rag = RAGSystem(mock_config)

        # Setup vector store to return search results when tool is executed
        mock_vector.search.return_value = SearchResults(
            documents=["Content about computer use."],
            metadata=[{"course_title": "Test Course", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        # Execute search tool directly through tool manager
        result = rag.tool_manager.execute_tool(
            "search_course_content", query="computer use"
        )

        # Verify tool execution worked
        assert "Content about computer use" in result
        assert "[Test Course - Lesson 1]" in result

        # Verify sources were tracked
        sources = rag.tool_manager.get_last_sources()
        assert len(sources) > 0
        assert sources[0]["text"] == "Test Course - Lesson 1"

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_without_session(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_config,
    ):
        """Test query without session ID"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai
        mock_ai.generate_response.return_value = "Test response"

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Create RAG system and query without session
        rag = RAGSystem(mock_config)
        response, sources = rag.query("Test question")

        # Verify no session operations occurred
        mock_session.get_conversation_history.assert_not_called()
        mock_session.add_exchange.assert_not_called()

        assert response == "Test response"

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_with_error_in_search(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_config,
    ):
        """Test handling search errors"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup search to return error
        mock_vector.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection failed"
        )

        # Create RAG system
        rag = RAGSystem(mock_config)

        # Execute search tool directly
        result = rag.tool_manager.execute_tool(
            "search_course_content", query="test query"
        )

        # Should return the error message
        assert result == "Database connection failed"
        assert rag.tool_manager.get_last_sources() == []

    @patch("rag_system.os.path.exists")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_document(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_exists,
        mock_config,
    ):
        """Test adding a course document"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup document processing
        mock_course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Test Instructor",
            lessons=[Lesson(lesson_number=0, title="Intro")],
        )

        mock_chunks = [
            CourseChunk(
                content="Test content",
                course_title="Test Course",
                lesson_number=0,
                chunk_index=0,
            )
        ]

        mock_processor.process_course_document.return_value = (mock_course, mock_chunks)

        # Create RAG system and add document
        rag = RAGSystem(mock_config)
        course, num_chunks = rag.add_course_document("test.txt")

        # Verify processing
        mock_processor.process_course_document.assert_called_once_with("test.txt")
        mock_vector.add_course_metadata.assert_called_once_with(mock_course)
        mock_vector.add_course_content.assert_called_once_with(mock_chunks)

        assert course == mock_course
        assert num_chunks == 1

    @patch("rag_system.os.listdir")
    @patch("rag_system.os.path.isfile")
    @patch("rag_system.os.path.exists")
    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_folder(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_exists,
        mock_isfile,
        mock_listdir,
        mock_config,
    ):
        """Test adding multiple course documents from folder"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector
        mock_vector.get_existing_course_titles.return_value = []

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup file system mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.txt", "image.jpg"]
        mock_isfile.side_effect = lambda x: True

        # Setup document processing
        mock_course1 = Course(title="Course 1", lessons=[])
        mock_course2 = Course(title="Course 2", lessons=[])
        mock_chunks1 = [
            CourseChunk(content="C1", course_title="Course 1", chunk_index=0)
        ]
        mock_chunks2 = [
            CourseChunk(content="C2", course_title="Course 2", chunk_index=0)
        ]

        mock_processor.process_course_document.side_effect = [
            (mock_course1, mock_chunks1),
            (mock_course2, mock_chunks2),
        ]

        # Create RAG system and add folder
        rag = RAGSystem(mock_config)
        num_courses, num_chunks = rag.add_course_folder("test_folder")

        # Verify processing
        assert mock_processor.process_course_document.call_count == 2
        assert mock_vector.add_course_metadata.call_count == 2
        assert mock_vector.add_course_content.call_count == 2

        assert num_courses == 2
        assert num_chunks == 2

    @patch("rag_system.VectorStore")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_source_tracking(
        self,
        mock_session_class,
        mock_processor_class,
        mock_ai_class,
        mock_vector_class,
        mock_config,
    ):
        """Test that sources are properly tracked and reset"""
        # Setup mocks
        mock_vector = MagicMock()
        mock_vector_class.return_value = mock_vector

        mock_ai = MagicMock()
        mock_ai_class.return_value = mock_ai
        mock_ai.generate_response.return_value = "Test response"

        mock_session = MagicMock()
        mock_session_class.return_value = mock_session

        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor

        # Setup search results
        mock_vector.search.return_value = SearchResults(
            documents=["Doc 1"],
            metadata=[{"course_title": "Course 1", "lesson_number": 1}],
            distances=[0.1],
            error=None,
        )

        mock_vector.get_lesson_link.return_value = "https://example.com/lesson1"

        # Create RAG system
        rag = RAGSystem(mock_config)

        # Execute tool to generate sources
        rag.tool_manager.execute_tool("search_course_content", query="test")

        # Get sources before query
        sources = rag.tool_manager.get_last_sources()
        assert len(sources) == 1

        # Execute query (should reset sources after retrieval)
        response, query_sources = rag.query("Test question")

        # Verify sources were retrieved and then reset
        assert query_sources == sources
        assert rag.tool_manager.get_last_sources() == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
