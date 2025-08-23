import pytest
from unittest.mock import MagicMock, patch
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from search_tools import CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test suite for CourseSearchTool"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseSearchTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["type"] == "object"
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]
        assert definition["input_schema"]["required"] == ["query"]
    
    def test_execute_basic_query(self, mock_vector_store):
        """Test basic query execution without filters"""
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(query="What is computer use?")
        
        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="What is computer use?",
            course_name=None,
            lesson_number=None
        )
        
        # Check result formatting
        assert "Building Towards Computer Use with Anthropic" in result
        assert "Welcome to Building Toward Computer Use" in result
        assert len(tool.last_sources) == 2
    
    def test_execute_with_course_filter(self, mock_vector_store):
        """Test query execution with course name filter"""
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(
            query="What is prompt caching?",
            course_name="Building Towards Computer Use"
        )
        
        # Verify vector store was called with course filter
        mock_vector_store.search.assert_called_once_with(
            query="What is prompt caching?",
            course_name="Building Towards Computer Use",
            lesson_number=None
        )
        
        assert "[Building Towards Computer Use with Anthropic" in result
    
    def test_execute_with_lesson_filter(self, mock_vector_store):
        """Test query execution with lesson number filter"""
        tool = CourseSearchTool(mock_vector_store)
        
        result = tool.execute(
            query="What topics are covered?",
            course_name="Building Towards Computer Use",
            lesson_number=1
        )
        
        # Verify vector store was called with both filters
        mock_vector_store.search.assert_called_once_with(
            query="What topics are covered?",
            course_name="Building Towards Computer Use",
            lesson_number=1
        )
    
    def test_execute_with_error(self, mock_vector_store):
        """Test execution when vector store returns an error"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Search error: Connection failed"
        )
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test query")
        
        assert result == "Search error: Connection failed"
        assert tool.last_sources == []
    
    def test_execute_with_empty_results(self, mock_vector_store):
        """Test execution when no results are found"""
        mock_vector_store.search.return_value = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error=None
        )
        
        tool = CourseSearchTool(mock_vector_store)
        
        # Test without filters
        result = tool.execute(query="nonexistent content")
        assert result == "No relevant content found."
        
        # Test with course filter
        result = tool.execute(query="test", course_name="Test Course")
        assert result == "No relevant content found in course 'Test Course'."
        
        # Test with lesson filter
        result = tool.execute(query="test", lesson_number=5)
        assert result == "No relevant content found in lesson 5."
        
        # Test with both filters
        result = tool.execute(query="test", course_name="Test Course", lesson_number=5)
        assert result == "No relevant content found in course 'Test Course' in lesson 5."
    
    def test_format_results_with_links(self, mock_vector_store):
        """Test result formatting with lesson links"""
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/lesson0",
            "https://example.com/lesson1"
        ]
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")
        
        # Check that sources have links
        assert len(tool.last_sources) == 2
        assert tool.last_sources[0]["text"] == "Building Towards Computer Use with Anthropic - Lesson 0"
        assert tool.last_sources[0]["link"] == "https://example.com/lesson0"
        assert tool.last_sources[1]["text"] == "Building Towards Computer Use with Anthropic - Lesson 1"
        assert tool.last_sources[1]["link"] == "https://example.com/lesson1"
    
    def test_format_results_without_lesson_numbers(self, mock_vector_store):
        """Test formatting when metadata doesn't include lesson numbers"""
        mock_vector_store.search.return_value = SearchResults(
            documents=["Content without lesson"],
            metadata=[{"course_title": "Test Course"}],
            distances=[0.1],
            error=None
        )
        
        tool = CourseSearchTool(mock_vector_store)
        result = tool.execute(query="test")
        
        assert "[Test Course]" in result
        assert "Lesson" not in result
        assert tool.last_sources[0]["text"] == "Test Course"
        assert tool.last_sources[0]["link"] is None


class TestCourseOutlineTool:
    """Test suite for CourseOutlineTool"""
    
    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is correctly structured"""
        tool = CourseOutlineTool(mock_vector_store)
        definition = tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_name"]
    
    def test_execute_with_valid_course(self, mock_vector_store):
        """Test getting outline for existing course"""
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Building Towards Computer Use")
        
        # Verify course info was requested
        mock_vector_store.get_course_info.assert_called_once_with("Building Towards Computer Use")
        
        # Check formatted output
        assert "Course Title: Building Towards Computer Use with Anthropic" in result
        assert "Course Link: https://www.deeplearning.ai" in result
        assert "Course Instructor: Colt Steele" in result
        assert "Lesson 0: Introduction" in result
        assert "Lesson 1: Getting Started" in result
        
        # Check sources
        assert len(tool.last_sources) == 1
        assert tool.last_sources[0]["text"] == "Building Towards Computer Use with Anthropic - Course Outline"
    
    def test_execute_with_nonexistent_course(self, mock_vector_store):
        """Test getting outline for non-existent course"""
        mock_vector_store.get_course_info.return_value = None
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Nonexistent Course")
        
        assert result == "No course found matching 'Nonexistent Course'"
        assert tool.last_sources == []
    
    def test_execute_without_instructor(self, mock_vector_store):
        """Test outline formatting when instructor is missing"""
        mock_vector_store.get_course_info.return_value = {
            'title': 'Test Course',
            'link': None,
            'instructor': None,
            'lessons': []
        }
        
        tool = CourseOutlineTool(mock_vector_store)
        result = tool.execute(course_name="Test Course")
        
        assert "Course Title: Test Course" in result
        assert "Course Instructor" not in result
        assert "Course Link" not in result


class TestToolManager:
    """Test suite for ToolManager"""
    
    def test_register_tool(self, mock_vector_store):
        """Test registering tools with the manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        
        manager.register_tool(tool)
        
        assert "search_course_content" in manager.tools
        assert manager.tools["search_course_content"] == tool
    
    def test_register_tool_without_name(self, mock_vector_store):
        """Test that registering a tool without name raises error"""
        manager = ToolManager()
        
        # Create a mock tool with invalid definition
        mock_tool = MagicMock()
        mock_tool.get_tool_definition.return_value = {"description": "test"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name'"):
            manager.register_tool(mock_tool)
    
    def test_get_tool_definitions(self, mock_vector_store):
        """Test getting all tool definitions"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        definitions = manager.get_tool_definitions()
        
        assert len(definitions) == 2
        assert any(d["name"] == "search_course_content" for d in definitions)
        assert any(d["name"] == "get_course_outline" for d in definitions)
    
    def test_execute_tool(self, mock_vector_store):
        """Test executing a tool through the manager"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        result = manager.execute_tool("search_course_content", query="test query")
        
        assert "Building Towards Computer Use" in result
        mock_vector_store.search.assert_called_once()
    
    def test_execute_nonexistent_tool(self):
        """Test executing a tool that doesn't exist"""
        manager = ToolManager()
        
        result = manager.execute_tool("nonexistent_tool", query="test")
        
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, mock_vector_store):
        """Test retrieving sources from last tool execution"""
        manager = ToolManager()
        tool = CourseSearchTool(mock_vector_store)
        manager.register_tool(tool)
        
        # Execute tool to generate sources
        manager.execute_tool("search_course_content", query="test")
        
        sources = manager.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["text"] == "Building Towards Computer Use with Anthropic - Lesson 0"
    
    def test_reset_sources(self, mock_vector_store):
        """Test resetting sources from all tools"""
        manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)
        
        manager.register_tool(search_tool)
        manager.register_tool(outline_tool)
        
        # Generate sources
        manager.execute_tool("search_course_content", query="test")
        assert manager.get_last_sources() != []
        
        # Reset sources
        manager.reset_sources()
        assert manager.get_last_sources() == []
        assert search_tool.last_sources == []
        assert outline_tool.last_sources == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])