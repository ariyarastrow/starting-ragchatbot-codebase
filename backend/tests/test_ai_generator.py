import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai_generator import AIGenerator
import anthropic


class TestAIGenerator:
    """Test suite for AIGenerator"""
    
    def test_initialization(self):
        """Test AIGenerator initialization"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            generator = AIGenerator("test-key", "claude-3-sonnet")
            
            mock_anthropic.assert_called_once_with(api_key="test-key")
            assert generator.model == "claude-3-sonnet"
            assert generator.base_params["model"] == "claude-3-sonnet"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_without_tools(self):
        """Test generating response without tool support"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            # Setup mock client
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            # Setup mock response
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="This is a general response.")]
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            result = generator.generate_response("What is AI?")
            
            # Verify API call
            mock_client.messages.create.assert_called_once()
            call_args = mock_client.messages.create.call_args[1]
            
            assert call_args["model"] == "claude-3-sonnet"
            assert call_args["messages"][0]["content"] == "What is AI?"
            assert "tools" not in call_args
            assert result == "This is a general response."
    
    def test_generate_response_with_tools_no_tool_use(self):
        """Test response with tools available but not used"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock response without tool use
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Direct answer without tools.")]
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            tools = [{"name": "search_course_content", "description": "Search tool"}]
            
            result = generator.generate_response(
                "What is the capital of France?",
                tools=tools
            )
            
            # Verify tools were passed but not used
            call_args = mock_client.messages.create.call_args[1]
            assert "tools" in call_args
            assert call_args["tool_choice"] == {"type": "auto"}
            assert result == "Direct answer without tools."
    
    def test_generate_response_with_tool_use(self):
        """Test response that triggers tool use"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock initial response with tool use
            mock_tool_block = MagicMock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": "prompt caching"}
            mock_tool_block.id = "tool_123"
            
            mock_initial_response = MagicMock()
            mock_initial_response.content = [mock_tool_block]
            mock_initial_response.stop_reason = "tool_use"
            
            # Mock final response after tool execution
            mock_final_response = MagicMock()
            mock_final_response.content = [MagicMock(text="Tool result processed.")]
            
            # Setup create to return different responses
            mock_client.messages.create.side_effect = [
                mock_initial_response,
                mock_final_response
            ]
            
            # Mock tool manager
            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.return_value = "Search results here"
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            tools = [{"name": "search_course_content"}]
            
            result = generator.generate_response(
                "What is prompt caching?",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Verify tool was executed
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="prompt caching"
            )
            
            # Verify two API calls were made
            assert mock_client.messages.create.call_count == 2
            assert result == "Tool result processed."
    
    def test_handle_tool_execution(self):
        """Test the _handle_tool_execution method directly"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            # Mock tool execution response
            mock_tool_block = MagicMock()
            mock_tool_block.type = "tool_use"
            mock_tool_block.name = "search_course_content"
            mock_tool_block.input = {"query": "computer use", "course_name": "Building Towards Computer Use"}
            mock_tool_block.id = "tool_456"
            
            mock_initial_response = MagicMock()
            mock_initial_response.content = [mock_tool_block]
            
            # Mock final response
            mock_final_response = MagicMock()
            mock_final_response.content = [MagicMock(text="Computer use allows models to interact with computers.")]
            mock_client.messages.create.return_value = mock_final_response
            
            # Mock tool manager
            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.return_value = "Course content about computer use found."
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            
            base_params = {
                "messages": [{"role": "user", "content": "What is computer use?"}],
                "system": "Test system prompt"
            }
            
            result = generator._handle_tool_execution(
                mock_initial_response,
                base_params,
                mock_tool_manager
            )
            
            # Verify tool execution
            mock_tool_manager.execute_tool.assert_called_once_with(
                "search_course_content",
                query="computer use",
                course_name="Building Towards Computer Use"
            )
            
            # Verify final API call
            final_call_args = mock_client.messages.create.call_args[1]
            assert len(final_call_args["messages"]) == 3  # user, assistant, user with tool results
            assert final_call_args["messages"][2]["content"][0]["type"] == "tool_result"
            assert result == "Computer use allows models to interact with computers."
    
    def test_generate_response_with_conversation_history(self):
        """Test response generation with conversation history"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            mock_response = MagicMock()
            mock_response.content = [MagicMock(text="Response with context.")]
            mock_response.stop_reason = "end_turn"
            mock_client.messages.create.return_value = mock_response
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            
            history = "User: Previous question\nAssistant: Previous answer"
            result = generator.generate_response(
                "Follow-up question",
                conversation_history=history
            )
            
            # Verify history was included in system prompt
            call_args = mock_client.messages.create.call_args[1]
            assert "Previous conversation:" in call_args["system"]
            assert history in call_args["system"]
            assert result == "Response with context."
    
    def test_multiple_tool_calls_in_response(self):
        """Test handling multiple tool calls in a single response"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            # Create multiple tool blocks
            tool_block1 = MagicMock()
            tool_block1.type = "tool_use"
            tool_block1.name = "search_course_content"
            tool_block1.input = {"query": "prompt caching"}
            tool_block1.id = "tool_1"
            
            tool_block2 = MagicMock()
            tool_block2.type = "tool_use"
            tool_block2.name = "get_course_outline"
            tool_block2.input = {"course_name": "Building Towards Computer Use"}
            tool_block2.id = "tool_2"
            
            text_block = MagicMock()
            text_block.type = "text"
            text_block.text = "Let me search for that."
            
            mock_initial_response = MagicMock()
            mock_initial_response.content = [text_block, tool_block1, tool_block2]
            mock_initial_response.stop_reason = "tool_use"
            
            mock_final_response = MagicMock()
            mock_final_response.content = [MagicMock(text="Combined results from both tools.")]
            
            mock_client.messages.create.side_effect = [
                mock_initial_response,
                mock_final_response
            ]
            
            mock_tool_manager = MagicMock()
            mock_tool_manager.execute_tool.side_effect = [
                "Search result 1",
                "Outline result"
            ]
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            tools = [
                {"name": "search_course_content"},
                {"name": "get_course_outline"}
            ]
            
            result = generator.generate_response(
                "Tell me about prompt caching and the course outline",
                tools=tools,
                tool_manager=mock_tool_manager
            )
            
            # Verify both tools were executed
            assert mock_tool_manager.execute_tool.call_count == 2
            mock_tool_manager.execute_tool.assert_any_call(
                "search_course_content",
                query="prompt caching"
            )
            mock_tool_manager.execute_tool.assert_any_call(
                "get_course_outline",
                course_name="Building Towards Computer Use"
            )
            
            assert result == "Combined results from both tools."
    
    def test_error_handling_in_api_call(self):
        """Test error handling when API call fails"""
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic_class:
            mock_client = MagicMock()
            mock_anthropic_class.return_value = mock_client
            
            # Simulate API error
            mock_client.messages.create.side_effect = Exception("API Error")
            
            generator = AIGenerator("test-key", "claude-3-sonnet")
            
            with pytest.raises(Exception, match="API Error"):
                generator.generate_response("Test query")
    
    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        generator = AIGenerator("test-key", "claude-3-sonnet")
        
        # Check key elements in system prompt
        assert "search_course_content" in generator.SYSTEM_PROMPT
        assert "get_course_outline" in generator.SYSTEM_PROMPT
        assert "Tool Usage Guidelines" in generator.SYSTEM_PROMPT
        assert "For content-specific queries" in generator.SYSTEM_PROMPT
        assert "For outline/syllabus/structure queries" in generator.SYSTEM_PROMPT


if __name__ == "__main__":
    pytest.main([__file__, "-v"])