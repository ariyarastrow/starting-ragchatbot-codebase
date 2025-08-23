import pytest
from unittest.mock import MagicMock, patch, call
import sys
import os
import json

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from vector_store import VectorStore, SearchResults
from models import Course, Lesson, CourseChunk


class TestSearchResults:
    """Test SearchResults dataclass"""
    
    def test_from_chroma(self):
        """Test creating SearchResults from ChromaDB response"""
        chroma_results = {
            'documents': [['doc1', 'doc2']],
            'metadatas': [[{'key': 'value1'}, {'key': 'value2'}]],
            'distances': [[0.1, 0.2]]
        }
        
        results = SearchResults.from_chroma(chroma_results)
        
        assert results.documents == ['doc1', 'doc2']
        assert results.metadata == [{'key': 'value1'}, {'key': 'value2'}]
        assert results.distances == [0.1, 0.2]
        assert results.error is None
    
    def test_empty(self):
        """Test creating empty SearchResults with error"""
        results = SearchResults.empty("No results found")
        
        assert results.documents == []
        assert results.metadata == []
        assert results.distances == []
        assert results.error == "No results found"
        assert results.is_empty()
    
    def test_is_empty(self):
        """Test is_empty method"""
        # Empty results
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        assert empty_results.is_empty()
        
        # Non-empty results
        full_results = SearchResults(
            documents=['doc'],
            metadata=[{}],
            distances=[0.1]
        )
        assert not full_results.is_empty()


class TestVectorStore:
    """Test VectorStore class"""
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_initialization(self, mock_embedding_func, mock_chroma_client):
        """Test VectorStore initialization"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        store = VectorStore("./test_db", "test-model", max_results=10)
        
        # Verify ChromaDB client was created
        mock_chroma_client.assert_called_once()
        
        # Verify collections were created
        assert mock_client.get_or_create_collection.call_count == 2
        mock_client.get_or_create_collection.assert_any_call(
            name="course_catalog",
            embedding_function=mock_embedding_func.return_value
        )
        mock_client.get_or_create_collection.assert_any_call(
            name="course_content",
            embedding_function=mock_embedding_func.return_value
        )
        
        assert store.max_results == 10
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_basic(self, mock_embedding_func, mock_chroma_client):
        """Test basic search without filters"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # Setup mock search results
        mock_content_collection.query.return_value = {
            'documents': [['Result 1', 'Result 2']],
            'metadatas': [[
                {'course_title': 'Course 1', 'lesson_number': 1},
                {'course_title': 'Course 1', 'lesson_number': 2}
            ]],
            'distances': [[0.1, 0.2]]
        }
        
        store = VectorStore("./test_db", "test-model")
        results = store.search("test query")
        
        # Verify search was called correctly
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where=None
        )
        
        assert len(results.documents) == 2
        assert results.documents[0] == 'Result 1'
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_course_filter(self, mock_embedding_func, mock_chroma_client):
        """Test search with course name filter"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # Setup course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [['Building Towards Computer Use']],
            'metadatas': [[{'title': 'Building Towards Computer Use'}]]
        }
        
        # Setup content search
        mock_content_collection.query.return_value = {
            'documents': [['Filtered result']],
            'metadatas': [[{'course_title': 'Building Towards Computer Use'}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore("./test_db", "test-model")
        results = store.search("test query", course_name="Computer Use")
        
        # Verify course was resolved
        mock_catalog_collection.query.assert_called_once_with(
            query_texts=["Computer Use"],
            n_results=1
        )
        
        # Verify content search with filter
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"course_title": "Building Towards Computer Use"}
        )
        
        assert len(results.documents) == 1
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_lesson_filter(self, mock_embedding_func, mock_chroma_client):
        """Test search with lesson number filter"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        mock_content_collection.query.return_value = {
            'documents': [['Lesson 3 content']],
            'metadatas': [[{'lesson_number': 3}]],
            'distances': [[0.1]]
        }
        
        store = VectorStore("./test_db", "test-model")
        results = store.search("test query", lesson_number=3)
        
        # Verify search with lesson filter
        mock_content_collection.query.assert_called_once_with(
            query_texts=["test query"],
            n_results=5,
            where={"lesson_number": 3}
        )
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_both_filters(self, mock_embedding_func, mock_chroma_client):
        """Test search with both course and lesson filters"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # Setup course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [['Test Course']],
            'metadatas': [[{'title': 'Test Course'}]]
        }
        
        mock_content_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        store = VectorStore("./test_db", "test-model")
        results = store.search("query", course_name="Test", lesson_number=2)
        
        # Verify combined filter
        mock_content_collection.query.assert_called_once_with(
            query_texts=["query"],
            n_results=5,
            where={"$and": [
                {"course_title": "Test Course"},
                {"lesson_number": 2}
            ]}
        )
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_course_not_found(self, mock_embedding_func, mock_chroma_client):
        """Test search when course name doesn't match"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # No course found
        mock_catalog_collection.query.return_value = {
            'documents': [[]],
            'metadatas': [[]],
            'distances': [[]]
        }
        
        store = VectorStore("./test_db", "test-model")
        results = store.search("query", course_name="Nonexistent Course")
        
        assert results.error == "No course found matching 'Nonexistent Course'"
        assert results.is_empty()
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_search_with_exception(self, mock_embedding_func, mock_chroma_client):
        """Test search error handling"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_content_collection = MagicMock()
        mock_catalog_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # Simulate query error
        mock_content_collection.query.side_effect = Exception("Database error")
        
        store = VectorStore("./test_db", "test-model")
        results = store.search("query")
        
        assert "Search error: Database error" in results.error
        assert results.is_empty()
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_metadata(self, mock_embedding_func, mock_chroma_client):
        """Test adding course metadata"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        store = VectorStore("./test_db", "test-model")
        
        course = Course(
            title="Test Course",
            course_link="https://example.com",
            instructor="Test Instructor",
            lessons=[
                Lesson(lesson_number=0, title="Intro", lesson_link="https://example.com/lesson0"),
                Lesson(lesson_number=1, title="Advanced", lesson_link=None)
            ]
        )
        
        store.add_course_metadata(course)
        
        # Verify add was called with correct data
        mock_catalog_collection.add.assert_called_once()
        call_args = mock_catalog_collection.add.call_args[1]
        
        assert call_args['documents'] == ["Test Course"]
        assert call_args['ids'] == ["Test Course"]
        
        metadata = call_args['metadatas'][0]
        assert metadata['title'] == "Test Course"
        assert metadata['instructor'] == "Test Instructor"
        assert metadata['course_link'] == "https://example.com"
        assert metadata['lesson_count'] == 2
        
        # Check lessons JSON
        lessons_data = json.loads(metadata['lessons_json'])
        assert len(lessons_data) == 2
        assert lessons_data[0]['lesson_number'] == 0
        assert lessons_data[0]['lesson_title'] == "Intro"
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_add_course_content(self, mock_embedding_func, mock_chroma_client):
        """Test adding course content chunks"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        store = VectorStore("./test_db", "test-model")
        
        chunks = [
            CourseChunk(
                content="Chunk 1 content",
                course_title="Test Course",
                lesson_number=0,
                chunk_index=0
            ),
            CourseChunk(
                content="Chunk 2 content",
                course_title="Test Course",
                lesson_number=1,
                chunk_index=1
            )
        ]
        
        store.add_course_content(chunks)
        
        # Verify add was called correctly
        mock_content_collection.add.assert_called_once()
        call_args = mock_content_collection.add.call_args[1]
        
        assert call_args['documents'] == ["Chunk 1 content", "Chunk 2 content"]
        assert call_args['ids'] == ["Test_Course_0", "Test_Course_1"]
        
        metadatas = call_args['metadatas']
        assert metadatas[0]['course_title'] == "Test Course"
        assert metadatas[0]['lesson_number'] == 0
        assert metadatas[1]['chunk_index'] == 1
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_get_course_info(self, mock_embedding_func, mock_chroma_client):
        """Test getting course information"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection
        ]
        
        # Setup course resolution
        mock_catalog_collection.query.return_value = {
            'documents': [['Building Towards Computer Use']],
            'metadatas': [[{'title': 'Building Towards Computer Use'}]]
        }
        
        # Setup course metadata retrieval
        lessons_json = json.dumps([
            {'lesson_number': 0, 'lesson_title': 'Intro', 'lesson_link': 'https://example.com/l0'},
            {'lesson_number': 1, 'lesson_title': 'Advanced', 'lesson_link': None}
        ])
        
        mock_catalog_collection.get.return_value = {
            'metadatas': [{
                'title': 'Building Towards Computer Use',
                'course_link': 'https://example.com/course',
                'instructor': 'Colt Steele',
                'lessons_json': lessons_json
            }]
        }
        
        store = VectorStore("./test_db", "test-model")
        course_info = store.get_course_info("Computer Use")
        
        assert course_info['title'] == 'Building Towards Computer Use'
        assert course_info['link'] == 'https://example.com/course'
        assert course_info['instructor'] == 'Colt Steele'
        assert len(course_info['lessons']) == 2
        assert course_info['lessons'][0]['number'] == 0
        assert course_info['lessons'][0]['title'] == 'Intro'
    
    @patch('vector_store.chromadb.PersistentClient')
    @patch('vector_store.chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction')
    def test_clear_all_data(self, mock_embedding_func, mock_chroma_client):
        """Test clearing all data from collections"""
        mock_client = MagicMock()
        mock_chroma_client.return_value = mock_client
        
        mock_catalog_collection = MagicMock()
        mock_content_collection = MagicMock()
        
        # Initial creation
        mock_client.get_or_create_collection.side_effect = [
            mock_catalog_collection,
            mock_content_collection,
            # After clear
            mock_catalog_collection,
            mock_content_collection
        ]
        
        store = VectorStore("./test_db", "test-model")
        store.clear_all_data()
        
        # Verify collections were deleted
        mock_client.delete_collection.assert_any_call("course_catalog")
        mock_client.delete_collection.assert_any_call("course_content")
        
        # Verify collections were recreated
        assert mock_client.get_or_create_collection.call_count == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])