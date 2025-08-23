"""
Real integration tests to diagnose 'query failed' issue.
These tests interact with actual ChromaDB and check real data.
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import Config
from document_processor import DocumentProcessor
from models import Course, CourseChunk, Lesson
from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import VectorStore


class TestRealIntegration:
    """Test with real ChromaDB to diagnose issues"""

    def test_check_existing_database(self):
        """Check if the existing ChromaDB has any data"""
        # Use the actual config
        config = Config()

        # Create vector store with actual ChromaDB
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Check course count
        course_count = vector_store.get_course_count()
        print(f"\nNumber of courses in database: {course_count}")

        # Get existing course titles
        course_titles = vector_store.get_existing_course_titles()
        print(f"Course titles: {course_titles}")

        # Try a basic search
        results = vector_store.search("computer use")
        print(f"\nSearch for 'computer use':")
        print(f"  - Documents found: {len(results.documents)}")
        print(f"  - Error: {results.error}")
        if results.documents:
            print(f"  - First result: {results.documents[0][:100]}...")

        # Try search with course filter
        if course_titles:
            results = vector_store.search(
                "prompt caching",
                course_name=course_titles[0] if course_titles else None,
            )
            print(
                f"\nSearch for 'prompt caching' in course '{course_titles[0] if course_titles else 'None'}':"
            )
            print(f"  - Documents found: {len(results.documents)}")
            print(f"  - Error: {results.error}")

    def test_document_processing(self):
        """Test if documents are being processed correctly"""
        config = Config()
        processor = DocumentProcessor(config.CHUNK_SIZE, config.CHUNK_OVERLAP)

        # Check if docs folder exists
        docs_path = "../docs"
        if os.path.exists(docs_path):
            files = [f for f in os.listdir(docs_path) if f.endswith(".txt")]
            print(f"\nFound {len(files)} .txt files in docs folder")

            if files:
                # Process first file
                file_path = os.path.join(docs_path, files[0])
                print(f"Processing: {files[0]}")

                try:
                    course, chunks = processor.process_course_document(file_path)
                    print(f"  - Course title: {course.title}")
                    print(f"  - Number of lessons: {len(course.lessons)}")
                    print(f"  - Number of chunks: {len(chunks)}")

                    if chunks:
                        print(f"  - First chunk preview: {chunks[0].content[:100]}...")
                except Exception as e:
                    print(f"  - Error processing: {e}")
        else:
            print(f"\nDocs folder not found at {docs_path}")

    def test_tool_execution_with_real_data(self):
        """Test tool execution with actual database"""
        config = Config()
        vector_store = VectorStore(
            config.CHROMA_PATH, config.EMBEDDING_MODEL, config.MAX_RESULTS
        )

        # Create and test search tool
        search_tool = CourseSearchTool(vector_store)

        print("\nTesting CourseSearchTool.execute:")

        # Test 1: Basic search
        result = search_tool.execute(query="computer use")
        print(f"\n1. Search for 'computer use':")
        print(f"   Result length: {len(result)}")
        print(f"   Sources found: {len(search_tool.last_sources)}")
        if result and len(result) > 100:
            print(f"   Preview: {result[:200]}...")
        else:
            print(f"   Full result: {result}")

        # Test 2: Search with specific content
        result = search_tool.execute(query="prompt caching")
        print(f"\n2. Search for 'prompt caching':")
        print(f"   Result length: {len(result)}")
        print(f"   Sources found: {len(search_tool.last_sources)}")

        # Test 3: Course outline tool
        outline_tool = CourseOutlineTool(vector_store)

        # Get any existing course
        course_titles = vector_store.get_existing_course_titles()
        if course_titles:
            result = outline_tool.execute(course_name=course_titles[0])
            print(f"\n3. Get outline for '{course_titles[0]}':")
            print(f"   Result length: {len(result)}")
            if result and "No course found" not in result:
                print(f"   Preview: {result[:300]}...")

    def test_rag_system_query(self):
        """Test the full RAG system query"""
        config = Config()

        # Check if API key is set
        if not config.ANTHROPIC_API_KEY:
            print("\n[WARNING] ANTHROPIC_API_KEY not set - skipping RAG query test")
            return

        try:
            # Create RAG system
            rag = RAGSystem(config)

            # Get course count
            analytics = rag.get_course_analytics()
            print(f"\nRAG System Analytics:")
            print(f"  - Total courses: {analytics['total_courses']}")
            print(f"  - Course titles: {analytics['course_titles']}")

            # Try to query
            print("\nTesting RAG queries:")

            # Test queries
            test_queries = [
                "What is computer use?",
                "Tell me about prompt caching",
                "What lessons are in the course?",
                "Show me the course outline",
            ]

            for query in test_queries:
                print(f"\nQuery: '{query}'")
                try:
                    response, sources = rag.query(query)
                    print(f"  Response length: {len(response)}")
                    print(f"  Sources: {len(sources)}")
                    if response and len(response) > 100:
                        print(f"  Response preview: {response[:150]}...")
                    else:
                        print(f"  Full response: {response}")
                except Exception as e:
                    print(f"  ERROR: {e}")

        except Exception as e:
            print(f"\nError initializing RAG system: {e}")

    def test_create_test_database(self):
        """Create a small test database to verify functionality"""
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            test_config = Config()
            test_config.CHROMA_PATH = os.path.join(temp_dir, "test_chroma")

            # Create test document
            test_doc_path = os.path.join(temp_dir, "test_course.txt")
            with open(test_doc_path, "w") as f:
                f.write(
                    """Course Title: Test Course for Debugging
Course Link: https://example.com/test
Course Instructor: Test Instructor

Lesson 0: Introduction to Testing
This is the introduction lesson content. We'll learn about testing RAG systems.
The content includes information about prompt caching and computer use.

Lesson 1: Advanced Testing
This lesson covers advanced testing topics including integration testing.
We discuss how to debug when queries return 'query failed'.
"""
                )

            # Create vector store and processor
            vector_store = VectorStore(
                test_config.CHROMA_PATH,
                test_config.EMBEDDING_MODEL,
                test_config.MAX_RESULTS,
            )

            processor = DocumentProcessor(
                test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP
            )

            # Process and add document
            course, chunks = processor.process_course_document(test_doc_path)
            vector_store.add_course_metadata(course)
            vector_store.add_course_content(chunks)

            print("\nTest database created:")
            print(f"  - Course: {course.title}")
            print(f"  - Lessons: {len(course.lessons)}")
            print(f"  - Chunks: {len(chunks)}")

            # Test search
            search_tool = CourseSearchTool(vector_store)
            result = search_tool.execute(query="prompt caching")
            print(f"\nSearch test in temporary database:")
            print(f"  - Result found: {len(result) > 0}")
            print(f"  - Sources: {len(search_tool.last_sources)}")

            if not result or "No relevant content found" in result:
                print("  - ERROR: Search failed even with test data!")
            else:
                print("  - SUCCESS: Search working with test data")


if __name__ == "__main__":
    # Run specific diagnostic tests
    test = TestRealIntegration()

    print("=" * 60)
    print("DIAGNOSTIC TESTS FOR 'QUERY FAILED' ISSUE")
    print("=" * 60)

    test.test_check_existing_database()
    test.test_document_processing()
    test.test_tool_execution_with_real_data()
    test.test_create_test_database()
    test.test_rag_system_query()

    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)
