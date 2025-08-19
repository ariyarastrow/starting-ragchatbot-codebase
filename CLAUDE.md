# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
# Quick start (recommended)
chmod +x run.sh && ./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Package Management
```bash
# Install dependencies
uv sync

# Add new dependency
uv add package-name

# Update dependencies
uv lock --upgrade

# IMPORTANT: Always use uv, never use pip directly
# To run Python files, use: uv run python filename.py
```

### Environment Setup
- Requires `.env` file in root with `ANTHROPIC_API_KEY=your_key_here`
- Python 3.13+ required
- Uses `uv` as package manager

### Access Points
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Architecture Overview

This is a RAG (Retrieval-Augmented Generation) system with a tool-augmented AI architecture:

### Core Flow
1. **Frontend** (`frontend/`) - Vanilla HTML/CSS/JS chat interface
2. **FastAPI Backend** (`backend/app.py`) - REST API with CORS for frontend
3. **RAG System** (`backend/rag_system.py`) - Central orchestrator
4. **AI Generator** (`backend/ai_generator.py`) - Claude API with tool calling
5. **Search Tools** (`backend/search_tools.py`) - Semantic search capabilities
6. **Vector Store** (`backend/vector_store.py`) - ChromaDB with sentence-transformers

### Key Architecture Patterns

**Tool-Augmented AI**: Claude decides when to search vs. using existing knowledge. The AI has access to a `search_course_content` tool that performs semantic search through the vector database.

**Modular Components**: Each backend module has single responsibility:
- `document_processor.py` - Parses course documents and chunks text
- `session_manager.py` - Maintains conversation history 
- `models.py` - Pydantic data models (Course, Lesson, CourseChunk)
- `config.py` - Centralized configuration with environment variables

**Vector Storage Strategy**: ChromaDB stores two collections:
- `course_content` - Text chunks with course/lesson metadata
- `course_metadata` - Course titles and descriptions for semantic course matching

### Document Processing Pipeline
1. Parse structured course files (Course Title/Link/Instructor, then Lessons)
2. Extract lessons using regex `Lesson \d+: [title]`
3. Chunk text (800 chars, 100 char overlap) with sentence boundaries
4. Add contextual prefixes: `"Course [title] Lesson [num] content: [chunk]"`
5. Generate embeddings using `sentence-transformers/all-MiniLM-L6-v2`

### Data Models
- **Course**: Contains title, optional link/instructor, list of lessons
- **Lesson**: lesson_number, title, optional lesson_link
- **CourseChunk**: content text, course_title, lesson_number, chunk_index

### Session Management
- Conversation history maintained per session (max 2 exchanges by default)
- Session IDs auto-generated, carried through API calls
- History formatted as "User: [msg]\nAssistant: [msg]" for Claude context

### Configuration
Key settings in `config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks  
- `MAX_RESULTS: 5` - Maximum search results returned
- `MAX_HISTORY: 2` - Conversation exchanges to remember

## Important Implementation Details

### Course Document Format
Expected format for documents in `docs/` folder:
```
Course Title: [title]
Course Link: [url]  
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [optional lesson url]
[lesson content...]

Lesson 1: [title]
[lesson content...]
```

### Frontend-Backend Communication
- Frontend uses relative URLs (`/api/query`, `/api/courses`)
- JSON payloads with query, optional session_id
- Response includes answer, sources array, session_id
- Sources displayed in collapsible UI sections

### Error Handling
- Document processing gracefully handles missing metadata
- Vector search returns empty results rather than errors
- API endpoints wrap operations in try/catch with HTTP 500 responses
- Frontend displays error messages in chat interface

### ChromaDB Storage
- Persistent storage in `./chroma_db/` directory
- Collections auto-created on first use
- Documents include metadata for filtering (course_title, lesson_number)
- Embeddings generated using sentence-transformers model
- always use uv to run the server do not use pip directly
- make sure to use uv to manage all dependencies
- use uv to run Python files
- use uv to run Python files.