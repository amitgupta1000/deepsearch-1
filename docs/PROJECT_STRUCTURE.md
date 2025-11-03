# INTELLISEARCH Project Structure Overview

## 🏗️ High-Level Architecture

INTELLISEARCH is an AI-powered research pipeline built with a modular architecture that combines web scraping, semantic search, and AI analysis to generate comprehensive research reports. The project uses LangGraph for workflow orchestration and Google Gemini AI for all language processing tasks.

## 📁 Root Directory Structure

```
INTELLISEARCH/
├── 🎯 Entry Points & Configuration
├── 🧠 Core Engine (src/)
├── 🌐 Web Application (web-app/)
├── 📚 Documentation (docs/)
├── 🧪 Tests (tests/)
└── 📦 Deployment Assets
```

---

## 🎯 Entry Points & Configuration

### **app.py** - Main Application Entry Point
- **Role**: Primary command-line interface and workflow orchestrator
- **Functions**: 
  - Parses command-line arguments
  - Initializes LangGraph workflow
  - Manages interactive vs automated modes
  - Handles batch processing
- **Key Methods**: `run_workflow()`, `interactive_mode()`, `batch_research()`

### **startup_validation.py** - Environment Validator
- **Role**: Validates system dependencies and API keys before execution
- **Functions**:
  - Checks Python environment and packages
  - Validates API key availability
  - Tests core imports and configurations
  - Provides startup diagnostics

### **run_interactive.bat** / **run_automated.bat** - Windows Automation
- **Role**: One-click execution scripts for Windows users
- **Functions**:
  - Sets up Python environment
  - Handles virtual environment activation
  - Provides user-friendly research execution
  - **Interactive**: Full user control with prompts
  - **Automated**: Fast execution with predefined settings

### **setup.py** - Package Configuration
- **Role**: Python package setup and dependency management
- **Functions**: Defines package metadata, dependencies, and entry points

### **requirements.txt** - Dependency Specification
- **Role**: Lists all Python dependencies with version constraints
- **Critical Dependencies**: langchain, google-generativeai, aiohttp, fpdf2, faiss-cpu

### **render.yaml** - Cloud Deployment Configuration
- **Role**: Infrastructure-as-code for Render.com deployment
- **Functions**: Defines backend/frontend services, environment variables, build commands

---

## 🧠 Core Engine (src/)

### **🎛️ Configuration & Setup**

#### **config.py** - Central Configuration Hub
- **Role**: Centralized configuration management for all system parameters
- **Key Sections**:
  - API settings (Google Gemini, Serper)
  - Search parameters (max results, concurrency)
  - Report generation settings (word limits, formats)
  - Retrieval configuration (hybrid search, embedding settings)
  - Performance tuning (timeouts, rate limits)

#### **api_keys.py** - API Key Management
- **Role**: Secure handling of API keys and authentication
- **Functions**: Loads Google API and Serper API keys from environment variables

#### **data_types.py** - Type Definitions
- **Role**: Defines TypedDict classes for structured data throughout the pipeline
- **Key Types**: `AgentState`, `SearchResult`, `ContentData`

### **🔍 Search & Retrieval System**

#### **search.py** - Web Search Engine
- **Role**: Handles web search using Serper API
- **Functions**:
  - Executes search queries
  - Processes search results
  - Manages rate limiting and error handling
  - Formats results for downstream processing

#### **scraper.py** - Web Content Extraction
- **Role**: Multi-strategy web scraping and content extraction
- **Strategies**:
  - Primary: requests-html with JavaScript support
  - Fallback: aiohttp for fast HTTP requests
  - Error handling and retry mechanisms
- **Functions**: Extracts clean text content, handles various content types

#### **hybrid_retriever.py** - Advanced Retrieval System
- **Role**: Implements hybrid search combining BM25 and vector search
- **Features**:
  - Multi-query retrieval for comprehensive coverage
  - Vector embeddings with Google GenerativeAI
  - BM25 keyword search with FAISS
  - Intelligent score fusion and ranking
- **Methods**: `retrieve_multi_query()`, `hybrid_search()`

#### **enhanced_embeddings.py** - Embedding Management
- **Role**: Advanced embedding generation and caching
- **Features**:
  - Google GenerativeAI embeddings integration
  - Intelligent caching with TTL
  - Batch processing optimization
  - Error handling and fallbacks

#### **enhanced_deduplication.py** - Content Deduplication
- **Role**: Intelligent removal of duplicate and redundant content
- **Features**:
  - Rule-based deduplication (exact matches, similarity)
  - LLM-powered intelligent deduplication
  - Caching system for performance
  - Configurable thresholds and fallbacks

### **🤖 AI & Language Processing**

#### **llm_calling.py** - AI Model Interface
- **Role**: Centralized interface for Google Gemini AI interactions
- **Functions**:
  - Handles all LLM API calls
  - Manages rate limiting and retries
  - Processes structured and unstructured responses
  - Error handling and fallback mechanisms

#### **llm_utils.py** - AI Utilities
- **Role**: Helper functions for AI processing
- **Functions**: Token counting, response parsing, prompt formatting

#### **prompt.py** - Prompt Templates
- **Role**: Centralized prompt management for different research types
- **Templates**:
  - General research prompts
  - Legal analysis prompts
  - Macro/economic research prompts
  - Deep search prompts
  - Person/entity search prompts
  - Investment analysis prompts

### **🔄 Workflow Management**

#### **graph.py** - LangGraph Workflow Definition
- **Role**: Defines the research workflow as a directed graph
- **Components**:
  - Node definitions and connections
  - Conditional routing logic
  - State management between nodes
  - Error handling and recovery paths

#### **nodes.py** - Workflow Node Implementations
- **Role**: Contains all individual workflow steps/nodes
- **Key Nodes**:
  - `question_analyzer()`: Analyzes and generates search queries
  - `search_web()`: Executes web searches
  - `scrape_and_evaluate()`: Scrapes and evaluates content
  - `embed_index_and_extract()`: Creates embeddings and extracts relevant chunks
  - `write_report()`: Generates final research reports
  - `evaluate_search_results()`: Assesses search quality
  - `choose_report_type()`: Determines report format

#### **conditions.py** - Workflow Logic
- **Role**: Contains conditional logic for workflow routing
- **Functions**: Determines next steps based on current state, handles decision points

#### **automation_config.py** - Automatcd web-app/backend && python main.py
cd web-app/frontend && npm run devion Management
- **Role**: Manages automated vs interactive execution modes
- **Profiles**:
  - Full automation (no user input)
  - Query-only automation (user chooses report type)
  - No automation (full interaction)

### **🔧 Utilities & Helpers**

#### **utils.py** - General Utilities
- **Role**: Common utility functions used throughout the system
- **Functions**:
  - File I/O operations
  - PDF generation
  - Data formatting and cleaning
  - Error handling helpers

#### **question_analyzer.py** - Query Processing
- **Role**: Analyzes research questions and generates optimized search queries
- **Functions**: Question decomposition, query optimization, search strategy planning

#### **import_validator.py** - Dependency Validation
- **Role**: Validates that all required packages are available
- **Functions**: Checks imports, version compatibility, provides helpful error messages

---

## 🌐 Web Application (web-app/)

### **Backend (web-app/backend/)**

#### **main.py** - FastAPI Web Server
- **Role**: REST API server for web interface
- **Endpoints**:
  - `POST /api/research` - Start new research
  - `GET /api/research/{session_id}` - Get research status/results
  - `GET /api/health` - Health check
  - `GET /api/debug` - System diagnostics
- **Features**: Session management, progress tracking, file downloads

#### **requirements.txt** - Backend Dependencies
- **Role**: Web-specific Python dependencies
- **Key Packages**: fastapi, uvicorn, pydantic, asyncio support

### **Frontend (web-app/frontend/)**

#### **src/main.tsx** - React Application Entry
- **Role**: Main React application initialization
- **Functions**: App mounting, router setup, global providers

#### **src/App.tsx** - Main Application Component
- **Role**: Root React component with routing
- **Features**: Layout structure, route definitions, global state management

#### **src/components/** - React Components
- **ResearchForm.tsx**: Main research interface with form controls
- **ResultsDisplay.tsx**: Research results presentation
- **ProgressBar.tsx**: Real-time progress visualization
- **LoadingSpinner.tsx**: Loading state indicators
- **Header.tsx**: Application header and navigation
- **Footer.tsx**: Application footer
- **WelcomeSection.tsx**: Landing page content

#### **src/context/ResearchContext.tsx** - State Management
- **Role**: Global state management for research operations
- **Functions**: Session state, progress tracking, result management

#### **src/types/index.ts** - TypeScript Definitions
- **Role**: Type definitions for the frontend application
- **Types**: Research requests, responses, state interfaces

#### **package.json** - Frontend Dependencies
- **Role**: Node.js dependencies and build scripts
- **Key Packages**: react, typescript, vite, tailwindcss

#### **vite.config.ts** - Build Configuration
- **Role**: Vite bundler configuration for development and production builds

#### **tailwind.config.js** - Styling Configuration
- **Role**: Tailwind CSS configuration for consistent styling

---

## 📚 Documentation (docs/)

### **EXECUTION_GUIDE.md** - User Guide
- **Role**: Comprehensive guide for running INTELLISEARCH
- **Content**: Setup instructions, execution options, troubleshooting

### **RENDER_DEPLOYMENT_GUIDE.md** - Deployment Guide
- **Role**: Step-by-step deployment instructions for Render.com
- **Content**: Service configuration, environment variables, scaling options

### **Enhancement Documentation**
- **CITATIONS_ENHANCEMENT.md**: Citation system improvements
- **CONTENT_REPETITION_FIX.md**: Deduplication system documentation
- **CURRENT_DATE_CONTEXT_IMPLEMENTATION.md**: Date awareness features

---

## 🧪 Testing (tests/)

### **test_workflow.py** - Integration Tests
- **Role**: End-to-end workflow testing
- **Functions**: Validates complete research pipeline, checks output quality

---

## 📦 Deployment Assets

### **web-application-export/** - Deployment Package
- **Role**: Clean deployment package without development files
- **Content**: Production-ready backend and frontend code

---

## 🔄 Data Flow Architecture

```
1. Query Input (app.py / web interface)
       ↓
2. Question Analysis (question_analyzer.py)
       ↓
3. Web Search (search.py)
       ↓
4. Content Scraping (scraper.py)
       ↓
5. Embedding & Indexing (enhanced_embeddings.py)
       ↓
6. Hybrid Retrieval (hybrid_retriever.py)
       ↓
7. Content Deduplication (enhanced_deduplication.py)
       ↓
8. AI Report Generation (nodes.py + llm_calling.py)
       ↓
9. Output Generation (utils.py - PDF/text)
```

---

## 🎯 Key Design Principles

1. **Modularity**: Each component has a single, well-defined responsibility
2. **Scalability**: Built-in concurrency management and rate limiting
3. **Reliability**: Multiple fallback strategies and error handling
4. **Flexibility**: Configurable parameters and multiple execution modes
5. **Performance**: Intelligent caching, parallel processing, and optimization
6. **Maintainability**: Clear separation of concerns and comprehensive documentation

---

## 🚀 Getting Started

1. **For Development**: Start with `startup_validation.py` to check your environment
2. **For Quick Use**: Run `run_interactive.bat` for guided experience
3. **For Automation**: Use `run_automated.bat` for fast, unattended research
4. **For Web Interface**: Deploy using `web-app/` components
5. **For Customization**: Modify `config.py` and `prompt.py` for your needs

This architecture enables INTELLISEARCH to be both powerful for advanced users and accessible for newcomers, while maintaining high performance and reliability across different usage patterns.