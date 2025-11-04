# INTELLISEARCH Deployment Dependencies Summary

## Backend Dependencies (requirements.txt)

### Core FastAPI Framework
- `fastapi>=0.104.1` - Main web framework
- `uvicorn[standard]>=0.24.0` - ASGI server with full features
- `python-multipart>=0.0.6` - File upload support
- `pydantic>=2.8.0` - Data validation and settings management

### HTTP & Async Support
- `httpx>=0.25.2` - Modern HTTP client for async requests
- `aiohttp>=3.9.0` - Async HTTP client/server
- `aiofiles>=23.2.0` - Async file operations
- `requests>=2.0.0` - Standard HTTP library

### INTELLISEARCH Core Dependencies
- `langchain>=1.0.0` - LLM application framework
- `langchain-core>=1.0.0` - Core langchain functionality
- `langchain-community>=0.4.0` - Community integrations
- `langchain-text-splitters>=1.0.0` - Text processing
- `langgraph>=1.0.0` - Workflow engine
- `google-genai>=1.0.0` - Google AI integration
- `langchain-google-genai>=3.0.0` - Langchain Google AI connector

### Web Scraping & Data Processing
- `beautifulsoup4>=4.12.2` - HTML/XML parsing
- `requests-html>=0.10.0` - JavaScript-enabled requests
- `lxml>=5.0.0` - Fast XML/HTML processing
- `trafilatura>=2.0.0` - Web content extraction
- `PyMuPDF>=1.26.0` - PDF processing
- `fpdf2>=2.8.0` - PDF generation

### Search & Ranking
- `rank-bm25>=0.2.2` - BM25 ranking algorithm
- `faiss-cpu>=1.7.4` - Vector similarity search

### Utilities
- `python-dotenv>=1.0.0` - Environment variable management
- `nest_asyncio>=1.5.6` - Nested async event loops
- `ratelimit>=2.2.1` - API rate limiting

### Production & Development
- `gunicorn>=21.2.0` - Production WSGI server
- `pytest>=7.4.3` - Testing framework
- `pytest-asyncio>=0.21.1` - Async testing support
- `black>=23.11.0` - Code formatting
- `flake8>=6.1.0` - Code linting

## Frontend Dependencies (package.json)

### Core React Framework
- `react@^19.1.1` - React library
- `react-dom@^19.1.1` - React DOM rendering
- `react-router-dom@^6.28.0` - Client-side routing

### Build Tools & Development
- `vite@^7.1.7` - Fast build tool and dev server
- `@vitejs/plugin-react@^5.0.4` - Vite React plugin
- `typescript@~5.9.3` - TypeScript compiler
- `typescript-eslint@^8.45.0` - TypeScript ESLint rules

### UI & Styling
- `tailwindcss@^3.4.15` - Utility-first CSS framework
- `autoprefixer@^10.4.20` - CSS vendor prefixing
- `postcss@^8.4.49` - CSS processing tool
- `@headlessui/react@^2.2.0` - Unstyled UI components
- `@heroicons/react@^2.1.5` - Icon library
- `lucide-react@^0.294.0` - Icon library
- `clsx@^2.1.1` - Conditional CSS classes
- `tailwind-merge@^2.5.4` - Tailwind class merging

### HTTP & API Communication
- `axios@^1.7.7` - HTTP client for API calls

### Code Quality & Linting
- `eslint@^9.36.0` - JavaScript/TypeScript linting
- `@eslint/js@^9.36.0` - ESLint JavaScript configs
- `eslint-plugin-react-hooks@^5.2.0` - React Hooks linting
- `eslint-plugin-react-refresh@^0.4.22` - React Refresh linting
- `globals@^16.4.0` - Global variables for ESLint

### Type Definitions
- `@types/node@^24.6.0` - Node.js type definitions
- `@types/react@^19.1.16` - React type definitions
- `@types/react-dom@^19.1.9` - React DOM type definitions

## Installation Commands

### Backend Setup
```bash
cd web-app/backend
pip install -r requirements.txt
```

### Frontend Setup
```bash
cd web-app/frontend
npm install --legacy-peer-deps
```

## Environment Requirements

### Backend Environment Variables
- `GOOGLE_API_KEY` - Google Gemini API key
- `SERPER_API_KEY` - Serper search API key
- `ENVIRONMENT` - Set to "production" for deployment
- `PORT` - Server port (automatically set by Render)

### Frontend Environment Variables
- `VITE_API_URL` - Backend API URL (e.g., https://backend.onrender.com)

## Version Compatibility

### Python Requirements
- Python 3.8+ (recommended: Python 3.11+)
- pip 21.0+

### Node.js Requirements
- Node.js 18.0+
- npm 8.0+

## Production Optimizations

### Backend
- Uses `uvicorn` for development, `gunicorn` for production
- Automatic port detection from environment
- Disabled reload in production mode
- CORS configured for cross-origin requests

### Frontend
- TypeScript for type safety
- Vite for fast builds and hot reload
- Tailwind CSS for optimized styling
- Tree-shaking for smaller bundle sizes
- Environment-based API URL configuration

This setup ensures your INTELLISEARCH application has all dependencies needed for both local development and production deployment on Render.