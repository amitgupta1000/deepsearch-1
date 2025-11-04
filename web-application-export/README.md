# INTELLISEARCH Web Application

A modern web-based interface for the INTELLISEARCH AI research pipeline, built with FastAPI backend and React frontend.

## Architecture

- **Backend**: FastAPI (Python) - Located in `web-app/backend/`
- **Frontend**: React + TypeScript + Vite + Tailwind CSS - Located in `web-app/frontend/`
- **Deployment**: Configured for Render.com deployment

## Features

- 🔍 **AI-Powered Research**: Leverages Google Gemini for intelligent research
- 📊 **Multiple Report Types**: Concise (~500 words) and Detailed (~1000 words)
- 📚 **Source Citations**: Numbered academic-style citations excluded from word counts
- 🌐 **Web Interface**: Modern, responsive React interface
- 🚀 **Real-time Updates**: Progress tracking for research tasks
- 🔐 **API Key Management**: Optional user-provided API keys

## Quick Start

### Prerequisites
- Python 3.11+
- Node.js 18+
- npm or yarn

### Development Setup

1. **Start the Backend** (Port 8000):
```bash
cd web-app/backend
python main.py
```

2. **Start the Frontend** (Port 3000):
```bash
cd web-app/frontend
npm install --legacy-peer-deps
npm run dev
```

3. **Access the Application**:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/api/docs

## Environment Configuration

### Frontend (.env.local)
```
VITE_API_URL=http://localhost:8000
```

### Backend
The backend automatically imports API keys from the main INTELLISEARCH configuration:
- Google Gemini API key from `src/config.py`
- Serper API key from `src/config.py`

## API Endpoints

### Main Endpoints
- `POST /research` - Start a new research task
- `GET /research/{session_id}` - Get research progress/results
- `GET /health` - Health check endpoint

### WebSocket (Future)
- `/ws/{session_id}` - Real-time progress updates

## Deployment

### Render.com Deployment
The application is configured for deployment on Render.com with:
- **Backend Service**: Python web service
- **Frontend Service**: Static site served by Node.js
- **Blueprint**: `render.yaml` for infrastructure as code

### Production Build
```bash
# Frontend production build
cd web-app/frontend
npm run build

# Backend is ready for production as-is
cd web-app/backend
python main.py
```

## Project Structure

```
web-app/
├── backend/
│   ├── main.py              # FastAPI application
│   ├── requirements.txt     # Python dependencies
│   └── start.sh            # Production start script
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   ├── context/        # React context providers
│   │   ├── types/          # TypeScript type definitions
│   │   └── App.tsx         # Main application component
│   ├── package.json        # Node.js dependencies
│   ├── vite.config.ts      # Vite configuration
│   ├── tailwind.config.js  # Tailwind CSS configuration
│   └── .env.example        # Environment variables template
└── render.yaml             # Render deployment configuration
```

## Integration with Core INTELLISEARCH

The web application integrates seamlessly with the existing INTELLISEARCH pipeline:
- **Graph Workflow**: Uses the compiled LangGraph workflow from `src/graph.py`
- **Node Functions**: Leverages all existing research nodes from `src/nodes.py`
- **Data Types**: Maintains compatibility with `AgentState` and other core types
- **Configuration**: Inherits API keys and settings from the main project

## Development Notes

- **React 19**: Uses the latest React version with TypeScript
- **Tailwind CSS**: For modern, responsive styling
- **FastAPI**: Provides automatic API documentation and validation
- **Legacy Peer Deps**: Required for Lucide React icons with React 19

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the backend can import from `../src/` directory
2. **Port Conflicts**: Change ports in configuration if 3000/8000 are in use
3. **API Key Issues**: Verify API keys are properly configured in `src/config.py`
4. **Dependency Conflicts**: Use `--legacy-peer-deps` for npm install

### Development Tips

- Use the browser's developer tools to monitor API calls
- Check FastAPI docs at `/api/docs` for API testing
- Monitor both frontend and backend logs for debugging

## Future Enhancements

- [ ] WebSocket integration for real-time updates
- [ ] User authentication and session management
- [ ] Research history and saved reports
- [ ] Advanced search filters and options
- [ ] Export functionality (PDF, Word, etc.)
- [ ] Multi-language support