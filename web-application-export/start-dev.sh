#!/bin/bash
# Development startup script for INTELLISEARCH Web Application
# This script starts both the backend and frontend in development mode

echo "🚀 Starting INTELLISEARCH Web Application..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}Warning: Port $port is already in use${NC}"
        return 1
    fi
    return 0
}

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}❌ Python is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python found${NC}"

# Check Node.js
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Node.js found${NC}"

# Check npm
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✅ npm found${NC}"

echo ""

# Check ports
echo -e "${BLUE}Checking ports...${NC}"
check_port 8000 || echo -e "${YELLOW}Backend may conflict on port 8000${NC}"
check_port 3000 || echo -e "${YELLOW}Frontend may conflict on port 3000${NC}"

echo ""

# Install dependencies if needed
echo -e "${BLUE}Installing dependencies...${NC}"

# Backend dependencies
cd web-app/backend
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Installing backend dependencies...${NC}"
    pip install fastapi uvicorn python-multipart
fi
cd ../..

# Frontend dependencies
cd web-app/frontend
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}Installing frontend dependencies...${NC}"
    npm install --legacy-peer-deps
fi
cd ../..

echo ""
echo -e "${GREEN}🎉 Starting services...${NC}"
echo ""

# Start backend in background
echo -e "${BLUE}Starting FastAPI backend on http://localhost:8000${NC}"
cd web-app/backend
python main.py &
BACKEND_PID=$!
cd ../..

# Wait a moment for backend to start
sleep 3

# Start frontend
echo -e "${BLUE}Starting React frontend on http://localhost:3000${NC}"
cd web-app/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo -e "${GREEN}🚀 INTELLISEARCH Web Application is starting!${NC}"
echo ""
echo -e "${BLUE}Frontend:${NC} http://localhost:3000"
echo -e "${BLUE}Backend:${NC}  http://localhost:8000"
echo -e "${BLUE}API Docs:${NC} http://localhost:8000/api/docs"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop both services${NC}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Stopping services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo -e "${GREEN}Services stopped. Goodbye!${NC}"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Wait for user to stop
wait