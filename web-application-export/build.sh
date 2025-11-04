#!/bin/bash
# Build script for INTELLISEARCH deployment testing

echo "🚀 Building INTELLISEARCH for deployment..."

# Build frontend
echo "📦 Building frontend..."
cd web-app/frontend
npm install --legacy-peer-deps
npm run build

if [ $? -eq 0 ]; then
    echo "✅ Frontend build successful!"
else
    echo "❌ Frontend build failed!"
    exit 1
fi

# Test backend
echo "🔧 Testing backend..."
cd ../backend
python -c "import main; print('✅ Backend imports successful!')"

if [ $? -eq 0 ]; then
    echo "✅ Backend test successful!"
else
    echo "❌ Backend test failed!"
    exit 1
fi

echo "🎉 Build completed successfully!"
echo ""
echo "Next steps:"
echo "1. Push your code to GitHub"
echo "2. Follow the RENDER_DEPLOYMENT_GUIDE.md"
echo "3. Deploy to Render.com"