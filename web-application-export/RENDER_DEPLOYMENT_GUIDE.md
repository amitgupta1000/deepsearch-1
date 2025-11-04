# Deploying INTELLISEARCH to Render

This guide will help you deploy your INTELLISEARCH web application to Render.com.

## Prerequisites

1. **GitHub Repository**: Your code should be in a GitHub repository
2. **Render Account**: Sign up at [render.com](https://render.com)
3. **API Keys**: Have your Google API Key and Serper API Key ready

## Step-by-Step Deployment Guide

### 1. Prepare Your Repository

Make sure your repository has the following structure:
```
INTELLISEARCH/
├── web-app/
│   ├── backend/
│   │   ├── main.py
│   │   ├── requirements.txt
│   │   └── Procfile
│   ├── frontend/
│   │   ├── src/
│   │   ├── package.json
│   │   ├── vite.config.ts
│   │   └── tsconfig.json
│   └── render.yaml
```

### 2. Deploy Backend Service

1. **Log into Render Dashboard**
   - Go to [dashboard.render.com](https://dashboard.render.com)
   - Click "New +" → "Web Service"

2. **Connect Repository**
   - Choose "Build and deploy from a Git repository"
   - Connect your GitHub account
   - Select your INTELLISEARCH repository

3. **Configure Backend Service**
   - **Name**: `intellisearch-backend`
   - **Environment**: `Python 3`
   - **Root Directory**: `web-app/backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`

4. **Set Environment Variables**
   - Add these environment variables in Render:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     SERPER_API_KEY=your_serper_api_key_here
     ENVIRONMENT=production
     ```

5. **Deploy Backend**
   - Click "Create Web Service"
   - Wait for deployment to complete (usually 5-10 minutes)
   - Note the backend URL (e.g., `https://intellisearch-backend.onrender.com`)

### 3. Deploy Frontend Service

1. **Create Frontend Service**
   - Click "New +" → "Static Site"
   - Select your repository again

2. **Configure Frontend Service**
   - **Name**: `intellisearch-frontend`
   - **Root Directory**: `web-app/frontend`
   - **Build Command**: `npm install --legacy-peer-deps && npm run build`
   - **Publish Directory**: `build`

3. **Set Environment Variables**
   - Add this environment variable:
     ```
     VITE_API_URL=https://your-backend-url.onrender.com
     ```
   - Replace `your-backend-url` with your actual backend URL from step 2

4. **Deploy Frontend**
   - Click "Create Static Site"
   - Wait for deployment to complete

### 4. Alternative: Blueprint Deployment

You can also use the `render.yaml` file for automated deployment:

1. **Create New Blueprint**
   - In Render Dashboard, click "New +" → "Blueprint"
   - Connect your repository
   - Render will automatically detect the `render.yaml` file

2. **Configure Environment Variables**
   - Set the required environment variables as prompted

3. **Deploy**
   - Click "Apply" to deploy both services simultaneously

## Environment Variables Reference

### Backend Variables
- `GOOGLE_API_KEY`: Your Google Gemini API key
- `SERPER_API_KEY`: Your Serper search API key
- `ENVIRONMENT`: Set to "production"

### Frontend Variables
- `VITE_API_URL`: Your backend service URL (e.g., `https://intellisearch-backend.onrender.com`)

## Troubleshooting

### Common Issues

1. **Build Failures**
   - Check that all dependencies are in `requirements.txt` and `package.json`
   - Ensure Python version compatibility (Render uses Python 3.7+ by default)

2. **Environment Variables**
   - Make sure all required API keys are set correctly
   - Check that the frontend can reach the backend URL

3. **CORS Issues**
   - The backend is already configured to allow CORS for all origins
   - If issues persist, check the frontend's API URL configuration

4. **Free Tier Limitations**
   - Render free tier services spin down after 15 minutes of inactivity
   - First request after spin-down may take 30-60 seconds

### Checking Logs

1. **Backend Logs**
   - Go to your backend service in Render Dashboard
   - Click "Logs" tab to see real-time logs

2. **Frontend Logs**
   - Go to your frontend service in Render Dashboard
   - Check build logs for any deployment issues

## Post-Deployment

### Testing Your Application

1. **Visit Frontend URL**
   - Your frontend will be available at `https://your-frontend-name.onrender.com`
   - Test the research functionality

2. **API Health Check**
   - Visit `https://your-backend-url.onrender.com/health`
   - Should return `{"status": "healthy"}`

### Custom Domain (Optional)

1. **Purchase Domain**
   - Buy a domain from any registrar

2. **Configure in Render**
   - Go to your frontend service
   - Click "Settings" → "Custom Domains"
   - Add your domain and configure DNS

## Monitoring and Maintenance

1. **Performance Monitoring**
   - Check service metrics in Render Dashboard
   - Monitor response times and error rates

2. **Updates**
   - Push changes to your GitHub repository
   - Render will automatically redeploy your services

3. **Scaling**
   - Upgrade to paid plans for:
     - Faster build times
     - Always-on services
     - More compute resources

## Support

- **Render Documentation**: [docs.render.com](https://docs.render.com)
- **Render Community**: [community.render.com](https://community.render.com)
- **Support**: [help.render.com](https://help.render.com)

---

Your INTELLISEARCH application will be live and accessible to users worldwide once deployed to Render!