# Backend Deployment Guide

This guide covers deploying your Crypto Chart API backend to either Vercel or Render.

## 🚀 Option 1: Deploy to Vercel (Recommended for quick setup)

### Prerequisites
- GitHub account
- Vercel account (free tier available)

### Steps to Deploy:

1. **Push your code to GitHub** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial backend commit"
   git remote add origin https://github.com/yourusername/yourrepo.git
   git push -u origin main
   ```

2. **Deploy to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will automatically detect the `vercel.json` configuration
   - Click "Deploy"

3. **Configuration**:
   - The `vercel.json` file is already configured
   - Python 3.11 runtime is specified
   - All routes are handled by `main.py`

### Vercel Deployment URL:
After deployment, you'll get a URL like: `https://your-project-name.vercel.app`

---

## 🚀 Option 2: Deploy to Render

### Prerequisites
- GitHub account
- Render account (free tier available)

### Steps to Deploy:

1. **Push your code to GitHub** (if not already done):
   ```bash
   git init
   git add .
   git commit -m "Initial backend commit"
   git remote add origin https://github.com/yourusername/yourrepo.git
   git push -u origin main
   ```

2. **Deploy to Render**:
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Use these settings:
     - **Name**: `kbot-backend` (or your preferred name)
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Click "Create Web Service"

3. **Alternative: Use render.yaml**:
   - The `render.yaml` file is already configured
   - Render will automatically detect and use this configuration

### Render Deployment URL:
After deployment, you'll get a URL like: `https://your-service-name.onrender.com`

---

## 🔗 Update Frontend Configuration

After deploying your backend, update your frontend to use the new backend URL:

### For Vercel deployment:
```javascript
const API_URL = "https://your-project-name.vercel.app";
```

### For Render deployment:
```javascript
const API_URL = "https://your-service-name.onrender.com";
```

---

## 🔧 CORS Configuration

The CORS middleware has been updated to allow your frontend domain:
- `https://mav-kbot-frontend-render.vercel.app` (your current frontend)
- Local development URLs for testing

If you deploy to a different domain, update the `allow_origins` list in `main.py`.

---

## 🐛 Troubleshooting

### Common Issues:

1. **CORS Errors**: 
   - Ensure your frontend URL is added to `allow_origins` in `main.py`

2. **Import Errors**: 
   - All dependencies are listed in `requirements.txt`
   - Both platforms will install them automatically

3. **WebSocket Issues**: 
   - WebSocket connections work on both Vercel and Render
   - Use `wss://` instead of `ws://` for secure connections

4. **Timeout Issues**: 
   - Vercel has a 10-second timeout for serverless functions
   - Render's free tier may have cold start delays

### Health Check Endpoints:
- `GET /` - Root endpoint
- `GET /api/timeframes` - Quick API test

---

## 📊 Performance Notes

### Vercel:
- ✅ Fast cold starts
- ✅ Global CDN
- ⚠️ 10-second execution limit
- ⚠️ Serverless (may have cold starts)

### Render:
- ✅ No execution time limits
- ✅ Persistent containers
- ⚠️ Slower cold starts on free tier
- ✅ Better for long-running processes

---

## 🔑 Environment Variables (if needed)

Both platforms support environment variables. Currently, your app doesn't require any, but you can add them later:

### Vercel:
- Set in project settings → Environment Variables

### Render:
- Set in service settings → Environment

---

## 🚀 Recommended Choice

**For this project, I recommend Vercel** because:
- Faster deployment process
- Better integration with your existing Vercel frontend
- Suitable for API endpoints
- Free tier is generous for API usage

Choose Render if you need:
- Longer execution times
- Background processes
- More control over the server environment