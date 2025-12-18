# Deployment Guide - Physical AI Chatbot

This guide explains how to deploy your Physical AI textbook with the RAG chatbot to production.

## Architecture

- **Frontend (Docusaurus)**: Deploy to Vercel
- **Backend (FastAPI)**: Deploy to Render (or Railway/Fly.io)

## Step 1: Deploy Backend to Render

### Option A: Using Render Dashboard (Easiest)

1. **Sign up/Login to Render**: https://render.com

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `beenishmuhammaddin/physical-ai-textbook`

3. **Configure the Service**:
   ```
   Name: physical-ai-backend
   Region: Choose closest to your users
   Branch: master
   Root Directory: backend
   Runtime: Python 3
   Build Command: pip install -r requirements.txt
   Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT
   ```

4. **Add Environment Variable**:
   - Key: `GOOGLE_API_KEY`
   - Value: `AIzaSyDznVRtphVgTiZufPJ4Geu-xF7SLOepzeo` (or your API key)

5. **Deploy**:
   - Click "Create Web Service"
   - Wait 5-10 minutes for deployment
   - You'll get a URL like: `https://physical-ai-backend.onrender.com`

### Option B: Using render.yaml (Infrastructure as Code)

The `backend/render.yaml` file is already configured. Simply:

1. Go to Render Dashboard
2. Click "New +" → "Blueprint"
3. Connect your repo
4. It will auto-detect `render.yaml`
5. Add `GOOGLE_API_KEY` in environment variables
6. Click "Apply"

## Step 2: Deploy Frontend to Vercel

### Update API URL

Before deploying, update the backend URL in your chat widget:

1. Open: `src/components/ChatWidget/index.js`

2. Find line 11 and update:
```javascript
const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://physical-ai-backend.onrender.com'  // Replace with YOUR backend URL
  : 'http://localhost:8000';
```

### Deploy to Vercel

#### Option A: Vercel Dashboard (If already connected)

1. **Login to Vercel**: https://vercel.com
2. **Find your project**: "physical-ai-textbook"
3. **Trigger Redeploy**:
   - Go to Deployments tab
   - Click "Redeploy" on latest deployment
   - Or simply push to GitHub (auto-deploys)

#### Option B: Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy
vercel --prod
```

#### Option C: GitHub Auto-Deploy

Your Vercel is likely already connected to GitHub. Simply:

```bash
git add .
git commit -m "Configure production deployment"
git push origin master
```

Vercel will automatically deploy!

## Step 3: Verify Deployment

### Test Backend

Visit your backend URL:
```
https://your-backend.onrender.com/health
```

Should return: `{"status":"ok"}`

### Test Frontend

1. Visit your Vercel URL: `https://your-site.vercel.app`
2. Click the purple chat button
3. Ask: "What is Physical AI?"
4. You should see a response!

## Step 4: Configure Environment Variables (Vercel)

If needed, add environment variables on Vercel:

1. Go to Vercel Dashboard → Your Project
2. Settings → Environment Variables
3. Add any needed variables (currently none required for frontend)

## Alternative Backend Hosting Options

### Railway.app

1. Sign up: https://railway.app
2. New Project → Deploy from GitHub
3. Select `backend` folder
4. Add `GOOGLE_API_KEY` environment variable
5. Railway auto-detects Python and runs it

### Fly.io

1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
2. Login: `flyctl auth login`
3. From `backend/` directory: `flyctl launch`
4. Set secret: `flyctl secrets set GOOGLE_API_KEY=your_key`
5. Deploy: `flyctl deploy`

## Troubleshooting

### Backend Not Loading

- Check Render logs for errors
- Verify `GOOGLE_API_KEY` is set
- Ensure `requirements.txt` has all dependencies
- Check that startup command is correct

### Frontend Can't Connect to Backend

- Check CORS is enabled in `backend/main.py` (already configured)
- Verify backend URL in `ChatWidget/index.js`
- Check browser console for errors
- Ensure backend is running (visit `/health` endpoint)

### Rate Limiting Issues

- Gemini API has rate limits (60 requests/minute for free tier)
- Consider upgrading your Google Cloud API key
- The chatbot has fallback responses, so it still works!

### Embedding API 403 Errors

- Enable "Generative Language API" in Google Cloud Console
- Ensure API key has embedding permissions
- You can still use the chatbot without embeddings (uses basic search)

## Cost Estimates

### Free Tier (Perfect for Start)

- **Render**: Free tier (sleeps after 15min inactivity, wakes on request)
- **Vercel**: Free tier (unlimited deployments)
- **Gemini API**: Free tier (60 requests/minute)

**Total: $0/month** ✨

### If You Need More

- **Render Starter**: $7/month (always on, more resources)
- **Vercel Pro**: $20/month (more bandwidth, analytics)
- **Gemini API Pay-as-you-go**: Very cheap (~$0.0001 per request)

## Post-Deployment Checklist

- [ ] Backend deployed and accessible
- [ ] Frontend deployed to Vercel
- [ ] Chat widget appears on all pages
- [ ] Chatbot responds to questions
- [ ] Sources are displayed correctly
- [ ] Mobile responsive (test on phone)
- [ ] API key secured (not in code)
- [ ] Rate limit fallback working

## Updating After Deployment

To update your deployed site:

```bash
# Make your changes
git add .
git commit -m "Your update message"
git push origin master
```

Both Vercel and Render will auto-deploy your changes!

## Production Tips

1. **Monitor Usage**: Check Render and Vercel dashboards regularly
2. **API Limits**: Monitor your Gemini API usage in Google Cloud Console
3. **Logs**: Check Render logs if chatbot stops working
4. **Backups**: Your code is on GitHub - safe!
5. **Domain**: Consider adding a custom domain on Vercel

## Need Help?

- Render Docs: https://render.com/docs
- Vercel Docs: https://vercel.com/docs
- Your setup guide: `CHATBOT_SETUP.md`

---

**Your chatbot is ready for the world! 🚀**
