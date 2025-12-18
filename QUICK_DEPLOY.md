# 🚀 Quick Deployment Guide

Your chatbot is ready to deploy! Follow these simple steps.

## ✅ What's Already Done

- ✅ Frontend code ready with chatbot
- ✅ Backend code ready
- ✅ GitHub repo updated
- ✅ Vercel connected (will auto-deploy)

## 🎯 Deploy in 3 Steps (10 Minutes Total)

### Step 1: Deploy Backend (5 minutes)

**Go to Render.com:**

1. Visit: https://render.com/
2. Click **"Get Started"** (sign up with GitHub)
3. Click **"New +"** → **"Web Service"**
4. Select your repo: **beenishmuhammaddin/physical-ai-textbook**

**Configure:**
```
Name: physical-ai-backend
Region: Singapore (or closest to you)
Branch: master
Root Directory: backend
Runtime: Python 3

Build Command: pip install -r requirements.txt
Start Command: uvicorn main:app --host 0.0.0.0 --port $PORT

Instance Type: Free
```

**Add Environment Variable:**
- Click "Advanced"
- Click "Add Environment Variable"
- Key: `GOOGLE_API_KEY`
- Value: `AIzaSyDznVRtphVgTiZufPJ4Geu-xF7SLOepzeo`

**Deploy:**
- Click **"Create Web Service"**
- Wait 5 minutes for build
- Copy your URL: `https://physical-ai-backend.onrender.com`

### Step 2: Update Frontend URL (2 minutes)

**If your Render URL is different:**

Open `src/components/ChatWidget/index.js` (line 12) and update:
```javascript
? 'https://YOUR-ACTUAL-RENDER-URL.onrender.com'
```

Then push to GitHub:
```bash
git add .
git commit -m "Update backend URL"
git push origin master
```

### Step 3: Wait for Vercel (Automatic!)

Vercel will automatically:
- ✅ Detect your GitHub push
- ✅ Build your site
- ✅ Deploy the update
- ✅ Chatbot will be live!

**Check Vercel Dashboard:** https://vercel.com/dashboard

You'll see the deployment in progress. Takes 2-3 minutes.

## 🎉 Done!

Visit your Vercel URL and click the purple chat button!

## 📱 Your URLs

- **Frontend (Vercel)**: Check your Vercel dashboard
- **Backend (Render)**: https://physical-ai-backend.onrender.com
- **API Health Check**: https://physical-ai-backend.onrender.com/health

## ⚡ Alternative: Even Faster with Railway

If Render is slow, try Railway:

1. Go to https://railway.app
2. "New Project" → "Deploy from GitHub"
3. Select your repo
4. Choose `backend` folder
5. Add `GOOGLE_API_KEY` environment variable
6. Deploy! (Faster than Render)

## 🔧 Troubleshooting

**Chatbot shows error:**
- Wait 1-2 minutes for backend to wake up (free tier sleeps)
- Check backend is running: visit `your-backend-url/health`
- Should return: `{"status":"ok"}`

**Vercel not deploying:**
- Check Vercel dashboard for errors
- Make sure you pushed to GitHub
- Vercel auto-deploys from GitHub

**Backend not working:**
- Check Render logs for errors
- Verify `GOOGLE_API_KEY` is set correctly
- Ensure build command ran successfully

## 💡 Pro Tips

1. **First request may be slow** - Free tier sleeps after 15 min inactivity
2. **Keep it awake** - Use UptimeRobot to ping every 14 minutes (free)
3. **Monitor usage** - Check Render dashboard for logs
4. **Custom domain** - Add your domain in Vercel settings

## 📊 Free Tier Limits

- **Render**: 750 hours/month (enough for 24/7)
- **Vercel**: Unlimited deployments
- **Gemini API**: 60 requests/minute

**All FREE!** 🎉

---

Need help? Check `DEPLOYMENT.md` for detailed guide.
