# Physical AI Textbook Chatbot - Setup Guide

This guide explains how to set up and use the RAG-based chatbot for your Docusaurus Physical AI textbook.

## What's Been Implemented

A complete RAG (Retrieval-Augmented Generation) chatbot system with:

1. **Backend (FastAPI)**:
   - Document processing and chunking system
   - Vector embeddings using Gemini Embedding API
   - Semantic search with cosine similarity
   - RAG endpoint that generates context-aware answers
   - Embedding caching for faster startups

2. **Frontend (React + Docusaurus)**:
   - Beautiful floating chat widget (bottom-right corner)
   - Smooth animations and responsive design
   - Message history and typing indicators
   - Source citations showing which chapters were used

## Setup Instructions

### 1. Backend Setup

#### Install Python Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Configure API Key
Make sure your `.env` file in the root directory contains:
```
GOOGLE_API_KEY=your_api_key_here
```

**Important**: The Gemini API key needs to have access to:
- Embedding API (`embedding-001` model)
- Generative AI API (`gemini-pro` model)

#### Fix API Endpoints (if needed)
If you encounter 403/404 errors, you may need to update the Gemini API endpoints in `backend/rag_system.py`.

Current endpoints (lines 125 and 185):
```python
# Embedding API
url = f"https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key={self.api_key}"

# Generation API
url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={self.api_key}"
```

You might need to try:
- `v1` instead of `v1beta`
- Different model names like `text-embedding-004` or `gemini-1.5-flash`

Check the latest Gemini API documentation at: https://ai.google.dev/docs

#### Start the Backend
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

The backend will:
1. Load all chapter markdown files from `docs/`
2. Split them into chunks by section
3. Generate embeddings (this may take a minute on first run)
4. Cache embeddings for faster future startups
5. Start the API server on http://localhost:8000

### 2. Frontend Setup

#### Install Node Dependencies
```bash
npm install
```

#### Update Backend URL (for production)
Edit `src/components/ChatWidget/index.js` line 11:
```javascript
const API_URL = process.env.NODE_ENV === 'production'
  ? 'https://your-backend-url.com'  // Update this!
  ? 'http://localhost:8000';
```

#### Start Docusaurus
```bash
npm start
```

The site will open at http://localhost:3000 with the chatbot widget in the bottom-right corner.

## Usage

### Testing the Chatbot

1. Open your Docusaurus site (http://localhost:3000)
2. Click the purple circular chat button in the bottom-right
3. Try asking questions like:
   - "What is Physical AI?"
   - "Explain humanoid robotics"
   - "What are the types of sensors used in robotics?"
   - "How do control systems work?"

### How It Works

1. **User asks a question** → Frontend sends to `/chat` endpoint
2. **Query is embedded** → Converted to vector representation
3. **Semantic search** → Finds top 3 most relevant document chunks
4. **Context building** → Creates prompt with relevant book sections
5. **LLM generates answer** → Gemini Pro generates contextual response
6. **Response with sources** → User sees answer + which chapters were referenced

## Troubleshooting

### Backend Issues

**Problem**: `403 Forbidden` or `404 Not Found` for Gemini API
**Solution**:
- Verify your API key is valid and has the right permissions
- Check if you need to enable the Generative Language API in Google Cloud Console
- Try updating the API endpoints to use `v1` instead of `v1beta`

**Problem**: `RAG system not initialized`
**Solution**:
- Check that your `.env` file has `GOOGLE_API_KEY` set
- Ensure the backend can access the `../docs` directory
- Check backend logs for initialization errors

**Problem**: Embeddings are all zeros (similarity scores of 0.0)
**Solution**:
- Delete `backend/embeddings_cache.json` and restart
- Check that embedding API calls are succeeding (check logs)
- Try a different embedding model or approach

### Frontend Issues

**Problem**: Chat widget not appearing
**Solution**:
- Clear your browser cache and reload
- Check browser console for errors
- Verify `src/theme/Root.js` exists and is correct

**Problem**: "Error processing chat request"
**Solution**:
- Ensure backend is running on http://localhost:8000
- Check browser console Network tab for failed requests
- Verify CORS is enabled in backend (already configured)

## Files Created/Modified

### Backend
- `backend/rag_system.py` - RAG implementation
- `backend/main.py` - Added CORS, chat endpoint, RAG initialization
- `backend/requirements.txt` - Added numpy, scikit-learn, fastapi-cors

### Frontend
- `src/components/ChatWidget/index.js` - Chat widget component
- `src/components/ChatWidget/styles.module.css` - Widget styling
- `src/theme/Root.js` - Integration with Docusaurus

## Next Steps

### Improvements You Can Make

1. **Better Error Handling**:
   - Add retry logic for API calls
   - Better error messages for users
   - Fallback responses when API is down

2. **Enhanced Features**:
   - Chat history persistence (localStorage)
   - Multi-turn conversations with context
   - Feedback buttons (thumbs up/down)
   - Export chat transcript

3. **Performance Optimization**:
   - Use a proper vector database (Pinecone, Weaviate, ChromaDB)
   - Implement streaming responses
   - Add rate limiting
   - Optimize chunk sizes

4. **Production Deployment**:
   - Deploy backend to a cloud service (Render, Railway, Fly.io)
   - Update frontend API_URL to production backend
   - Set up environment variables securely
   - Add authentication if needed

## API Endpoints

### Backend Endpoints

- `GET /` - Health check
- `GET /health` - Status check
- `POST /chat` - RAG chatbot endpoint
  ```json
  Request:
  {
    "message": "What is Physical AI?"
  }

  Response:
  {
    "message": "What is Physical AI?",
    "answer": "Physical AI is...",
    "sources": [
      {
        "title": "Introduction",
        "section": "Introduction",
        "similarity": 0.89
      }
    ]
  }
  ```

## Support

If you encounter issues:
1. Check the backend logs for detailed error messages
2. Verify your Gemini API key has the right permissions
3. Try the Gemini API playground to test your key: https://ai.google.dev/
4. Check Docusaurus build logs if the frontend doesn't load

## License

Same as your Physical AI Textbook project.
