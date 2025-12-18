import os
import re
import json
import requests
from typing import List, Dict, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class RAGSystem:
    def __init__(self, api_key: str, docs_path: str = "../docs"):
        self.api_key = api_key
        self.docs_path = docs_path
        self.documents = []
        self.embeddings = []
        self.embedding_cache_file = "embeddings_cache.json"

    def load_documents(self):
        """Load all markdown documents from the docs directory"""
        docs_dir = os.path.join(os.path.dirname(__file__), self.docs_path)

        for filename in os.listdir(docs_dir):
            if filename.endswith('.md') and filename.startswith('chapter'):
                filepath = os.path.join(docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Extract title from frontmatter or first heading
                title = self._extract_title(content)

                # Remove frontmatter
                content = re.sub(r'^---\n.*?\n---\n', '', content, flags=re.DOTALL)

                # Split into chunks (by sections)
                chunks = self._split_into_chunks(content, filename)

                for chunk in chunks:
                    self.documents.append({
                        'filename': filename,
                        'title': title,
                        'content': chunk['content'],
                        'section': chunk['section']
                    })

        print(f"Loaded {len(self.documents)} document chunks")

    def _extract_title(self, content: str) -> str:
        """Extract title from markdown frontmatter or first heading"""
        # Try frontmatter first
        match = re.search(r'^---\n.*?title:\s*(.+?)\n.*?---', content, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try first heading
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        return "Unknown"

    def _split_into_chunks(self, content: str, filename: str) -> List[Dict]:
        """Split content into chunks by sections"""
        chunks = []

        # Split by ## headings
        sections = re.split(r'\n##\s+', content)

        # First section (before any ##) - usually the main heading and intro
        if sections[0].strip():
            chunks.append({
                'section': 'Introduction',
                'content': sections[0].strip()
            })

        # Remaining sections
        for section in sections[1:]:
            lines = section.split('\n', 1)
            section_title = lines[0].strip()
            section_content = lines[1].strip() if len(lines) > 1 else ""

            if section_content:
                chunks.append({
                    'section': section_title,
                    'content': f"## {section_title}\n\n{section_content}"
                })

        return chunks if chunks else [{'section': 'Full Document', 'content': content}]

    def generate_embeddings(self):
        """Generate embeddings for all documents using Gemini API"""
        # Try to load from cache first
        if os.path.exists(self.embedding_cache_file):
            try:
                with open(self.embedding_cache_file, 'r') as f:
                    cache = json.load(f)
                    if len(cache['embeddings']) == len(self.documents):
                        self.embeddings = np.array(cache['embeddings'])
                        print("Loaded embeddings from cache")
                        return
            except Exception as e:
                print(f"Could not load cache: {e}")

        print("Generating embeddings...")
        embeddings = []

        for i, doc in enumerate(self.documents):
            embedding = self._get_embedding(doc['content'])
            embeddings.append(embedding)

            if (i + 1) % 5 == 0:
                print(f"Generated {i + 1}/{len(self.documents)} embeddings")

        self.embeddings = np.array(embeddings)

        # Cache embeddings
        try:
            with open(self.embedding_cache_file, 'w') as f:
                json.dump({'embeddings': embeddings}, f)
            print("Embeddings cached successfully")
        except Exception as e:
            print(f"Could not cache embeddings: {e}")

    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini API"""
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "content": {
                    "parts": [{"text": text[:2048]}]  # Limit text length
                }
            }

            response = requests.post(url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()

            return data['embedding']['values']

        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Search for most relevant documents using semantic similarity"""
        query_embedding = self._get_embedding(query)
        query_embedding = np.array(query_embedding).reshape(1, -1)

        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': float(similarities[idx])
            })

        return results

    def generate_answer(self, query: str, context_docs: List[Dict]) -> str:
        """Generate answer using Gemini API with context"""
        # Build context from retrieved documents
        context = "\n\n".join([
            f"From {doc['document']['title']} - {doc['document']['section']}:\n{doc['document']['content']}"
            for doc in context_docs
        ])

        # Create prompt
        prompt = f"""You are a helpful assistant answering questions about a Physical AI textbook. Use the following context from the book to answer the user's question. If the answer is not in the context, say so politely and provide general guidance if possible.

Context from the textbook:
{context}

User Question: {query}

Answer:"""

        try:
            # Use Gemini 2.0 Flash API for generation
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={self.api_key}"
            headers = {"Content-Type": "application/json"}
            payload = {
                "contents": [{
                    "parts": [{"text": prompt}]
                }],
                "generationConfig": {
                    "temperature": 0.7,
                    "maxOutputTokens": 500
                }
            }

            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # Extract the generated text
            answer = data['candidates'][0]['content']['parts'][0]['text']
            return answer

        except requests.exceptions.HTTPError as e:
            print(f"Error generating answer: {e}")

            # If rate limited or API error, provide fallback with retrieved context
            if "429" in str(e) or "Too Many Requests" in str(e):
                fallback = f"⏳ The AI is temporarily rate-limited. Here are relevant excerpts from your textbook:\n\n{context}\n\n💡 Please try again in a few minutes for an AI-generated response."
                return fallback

            # For other errors, provide context as fallback
            if context.strip():
                return f"I found this relevant information from the textbook:\n\n{context}\n\n(Note: AI generation unavailable due to: {str(e)})"

            return f"I apologize, but I encountered an error: {str(e)}"
        except Exception as e:
            print(f"Error generating answer: {e}")
            # Provide context as fallback
            if context.strip():
                return f"Here's what I found in the textbook:\n\n{context}"
            return "I apologize, but I encountered an error processing your request."

    def chat(self, query: str) -> Dict:
        """Main chat function that performs RAG"""
        # Search for relevant documents
        relevant_docs = self.search(query, top_k=3)

        # Generate answer
        answer = self.generate_answer(query, relevant_docs)

        # Return response with sources
        return {
            'answer': answer,
            'sources': [
                {
                    'title': doc['document']['title'],
                    'section': doc['document']['section'],
                    'similarity': doc['similarity']
                }
                for doc in relevant_docs
            ]
        }
