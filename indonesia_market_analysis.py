from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
import pickle
import time
import datetime
import requests
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import faiss
import PyPDF2
from openai import OpenAI
import google.generativeai as genai
import trafilatura
from bs4 import BeautifulSoup

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Indonesia Market Analysis API",
    description="API for Indonesia payments and QRIS market intelligence",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class MarketQueryRequest(BaseModel):
    query: str = Field(..., description="Market analysis query about Indonesia payments/QRIS")

class MarketQueryResponse(BaseModel):
    query: str
    analysis: str
    service: str = "indonesia_market_analysis"
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    service: str
    market_chunks_loaded: int
    gemini_available: bool
    pplx_available: bool
    serpapi_available: bool

# Indonesia Market Analysis System (embedded in main file)
class IndonesiaMarketAnalysis:
    def __init__(self):
        # Initialize OpenAI client (primary)
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.openai_client = OpenAI(api_key=openai_key)
        
        # Initialize Gemini client (fallback)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            self.gemini_available = True
        else:
            self.gemini_client = None
            self.gemini_available = False

        # Initialize external API keys
        self.pplx_api_key = os.getenv("PPLX_API_KEY")
        self.serpapi_key = os.getenv("SERPAPI_KEY")

        # Directory setup
        self.market_docs_dir = Path("market_research")
        self.market_embeddings_dir = Path("market_embeddings")

        # Configuration
        self.CHUNK_SIZE = 1500
        self.CHUNK_OVERLAP = 300
        self.TOP_K_RESULTS = 8
        self.EMBEDDING_MODEL = "text-embedding-3-small"

        # Storage for market research RAG
        self.market_chunks = []
        self.market_metadata = []
        self.market_embeddings = None
        self.market_faiss_index = None

        # Load existing embeddings if available
        self.load_market_embeddings()

    def load_market_embeddings(self) -> bool:
        try:
            chunks_file = self.market_embeddings_dir / "market_chunks_data.pkl"
            index_file = self.market_embeddings_dir / "market_faiss_index.bin"
            
            if not chunks_file.exists() or not index_file.exists():
                print("❌ No pre-built market embeddings found.")
                return False
                
            with open(chunks_file, "rb") as f:
                data = pickle.load(f)
                self.market_chunks = data['chunks']
                self.market_metadata = data['metadata']
            
            self.market_faiss_index = faiss.read_index(str(index_file))
            print(f"✅ Loaded {len(self.market_chunks)} market research chunks")
            return True
        except Exception as e:
            print(f"❌ Failed to load market embeddings: {e}")
            return False

    def query_market_research(self, query: str) -> List[Tuple[str, Dict, float]]:
        if not self.market_chunks or self.market_faiss_index is None:
            return []
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.EMBEDDING_MODEL, 
                input=[query]
            )
            query_embedding = np.array([response.data[0].embedding])
            
            search_k = min(self.TOP_K_RESULTS, len(self.market_chunks))
            distances, indices = self.market_faiss_index.search(query_embedding.astype(np.float32), search_k)
            
            results = []
            for i, d in zip(indices[0], distances[0]):
                results.append((self.market_chunks[i], self.market_metadata[i], float(d)))
            
            return results
        except Exception:
            return []
        
    def pplx_research(self, query: str) -> List[str]:
        if not self.pplx_api_key:
            return []
        
        url = "https://api.perplexity.ai/chat/completions"
        headers = {"Authorization": f"Bearer {self.pplx_api_key}"}
        payload = {
            "model": "sonar-pro",
            "messages": [{"role": "user", "content": f"Deep research on {query}"}],
            "max_tokens": 700,
        }
        
        for attempt in range(3):
            try:
                r = requests.post(url, json=payload, headers=headers, timeout=(10, 120))
                r.raise_for_status()
                response_data = r.json()
                citations = response_data.get("citations", [])
                return citations
                
            except requests.RequestException:
                if attempt == 2:
                    return []
                time.sleep(2 ** attempt)

    def serpapi_news(self, query: str) -> List[str]:
        if not self.serpapi_key:
            return []
        
        params = {
            "engine": "google_news",
            "q": query,
            "num": 10,
            "api_key": self.serpapi_key
        }
        
        try:
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=20)
            r.raise_for_status()
            news_results = r.json().get("news_results", [])
            return [article["link"] for article in news_results if "link" in article]
        except Exception:
            return []

    def scrape_clean(self, url: str, timeout: int = 25) -> str:
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            
            text = trafilatura.extract(r.text)
            if not text:
                soup = BeautifulSoup(r.text, "html.parser")
                text = soup.get_text(" ", strip=True)
            
            return text[:4000] if len(text) > 300 else ""
        except Exception:
            return ""

    def get_real_time_insights(self, query: str) -> Dict:
        pplx_urls = self.pplx_research(query)
        serp_urls = self.serpapi_news(query) if not pplx_urls else []
        
        all_urls = pplx_urls + serp_urls
        if not all_urls:
            return {"content": "", "sources": []}
        
        all_content = []
        processed_sources = []
        
        for url in all_urls[:12]:
            content = self.scrape_clean(url)
            if content:
                all_content.append(f"--- Source: {url} ---\n{content}\n")
                processed_sources.append(url)
        
        combined_content = "\n".join(all_content)
        
        return {
            "content": combined_content,
            "sources": processed_sources
        }

    def generate_with_fallback(self, system_prompt: str, user_prompt: str) -> str:
        # First, try OpenAI
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=700
            )
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a quota/rate limit error
            if any(keyword in error_str for keyword in ['quota', 'rate limit', 'insufficient_quota', '429', 'exceeded']):
                print(f"⚠️ OpenAI quota/rate limit exceeded: {e}")
                
                # Try Gemini fallback
                if self.gemini_available:
                    try:
                        print("🔄 Falling back to Gemini...")
                        full_prompt = f"{system_prompt}\n\n{user_prompt}"
                        
                        response = self.gemini_client.generate_content(
                            full_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.1,
                                max_output_tokens=700
                            )
                        )
                        print("✅ Gemini fallback analysis completed successfully")
                        return response.text
                        
                    except Exception as gemini_error:
                        print(f"❌ Gemini fallback also failed: {gemini_error}")
                        return f"❌ Both OpenAI and Gemini failed. OpenAI: {e}, Gemini: {gemini_error}"
                else:
                    return f"❌ OpenAI quota exceeded and Gemini fallback not available: {e}"
            else:
                print(f"❌ OpenAI error (not quota-related): {e}")
                return f"❌ Error generating analysis: {e}"

    def comprehensive_analysis(self, query: str) -> str:
        # 1. Query market research documents
        market_chunks = self.query_market_research(query)
        
        market_context = ""
        if market_chunks:
            for text, meta, score in market_chunks[:4]:
                market_context += f"--- Research Document: {meta['filename']} ---\n{text[:1000]}...\n\n"
        
        # 2. Get real-time insights
        real_time_data = self.get_real_time_insights(query)
        real_time_content = real_time_data["content"]
        
        if len(real_time_content) > 4000:
            real_time_content = real_time_content[:4000] + "\n... [Content truncated]"
        
        # 3. SPECIALIZED SYSTEM PROMPT FOR INDONESIA PAYMENTS & QRIS
        system_prompt = """You are an expert analyst specializing in Indonesia payments and QRIS market intelligence.

CORE PRINCIPLES:
1. Answer exactly what is asked - no more, no less
2. Use the most appropriate format for the information (table, list, or paragraph)
3. Be accurate and concise
4. If data is missing, clearly state "Information not available in sources"

EXPERTISE AREAS:
- QRIS transaction data and market metrics
- Indonesian payment regulations and policies
- Digital payment infrastructure and devices
- Competitor landscape (GoPay, OVO, Dana, LinkAja, etc.)
- Merchant adoption and pricing models

RESPONSE APPROACH:
- For data requests: Provide the specific data requested
- For comparisons: Use whatever format makes the information clearest
- For analysis: Focus on key insights relevant to the query
- Always prioritize accuracy over completeness"""

        # Prepare context
        context_parts = []
        if market_context:
            context_parts.append(f"PRIMARY RESEARCH DOCUMENTS:\n{market_context}")
        if real_time_content:
            context_parts.append(f"RECENT INFORMATION SOURCES:\n{real_time_content}")
        
        combined_context = "\n\n".join(context_parts) if context_parts else "Limited context available for this query."
        
        # Ensure context fits within limits
        if len(combined_context) > 6000:
            combined_context = combined_context[:6000] + "\n... [Context truncated to fit limits]"

        user_prompt = f"""
AVAILABLE INFORMATION:
{combined_context}

QUERY: {query}

Please provide a comprehensive analysis based on the available information, focusing on Indonesia payments and QRIS market intelligence.
"""

        # Use the fallback mechanism
        return self.generate_with_fallback(system_prompt, user_prompt)

# Initialize the market analysis system globally
market_system = IndonesiaMarketAnalysis()

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    return {
        "service": "Indonesia Market Analysis API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="Indonesia Market Analysis",
        market_chunks_loaded=len(market_system.market_chunks),
        gemini_available=market_system.gemini_available,
        pplx_available=bool(market_system.pplx_api_key),
        serpapi_available=bool(market_system.serpapi_key)
    )

@app.post("/api/analyze", response_model=MarketQueryResponse)
async def analyze_market(request: MarketQueryRequest):
    try:
        result = market_system.comprehensive_analysis(request.query)
        
        return MarketQueryResponse(
            query=request.query,
            analysis=result,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8001))
    uvicorn.run("indonesia_market_analysis:app", host="0.0.0.0", port=port, reload=False)
