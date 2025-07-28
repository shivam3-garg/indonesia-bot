from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import os
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import faiss
import PyPDF2
from openai import OpenAI
import google.generativeai as genai  # Added for Gemini fallback

# Load environment variables
load_dotenv()

app = FastAPI(
    title="QRIS RAG API",
    description="API for querying QRIS documentation using RAG",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models (keep all your existing models)
class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about QRIS documents")
    debug: bool = Field(False, description="Enable debug mode")

class SearchRequest(BaseModel):
    keyword: str = Field(..., description="Keyword to search for")
    limit: int = Field(10, description="Maximum results", ge=1, le=50)

class Source(BaseModel):
    filename: str
    chunk_id: int
    relevance_score: float
    contains_table: bool
    table_types: List[str]

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[Source]
    service: str = "qris_rag"

class SearchResponse(BaseModel):
    keyword: str
    matches_found: int
    results: List[dict]
    service: str = "qris_rag"

class HealthResponse(BaseModel):
    status: str
    service: str
    chunks_loaded: int
    embeddings_available: bool
    gemini_available: bool  # Added Gemini status

# Enhanced QRIS RAG System with Gemini Fallback
class QRISRAGSystem:
    def __init__(self):
        # Initialize OpenAI client (primary)
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.openai_client = OpenAI(api_key=api_key)

        # Initialize Gemini client (fallback)
        gemini_key = os.getenv("GEMINI_API_KEY")
        if gemini_key:
            genai.configure(api_key=gemini_key)
            self.gemini_client = genai.GenerativeModel('gemini-pro')
            self.gemini_available = True
            print("✅ Gemini fallback initialized")
        else:
            self.gemini_client = None
            self.gemini_available = False
            print("⚠️ GEMINI_API_KEY not found. Gemini fallback unavailable.")

        self.documents_dir = Path("documents")
        self.embeddings_dir = Path("embeddings")
        
        # Configuration
        self.CHUNK_SIZE = 2000
        self.CHUNK_OVERLAP = 400
        self.TOP_K_RESULTS = 12
        self.MAX_CONTEXT_LENGTH = 16000
        self.EMBEDDING_MODEL = "text-embedding-3-small"

        # Storage
        self.chunks = []
        self.chunk_metadata = []
        self.embeddings = None
        self.faiss_index = None
        
        # Load embeddings on startup
        self.load_embeddings()

    def load_embeddings(self) -> bool:
        try:
            chunks_file = self.embeddings_dir / "chunks_data.pkl"
            index_file = self.embeddings_dir / "faiss_index.bin"
            
            if not chunks_file.exists() or not index_file.exists():
                print("❌ No pre-built embeddings found.")
                return False
                
            with open(chunks_file, "rb") as f:
                data = pickle.load(f)
                self.chunks, self.chunk_metadata = data['chunks'], data['chunk_metadata']
            
            self.faiss_index = faiss.read_index(str(index_file))
            print(f"✅ Loaded {len(self.chunks)} chunks")
            return True
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            return False

    def retrieve_relevant_chunks(self, query: str) -> List[Tuple[str, Dict, float]]:
        try:
            # Try OpenAI embeddings first
            response = self.openai_client.embeddings.create(
                model=self.EMBEDDING_MODEL, 
                input=[query]
            )
            query_embedding = np.array([response.data[0].embedding])
            
            search_k = min(self.TOP_K_RESULTS * 3, len(self.chunks))
            distances, indices = self.faiss_index.search(query_embedding.astype(np.float32), search_k)
            
            results = [(self.chunks[i], self.chunk_metadata[i], float(d)) for i, d in zip(indices[0], distances[0])]
            
            # Enhanced query-specific prioritization (keep your existing logic)
            query_upper = query.upper()
            
            if 'PAYMENT CREDIT API' in query_upper or 'PAYMENT CREDIT' in query_upper:
                payment_credit_chunks = []
                other_api_chunks = []
                general_chunks = []
                
                for chunk_text, meta, score in results:
                    chunk_upper = chunk_text.upper()
                    if ('PAYMENT CREDIT' in chunk_upper or 'QRPAYMENTCREDIT' in chunk_upper or 
                        'PAYMENTCREDIT' in chunk_upper):
                        payment_credit_chunks.append((chunk_text, meta, score))
                    elif 'api_spec' in meta.get('table_types', []) or 'API' in chunk_upper:
                        other_api_chunks.append((chunk_text, meta, score))
                    else:
                        general_chunks.append((chunk_text, meta, score))
                
                results = payment_credit_chunks + other_api_chunks + general_chunks
                
            elif any(keyword in query_upper for keyword in ['MDR', 'RATE', 'FEE', 'PRICING', 'CHARGE', 'DISTRIBUTION']):
                pricing_chunks = [r for r in results if 'pricing' in r[1].get('table_types', []) or 'distribution' in r[1].get('table_types', [])]
                other_table_chunks = [r for r in results if r[1].get('contains_table', False) and r not in pricing_chunks]
                non_table_chunks = [r for r in results if not r[1].get('contains_table', False)]
                results = pricing_chunks + other_table_chunks + non_table_chunks
                
            elif any(keyword in query_upper for keyword in ['API', 'FIELD', 'REQUIRED', 'CONTRACT', 'PARAMETER']):
                api_chunks = [r for r in results if 'api_spec' in r[1].get('table_types', [])]
                other_table_chunks = [r for r in results if r[1].get('contains_table', False) and r not in api_chunks]
                non_table_chunks = [r for r in results if not r[1].get('contains_table', False)]
                results = api_chunks + other_table_chunks + non_table_chunks
            
            return results[:self.TOP_K_RESULTS]
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a quota/rate limit error
            if any(keyword in error_str for keyword in ['quota', 'rate limit', 'insufficient_quota', '429', 'exceeded']):
                print(f"⚠️ OpenAI quota/rate limit exceeded for embeddings: {e}")
                print("🔄 Falling back to keyword search...")
                
                # Fallback to keyword search when embeddings fail
                return self.keyword_search_fallback(query)
            else:
                print(f"❌ Retrieval error: {e}")
                return []

    def keyword_search_fallback(self, query: str) -> List[Tuple[str, Dict, float]]:
        """Fallback to keyword search when embeddings fail"""
        # Extract keywords from query
        query_words = query.upper().replace('?', '').replace(',', '').split()
        
        # Filter out common words
        stop_words = {'WHAT', 'IS', 'ARE', 'THE', 'HOW', 'DO', 'DOES', 'CAN', 'WILL', 'FOR', 'OF', 'IN', 'ON', 'AT', 'TO', 'A', 'AN', 'AND', 'OR'}
        keywords = [word for word in query_words if word not in stop_words and len(word) > 2]
        
        # Score chunks based on keyword matches
        scored_chunks = []
        for i, chunk in enumerate(self.chunks):
            chunk_upper = chunk.upper()
            score = 0
            
            # Calculate relevance score based on keyword frequency
            for keyword in keywords:
                if keyword in chunk_upper:
                    # Give higher score for exact matches
                    score += chunk_upper.count(keyword) * 2
                    
                    # Bonus for table data containing the keyword
                    if '[TABLE_DATA]' in chunk and keyword in chunk_upper:
                        score += 5
            
            if score > 0:
                # Convert to distance (lower is better for consistency with FAISS)
                distance = 1.0 / (1.0 + score)
                scored_chunks.append((chunk, self.chunk_metadata[i], distance))
        
        # Sort by relevance (lowest distance first) and return top results
        scored_chunks.sort(key=lambda x: x[2])
        return scored_chunks[:self.TOP_K_RESULTS]

    def generate_answer_with_fallback(self, query: str, context: str) -> str:
        """Generate answer with OpenAI primary and Gemini fallback"""
        system_prompt = """You are an expert assistant for Indonesia QRIS (Quick Response Code Indonesian Standard) international acquiring system. 

Your role is to answer questions ONLY based on the provided document context. Follow these rules:

1. Answer ONLY based on the provided context from QRIS documents
2. If information is not in the context, clearly state "This information is not available in the provided documents"
3. For technical details (like API fields, MDR rates, settlement times), be precise and quote specific values
4. When you see tabular data or pricing information, present it in a clear, structured format using tables or lists
5. Pay special attention to sections marked as "[CONTAINS" or "[TABLE_DATA]"

CRITICAL COMPLETENESS RULES:
6. For API field questions: You MUST list EVERY SINGLE field mentioned across ALL provided chunks. Do not stop early. Scan through every chunk completely.
7. For distribution/percentage questions: You MUST include ALL percentages and parties mentioned. Add up percentages to verify completeness (they should total 100%).
8. For pricing/rate questions: Include ALL categories, merchant types, and rates mentioned across all chunks.

SPECIFIC API INSTRUCTIONS:
9. When asked about a specific API (like "Payment Credit API"), focus ONLY on chunks that mention that specific API name
10. Look for exact API names like "QRPaymentCreditRQ", "QRPaymentCreditRS", "Payment Credit Request", "Payment Credit Response"
11. Distinguish between different APIs - don't mix fields from different API specifications
12. If chunks contain multiple APIs, clearly separate the fields by API type

13. When referencing information, mention which document it comes from when possible
14. If you find structured data like pricing tables, recreate them in a readable format
15. EXHAUSTIVE SCANNING: Read through every single chunk provided in the context completely before answering
16. VERIFICATION: For lists or tables, count items and verify completeness - if you see "24 fields" mentioned somewhere but only list 20, that's incomplete

FORMAT RULES:
- For API specifications: Create a numbered list with field name, description, type, and length
- For distribution information: Present as "Party Name: Percentage" and verify all percentages add to 100%
- For incomplete information: Explicitly state "Based on the provided context, X items were found, but the documentation may contain additional items not included in these chunks"

IMPORTANT: Your goal is COMPLETE and EXHAUSTIVE answers. If a question asks for "required fields" and there are 30 fields, list all 30. If there are 5 distribution parties, list all 5 with their percentages."""

        user_prompt = f"""Based on the following QRIS documentation context, please answer the question.

CONTEXT (READ ALL CHUNKS COMPLETELY):
{context}

QUESTION: {query}

INSTRUCTIONS:
- Read through EVERY SINGLE chunk above completely before answering
- For lists (like API fields), count the total items and list ALL of them
- For percentages/distributions, ensure they add up to 100% and list ALL parties
- If you find partial information in early chunks, continue reading all chunks for complete information
- Be EXHAUSTIVE and COMPLETE in your answer
- If the question asks about a specific API, focus only on that API and don't mix with other APIs
- Look for complete field lists, tables, and specifications within the chunks

ANSWER:"""

        # Try OpenAI first
        try:
            print("🤖 Attempting answer generation with OpenAI...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1500
            )
            print("✅ OpenAI answer generation successful")
            return response.choices[0].message.content
            
        except Exception as e:
            error_str = str(e).lower()
            
            # Check if it's a quota/rate limit error
            if any(keyword in error_str for keyword in ['quota', 'rate limit', 'insufficient_quota', '429', 'exceeded']):
                print(f"⚠️ OpenAI quota/rate limit exceeded for chat: {e}")
                
                # Try Gemini fallback
                if self.gemini_available:
                    try:
                        print("🔄 Falling back to Gemini for answer generation...")
                        # Combine prompts for Gemini
                        full_prompt = f"{system_prompt}\n\n{user_prompt}"
                        
                        response = self.gemini_client.generate_content(
                            full_prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.1,
                                max_output_tokens=1500
                            )
                        )
                        print("✅ Gemini fallback answer generation successful")
                        return response.text
                        
                    except Exception as gemini_error:
                        print(f"❌ Gemini fallback also failed: {gemini_error}")
                        return f"❌ Both OpenAI and Gemini failed. OpenAI: {e}, Gemini: {gemini_error}"
                else:
                    return f"❌ OpenAI quota exceeded and Gemini fallback not available: {e}"
            else:
                print(f"❌ OpenAI error (not quota-related): {e}")
                return f"❌ Error generating answer: {e}"

    def create_context(self, chunks: List[Tuple[str, Dict, float]]) -> str:
        context, total = [], 0
        for text, meta, distance in chunks:
            table_types = meta.get('table_types', [])
            table_info = f" [CONTAINS: {', '.join(table_types).upper()} TABLE]" if table_types else (" [CONTAINS TABULAR DATA]" if meta.get('contains_table', False) else "")
            
            section = f"\n--- Document: {meta['filename']} (Chunk {meta['chunk_id']}, Relevance Score: {distance:.3f}){table_info} ---\n{text}"
            
            if total + len(section) > self.MAX_CONTEXT_LENGTH:
                break
            context.append(section)
            total += len(section)
        return "\n".join(context)

    def search_chunks_by_keyword(self, keyword: str) -> List[Tuple[str, Dict]]:
        matching_chunks = []
        keyword_upper = keyword.upper()
        
        for i, chunk in enumerate(self.chunks):
            if keyword_upper in chunk.upper():
                matching_chunks.append((chunk, self.chunk_metadata[i]))
        
        return matching_chunks

    def query_with_debug(self, question: str) -> Dict:
        if not self.chunks:
            return {"answer": "❌ No documents loaded.", "context": "", "sources": []}
        
        chunks = self.retrieve_relevant_chunks(question)
        
        if not chunks:
            return {"answer": "No relevant information found.", "context": "", "sources": []}
        
        context = self.create_context(chunks)
        answer = self.generate_answer_with_fallback(question, context)  # Using fallback method
        sources = [{"filename": m["filename"], "chunk_id": m["chunk_id"], "relevance_score": s, 
                   "contains_table": m.get('contains_table', False), "table_types": m.get('table_types', [])}
                   for _, m, s in chunks]
        
        return {"answer": answer, "context": context, "sources": sources}

    def query(self, question: str) -> Dict:
        if not self.chunks:
            return {"answer": "❌ No documents loaded.", "context": "", "sources": []}
        chunks = self.retrieve_relevant_chunks(question)
        if not chunks:
            return {"answer": "No relevant information found.", "context": "", "sources": []}
        context = self.create_context(chunks)
        answer = self.generate_answer_with_fallback(question, context)  # Using fallback method
        sources = [{"filename": m["filename"], "chunk_id": m["chunk_id"], "relevance_score": s, 
                   "contains_table": m.get('contains_table', False), "table_types": m.get('table_types', [])}
                   for _, m, s in chunks]
        return {"answer": answer, "context": context, "sources": sources}

# Initialize the RAG system globally
rag_system = QRISRAGSystem()

# API Endpoints
@app.get("/", response_model=dict)
async def root():
    return {
        "service": "QRIS RAG API",
        "version": "1.0.0",
        "status": "active",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        service="QRIS RAG System",
        chunks_loaded=len(rag_system.chunks),
        embeddings_available=rag_system.faiss_index is not None,
        gemini_available=rag_system.gemini_available  # Added Gemini status
    )

@app.post("/api/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    try:
        if request.debug:
            result = rag_system.query_with_debug(request.question)
        else:
            result = rag_system.query(request.question)
        
        return QueryResponse(
            question=request.question,
            answer=result['answer'],
            sources=result['sources']
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )

@app.post("/api/search", response_model=SearchResponse)
async def search_keywords(request: SearchRequest):
    try:
        matching_chunks = rag_system.search_chunks_by_keyword(request.keyword)
        
        results = []
        for chunk, meta in matching_chunks[:request.limit]:
            results.append({
                "filename": meta['filename'],
                "chunk_id": meta['chunk_id'],
                "preview": chunk[:200] + "..." if len(chunk) > 200 else chunk,
                "contains_table": meta.get('contains_table', False),
                "table_types": meta.get('table_types', [])
            })
        
        return SearchResponse(
            keyword=request.keyword,
            matches_found=len(matching_chunks),
            results=results
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 8000))
    uvicorn.run("main_openai_embeddings:app", host="0.0.0.0", port=port, reload=False)
