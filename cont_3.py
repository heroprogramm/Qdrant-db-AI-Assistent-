from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import logging
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import traceback
# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o"

app = FastAPI(
    title="PDF QA System",
    description="Process PDFs and ask questions with context",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Qdrant setup
qdrant_client = QdrantClient(host="localhost", port=6333)
COLLECTION_NAME = "pdf_chunks"

class QA(BaseModel):
    question: str
    answer: str

class QuestionInput(BaseModel):
    question: str

class UploadResponse(BaseModel):
    message: str
    chunks: int

class AnswerResponse(BaseModel):
    question: str
    answer: str
    history: List[QA]

def detect_user_correction(text: str) -> bool:
    corrections = ["not talking about", "that's not", "i meant", "different topic", 
                   "ignore that", "change topic", "not what i meant", 
                   "wrong answer", "no, i'm asking"]
    return any(phrase in text.lower() for phrase in corrections)

def extract_text_from_pdf(file: bytes) -> str:
    try:
        pdf = fitz.open(stream=file, filetype="pdf")
        text = ""
        for page in pdf:
            text += page.get_text()
        return text
    except Exception as e:
        logger.error(f"PDF error: {e}")
        raise ValueError("Failed to extract text from PDF")

def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks: List[str]) -> dict:
    try:
        vectors = embedder.encode(chunks).tolist()

        if qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.delete_collection(COLLECTION_NAME)

        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )

        points = [
            PointStruct(id=str(uuid.uuid4()), vector=vector, payload={"chunk": chunks[i]})
            for i, vector in enumerate(vectors)
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

        tfidf = TfidfVectorizer().fit(chunks)
        tfidf_matrix = tfidf.transform(chunks)

        return {"tfidf": tfidf, "tfidf_matrix": tfidf_matrix, "chunks": chunks}
    except Exception as e:
        logger.error(f"Qdrant vector store error: {e}")
        raise RuntimeError("Failed to create Qdrant vector store")

def retrieve_relevant_chunks(query: str, store: dict, chunks: List[str], top_k=3) -> List[str]:
    try:
        query_vec = embedder.encode([query])[0]
        results = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vec,
            limit=top_k,
            with_payload=True
        )
        sem_chunks = [r.payload["chunk"] for r in results]

        query_tfidf = store["tfidf"].transform([query])
        sim_scores = cosine_similarity(query_tfidf, store["tfidf_matrix"]).flatten()
        top_bm25_idx = sim_scores.argsort()[::-1][:top_k]
        keyword_chunks = [chunks[i] for i in top_bm25_idx if i < len(chunks)]

        return list(dict.fromkeys(sem_chunks + keyword_chunks))[:top_k]
    except Exception as e:
        logger.error(f"Qdrant hybrid retrieval error: {e}")
        raise RuntimeError("Chunk retrieval failed")

def generate_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n---\n".join(list(dict.fromkeys(context_chunks))[:3])
    prompt = f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"].strip()
        else:
            logger.error(f"OpenAI API error: {response.text}")
            return "Sorry, LLM failed to generate a response."
    except Exception as e:
        logger.error(f"OpenAI request error: {e}")
        return "Sorry, LLM request failed."




@app.post("/upload", response_model=UploadResponse)
async def upload_file(request: Request, file: UploadFile = File(...)):
    try:
        content = await file.read()
        filename = file.filename.lower()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
        else:
            raise HTTPException(status_code=400, detail="Only PDF files are supported")

        chunks = chunk_text(text)
        store = create_vector_store(chunks)

        app.state.vector_store = store
        app.state.chunks = chunks

        return UploadResponse(message="PDF processed", chunks=len(chunks))
    except ValueError as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Upload failed: %s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: Request, input: QuestionInput):
    try:
        store = app.state.vector_store
        chunks = app.state.chunks

        relevant_chunks = retrieve_relevant_chunks(input.question, store, chunks)
        answer = generate_answer(input.question, relevant_chunks)

        return AnswerResponse(question=input.question, answer=answer, history=[QA(question=input.question, answer=answer)])
    except Exception as e:
        logger.error(f"Answering error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.get("/health")
def health():
    return {"status": "healthy"}
