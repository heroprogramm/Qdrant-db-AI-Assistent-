from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import logging
import pytesseract
from PIL import Image
import io
import os
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import numpy as np

# Load environment variables
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = "mistralai/mistral-7b-instruct"
session_store: Dict[str, Dict[str, Any]] = {}

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

def get_session_data(session_id: str) -> Dict[str, Any]:
    if session_id not in session_store:
        session_store[session_id] = {
            "vector_store": None,
            "chunks": [],
            "base_qa": None,
            "last_qa": None
        }
    return session_store[session_id]

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
            for img in page.get_images(full=True):
                xref = img[0]
                base_image = pdf.extract_image(xref)
                image_bytes = base_image["image"]
                try:
                    text += "\n" + pytesseract.image_to_string(Image.open(io.BytesIO(image_bytes)))
                except Exception as e:
                    logger.warning(f"OCR failed: {e}")
        return text
    except Exception as e:
        logger.error(f"PDF error: {e}")
        raise ValueError("Failed to extract text from PDF")

def extract_text_from_image(file: bytes) -> str:
    try:
        image = Image.open(io.BytesIO(file))
        return pytesseract.image_to_string(image)
    except Exception as e:
        logger.error(f"OCR image error: {e}")
        raise ValueError("Failed to extract image text")

def chunk_text(text: str, chunk_size: int = 300) -> List[str]:
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def create_vector_store(chunks: List[str]) -> Dict[str, Any]:
    try:
        vectors = embedder.encode(chunks).tolist()

        if qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.delete_collection(COLLECTION_NAME)

        qdrant_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=len(vectors[0]), distance=Distance.COSINE)
        )

        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"chunk": chunks[i]}
            )
            for i, vector in enumerate(vectors)
        ]
        qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)

        tfidf = TfidfVectorizer().fit(chunks)
        tfidf_matrix = tfidf.transform(chunks)

        return {"tfidf": tfidf, "tfidf_matrix": tfidf_matrix, "chunks": chunks}
    except Exception as e:
        logger.error(f"Qdrant vector store error: {e}")
        raise RuntimeError("Failed to create Qdrant vector store")

def retrieve_relevant_chunks(query: str, store: Dict[str, Any], chunks: List[str], top_k=3) -> List[str]:
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

def generate_answer(question: str, context_chunks: List[str], base_qa: Optional[QA], last_qa: Optional[QA]) -> str:
    context = "\n---\n".join(list(dict.fromkeys(context_chunks))[:3])
    history_note = ""

    if detect_user_correction(question):
        history_note = "[User corrected topic]\n"
    elif base_qa and last_qa:
        history_note = f"Previous Q: {last_qa.question}\nPrevious A: {last_qa.answer}\n"
    elif base_qa:
        history_note = f"Context from: {base_qa.question}\n"

    prompt = f"{history_note}Context:\n{context}\n\nQuestion: {question}\nAnswer:"

    try:
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            logger.error(f"OpenRouter API error: {response.text}")
            return "Sorry, LLM failed to generate a response."
    except Exception as e:
        logger.error(f"OpenRouter request error: {e}")
        return "Sorry, LLM request failed."

@app.post("/upload", response_model=UploadResponse)
async def upload_file(request: Request, file: UploadFile = File(...), session_id: str = Query(...)):
    session_data = get_session_data(session_id)
    filename = file.filename.lower()
    try:
        content = await file.read()
        if filename.endswith(".pdf"):
            text = extract_text_from_pdf(content)
            filetype = "PDF"
        elif filename.endswith((".jpg", ".jpeg", ".png")):
            text = extract_text_from_image(content)
            filetype = "Image"
        else:
            raise HTTPException(status_code=400, detail="Only PDF or image allowed")

        chunks = chunk_text(text)
        store = create_vector_store(chunks)

        session_data.update({
            "vector_store": store,
            "chunks": chunks,
            "base_qa": None,
            "last_qa": None
        })

        return UploadResponse(message=f"{filetype} processed", chunks=len(chunks))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: Request, input: QuestionInput, session_id: str = Query(...)):
    session_data = get_session_data(session_id)
    if session_data["vector_store"] is None:
        raise HTTPException(status_code=400, detail="Please upload a file first")

    try:
        chunks = retrieve_relevant_chunks(input.question, session_data["vector_store"], session_data["chunks"])

        if detect_user_correction(input.question):
            session_data["base_qa"] = None
            session_data["last_qa"] = None
        elif session_data["base_qa"] and session_data["last_qa"]:
            session_data["base_qa"] = session_data["last_qa"]

        answer = generate_answer(input.question, chunks, session_data["base_qa"], session_data["last_qa"])
        session_data["last_qa"] = QA(question=input.question, answer=answer)

        if not session_data["base_qa"]:
            session_data["base_qa"] = session_data["last_qa"]

        return AnswerResponse(question=input.question, answer=answer, history=[session_data["last_qa"]])
    except Exception as e:
        logger.error(f"Answering error: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

@app.post("/reset")
async def reset_session(session_id: str = Query(...)):
    session_store[session_id] = {
        "vector_store": None,
        "chunks": [],
        "base_qa": None,
        "last_qa": None
    }
    return {"message": "Session reset successfully"}

@app.get("/health")
def health():
    return {"status": "healthy"}
