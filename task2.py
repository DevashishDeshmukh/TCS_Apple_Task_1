# TASK-2 CODE

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
import logging
import ollama

import fitz 
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from typing import List

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize embedding model & vector DB
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_store",
    anonymized_telemetry=False
))
collection = chroma_client.get_or_create_collection(name="resume_rag")

@app.get("/")
def serve_frontend(request: Request):
    request_id = getattr(request.state, "request_id", "N/A")
    logging.info(f"[Request ID: {request_id}] GET / request received.")
    return FileResponse("static/index.html")

@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

class PromptRequest(BaseModel):
    prompt: str

def log_chat_interaction(request_id: str, prompt: str, model_response: str):
    logging.info(f"[Request ID: {request_id}] User Prompt: {prompt}")
    logging.info(f"[Request ID: {request_id}] Model Response: {model_response}")


# --------- TASK-2 RAG related functions ------------

# Overlapping Chunking
def chunk_text(text: str, chunk_size=500, overlap=100) -> List[str]:
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size-overlap)]


def embed_and_store(chunks, source_doc_id):
    embeddings = embedding_model.encode(chunks).tolist()
    ids = [f"{source_doc_id}_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": i, "doc_id": source_doc_id} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"Stored {len(chunks)} chunks in vector DB")

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are supported currently."}
    
    file_bytes = await file.read()
    
    
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    
    chunks = chunk_text(full_text)
    source_doc_id = str(uuid4())
    embed_and_store(chunks, source_doc_id)
    
    return {"message": f"File ingested with id {source_doc_id}", "chunks_stored": len(chunks)}




# # Zero-shot RAG

# @app.post("/ragChat/")
# async def query_resume(request: Request, body: PromptRequest):
#     question = body.prompt
#     request_id = getattr(request.state, "request_id", "N/A")

    
#     question_embedding = embedding_model.encode([question]).tolist()

    
#     results = collection.query(
#         query_embeddings=question_embedding,
#         n_results=3
#     )

#     retrieved_chunks = results['documents'][0] if results['documents'] else []

#     if not retrieved_chunks:
#         return {"response": "No relevant information found.", "request_id": request_id}


#     context_text = "\n---\n".join(retrieved_chunks)
#     prompt = f"Use the following information to answer the question:\n{context_text}\n\nQuestion: {question}\nAnswer:"

#     try:
#         chat_response = ollama.chat(model='tinyllama', messages=[{'role': 'user', 'content': prompt}])
#         answer = chat_response['message']['content']
#     except Exception as e:
#         answer = f"Error calling LLM: {str(e)}"

#     log_chat_interaction(request_id, body.prompt, answer)
    
#     return {"response": answer, "request_id": request_id}







# # RAG with Re-ranking


# from sentence_transformers import CrossEncoder
# cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# @app.post("/ragChat/")
# async def query_resume(request: Request, body: PromptRequest):
#     question = body.prompt
#     request_id = getattr(request.state, "request_id", "N/A")

    
#     question_embedding = embedding_model.encode([question]).tolist()

    
#     results = collection.query(
#         query_embeddings=question_embedding,
#         n_results=10
#     )

#     retrieved_chunks = results['documents'][0] if results['documents'] else []

#     if not retrieved_chunks:
#         return {"response": "No relevant information found.", "request_id": request_id}
    
#     # Re-rank using CrossEncoder
    
#     rerank_inputs = [[question, chunk] for chunk in retrieved_chunks]
#     scores = cross_encoder.predict(rerank_inputs)
#     scored_chunks = list(zip(scores, retrieved_chunks))
#     scored_chunks.sort(key=lambda pair: pair[0], reverse=True)
#     top_chunks = [chunk for score, chunk in scored_chunks[:3]]


#     context_text = "\n---\n".join(top_chunks)
#     prompt = f"Use the following information to answer the question:\n{context_text}\n\nQuestion: {question}\nAnswer:"

#     try:
#         chat_response = ollama.chat(model='tinyllama', messages=[{'role': 'user', 'content': prompt}])
#         answer = chat_response['message']['content']
#     except Exception as e:
#         answer = f"Error calling LLM: {str(e)}"

#     log_chat_interaction(request_id, body.prompt, answer)
    
#     return {"response": answer, "request_id": request_id}













# # Query-focused RAG (with Query Rewriting)

# @app.post("/ragChat/")
# async def query_resume(request: Request, body: PromptRequest):
#     question = body.prompt
#     request_id = getattr(request.state, "request_id", "N/A")
    
#     try:
#         rewrite_prompt = f"Rewrite the following resume-related question to be more specific and informative:\n\n'{question}'"
#         rewrite_response = ollama.chat(model='tinyllama', messages=[{'role': 'user', 'content': rewrite_prompt}])
#         rewritten_question = rewrite_response['message']['content'].strip()
#     except Exception as e:
#         rewritten_question = original_question  # Fallback to original
#         logging.warning(f"[Request ID: {request_id}] Query rewriting failed: {str(e)}")

    
#     question_embedding = embedding_model.encode([rewritten_question]).tolist()

    
#     results = collection.query(
#         query_embeddings=question_embedding,
#         n_results=3
#     )

#     retrieved_chunks = results['documents'][0] if results['documents'] else []

#     if not retrieved_chunks:
#         return {"response": "No relevant information found.", "request_id": request_id}


#     context_text = "\n---\n".join(retrieved_chunks)
#     prompt = f"Use the following information to answer the question:\n{context_text}\n\nQuestion: {question}\nAnswer:"

#     try:
#         chat_response = ollama.chat(model='tinyllama', messages=[{'role': 'user', 'content': prompt}])
#         answer = chat_response['message']['content']
#     except Exception as e:
#         answer = f"Error calling LLM: {str(e)}"

#     log_chat_interaction(request_id, body.prompt, answer)
    
#     return {"response": answer, "request_id": request_id}









# Chunk Summarization

@app.post("/ragChat/")
async def query_resume(request: Request, body: PromptRequest):
    question = body.prompt
    request_id = getattr(request.state, "request_id", "N/A")
    
    question_embedding = embedding_model.encode([question]).tolist()

    
    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_chunks = results['documents'][0] if results['documents'] else []

    if not retrieved_chunks:
        return {"response": "No relevant information found.", "request_id": request_id}


    context_text = "\n---\n".join(retrieved_chunks)
    
    # Summarize the retrieved context
    try:
        summary_prompt = f"Summarize the following resume snippets concisely:\n\n{context_text}\n\nSummary:"
        summary_response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": summary_prompt}])
        summary_text = summary_response['message']['content'].strip()
    except Exception as e:
        summary_text = context_text 
        logging.warning(f"[Request ID: {request_id}] Summarization failed: {str(e)}")

    prompt = f"Use the following information to answer the question:\n{summary_text}\n\nQuestion: {question}\nAnswer:"

    try:
        chat_response = ollama.chat(model='tinyllama', messages=[{'role': 'user', 'content': prompt}])
        answer = chat_response['message']['content']
    except Exception as e:
        answer = f"Error calling LLM: {str(e)}"

    log_chat_interaction(request_id, body.prompt, answer)
    
    # 7. Return answer + request_id
    return {"response": answer, "request_id": request_id}
