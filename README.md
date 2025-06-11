# TCS_Apple_Training_Task_1 & Task_2

## Task 1  
1. Build a FastAPI application to query general knowledge questions to LLM. LLM must be running locally.  
2. Log each API call with a unique identifier in the console log along with the user query and LLM response. This identifier should be tracked at the API Level.

### Model used: TinyLlama  
### This model is running locally using Ollama.

**Part 1**: Build a FastAPI application to query general knowledge questions to LLM. LLM must be running locally.  
**Part 2**: Log each API call with a unique identifier in the console log along with the user query and LLM response. This identifier should be tracked at the API Level.

### Steps to run on the local system
1. Clone the repo.
2. Download and install Ollama on the local system.
3. Set up TinyLlama using Ollama:  
   ```bash
   ollama pull tinyllama
   ```
4.Then run TinyLlama using:
   ```bash
   ollama run tinyllama
   ```
5.Then run code using:
   ```bash
   uvicorn main:app --reload
   ```
6. Open the link in a browser

### Steps to use
1. Enter prompt
2. Click the submit button



.


## Task 2
1. Build RAG for data ingestion, use any vector DB. 
2. ⁠Build Fast API to ingest any file. for example your resume using RAG.
3. ⁠Ask sample questions against your resume.
4. ⁠figure it out different RAG strategy and build sample code for data ingestion.
5. ⁠update the readme file.

   
### Vector DB used: ChromaDB

### PDF Ingestion
1. Extracts text using PyMuPDF
2. Splits into ~500 character chunks
3. Embeds each chunk using all-MiniLM-L6-v2
4. Stores embeddings + metadata in ChromaDB


### RAG Logic
1. Encodes question
2. Queries top-n most relevant chunks
3. Assembles prompt with context
4. Sends to tinyllama via Ollama

### Different RAG Strategies Tried
1. Zero-shot RAG – Basic retrieval and answer without any fine-tuning or extra guidance.
2. RAG with Re-ranking – Used a secondary model(Re-rank using CrossEncoder) to re-rank the top-k retrieved chunks before sending to the LLM.
3. Query-focused RAG (with Query Rewriting) – Modified or enriched the user query to improve retrieval relevance.
4. Chunk Summarization – Summarized larger content chunks before embedding to reduce noise and increase semantic density.
