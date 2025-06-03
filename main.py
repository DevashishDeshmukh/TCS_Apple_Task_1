# TASK-1 CODE


from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from uuid import uuid4
import logging
import ollama

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

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

@app.post("/chat")
async def chat(request: Request, body: PromptRequest):
    try:
        request_id = getattr(request.state, "request_id", "N/A")

        chat_response = ollama.chat(model='tinyllama', messages=[{'role': 'user', 'content': body.prompt}])
        message = chat_response['message']['content']

        log_chat_interaction(request_id, body.prompt, message)

        return JSONResponse(
            status_code=200,
            content={"response": message, "request_id": request_id},
        )
    
    except Exception as e:
        request_id = getattr(request.state, "request_id", "N/A")
        logging.error(f"[Request ID: {request_id}] Exception occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"{str(e)} | Request ID: {request_id}")
    
    
    