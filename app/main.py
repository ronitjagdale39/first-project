from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .router.chat import router as chat_router
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="ML Chatbot Service", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix="/chat")

@app.get("/")
def root():
    return {"status": "ok", "service": "ml-chatbot"}
