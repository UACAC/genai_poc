# main.py
from fastapi import FastAPI
from services.database import init_db
import uvicorn
from routers.chat import router as chat_router


app = FastAPI()

@app.on_event("startup")
def on_startup():
    init_db()

# Include the router that has /chat, /chat-rag, etc.
app.include_router(chat_router)

@app.get("/")
async def root():
    return {"message": "Welcome to the SEI Agent LLM Chatbot!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=9020, reload=False)
