"""
Minimal backend for Render deployment - without heavy ML dependencies
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import uuid

app = FastAPI(title="NotebookPRO API", version="2.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Models
class SpaceCreate(BaseModel):
    name: str

class SpaceResponse(BaseModel):
    id: str
    name: str
    created_at: str
    file_count: int

class ChatRequest(BaseModel):
    query: str
    space_id: str
    chat_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    chat_id: str
    timestamp: str

# Endpoints
@app.get("/")
async def root():
    return {"status": "NotebookPRO API is running", "version": "2.0.0 (minimal)"}

@app.get("/api/spaces")
async def get_spaces():
    return []

@app.post("/api/spaces")
async def create_space(space: SpaceCreate):
    space_id = space.name.lower().replace(" ", "_")
    return SpaceResponse(
        id=space_id,
        name=space.name,
        created_at=datetime.now().isoformat(),
        file_count=0
    )

@app.post("/api/chat")
async def chat(request: ChatRequest):
    # Minimal response for testing
    return ChatResponse(
        response="Backend deployed successfully! Full ML features coming soon.",
        sources=[],
        chat_id=request.chat_id or str(uuid.uuid4()),
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
