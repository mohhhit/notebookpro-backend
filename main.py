"""
FastAPI Backend for NotebookPRO
Handles RAG, LLM, file processing, and chat management
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os
import shutil

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.document_processor import DocumentProcessor
from utils.vector_db import VectorDatabase
from utils.hybrid_retriever import HybridRetriever
from utils.llm_generator import LLMGenerator
from utils.config_manager import ConfigManager
from utils.spaces_manager import SpacesManager

# Initialize FastAPI
app = FastAPI(title="NotebookPRO API", version="2.0.0")

# CORS - Allow Flutter web to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter web URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config_manager = ConfigManager()
spaces_manager = SpacesManager()
vector_db = None
llm_generator = None
current_space = None

# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class ChatRequest(BaseModel):
    query: str
    space_id: str
    chat_id: Optional[str] = None
    workflow: str = "chat"

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    chat_id: str
    timestamp: str

class SpaceCreate(BaseModel):
    name: str

class SpaceResponse(BaseModel):
    id: str
    name: str
    created_at: str
    file_count: int

class ChatInfo(BaseModel):
    id: str
    title: str
    preview: str
    created_at: str
    updated_at: str
    message_count: int

class ConfigResponse(BaseModel):
    groq_api_key: Optional[str]
    gemini_api_key: Optional[str]

class ConfigUpdate(BaseModel):
    groq_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

# ==================== Helper Functions ====================

def get_data_dir():
    """Get data directory path"""
    return Path(__file__).parent.parent / "data"

def get_space_dir(space_id: str):
    """Get space-specific directory"""
    return get_data_dir() / "spaces" / space_id

def load_chats_for_space(space_id: str) -> List[Dict]:
    """Load all chats for a space"""
    chats_file = get_space_dir(space_id) / "chats.json"
    if chats_file.exists():
        with open(chats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_chats_for_space(space_id: str, chats: List[Dict]):
    """Save chats for a space"""
    chats_file = get_space_dir(space_id) / "chats.json"
    chats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chats_file, 'w', encoding='utf-8') as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def get_chat_title(messages: List[Dict]) -> str:
    """Generate chat title from first user message"""
    for msg in messages:
        if msg['role'] == 'user':
            content = msg['content'][:50]
            return content + "..." if len(msg['content']) > 50 else content
    return "New Chat"

def initialize_space(space_id: str):
    """Initialize vector DB and components for a space"""
    global vector_db, llm_generator
    
    # Get API keys
    groq_key = config_manager.get_api_key('groq')
    gemini_key = config_manager.get_api_key('gemini')
    
    if not groq_key and not gemini_key:
        raise HTTPException(status_code=400, detail="No API keys configured. Please add Groq or Gemini API key.")
    
    # Initialize vector database for this space
    space_dir = get_space_dir(space_id)
    vector_db = VectorDatabase(collection_name=f"space_{space_id}")
    
    # Initialize LLM generator - choose provider based on available keys
    if groq_key:
        llm_generator = LLMGenerator(provider="groq", api_key=groq_key)
    else:
        llm_generator = LLMGenerator(provider="gemini", api_key=gemini_key)

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check"""
    return {"status": "NotebookPRO API is running", "version": "2.0.0"}

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current API keys (masked)"""
    groq_key = config_manager.get_api_key('groq')
    gemini_key = config_manager.get_api_key('gemini')
    
    return ConfigResponse(
        groq_api_key="***" + groq_key[-4:] if groq_key else None,
        gemini_api_key="***" + gemini_key[-4:] if gemini_key else None
    )

@app.post("/api/config")
async def update_config(config_update: ConfigUpdate):
    """Update API keys"""
    if config_update.groq_api_key:
        config_manager.save_api_key('groq', config_update.groq_api_key)
    if config_update.gemini_api_key:
        config_manager.save_api_key('gemini', config_update.gemini_api_key)
    
    return {"status": "success", "message": "Configuration updated"}

@app.get("/api/spaces", response_model=List[SpaceResponse])
async def get_spaces():
    """Get all spaces"""
    spaces = spaces_manager.get_all_spaces()
    
    result = []
    for space in spaces:
        space_id = space['id']
        space_dir = get_space_dir(space_id)
        processed_file = space_dir / "processed_files.json"
        
        file_count = 0
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                file_count = len(json.load(f))
        
        result.append(SpaceResponse(
            id=space_id,
            name=space['name'],
            created_at=space['created_at'],
            file_count=file_count
        ))
    
    return result

@app.post("/api/spaces", response_model=SpaceResponse)
async def create_space(space_data: SpaceCreate):
    """Create a new space"""
    try:
        space = spaces_manager.create_space(space_data.name)
        
        return SpaceResponse(
            id=space['id'],
            name=space['name'],
            created_at=space['created_at'],
            file_count=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/spaces/{space_id}")
async def delete_space(space_id: str):
    """Delete a space"""
    spaces_manager.delete_space(space_id)
    
    # Delete space directory
    space_dir = get_space_dir(space_id)
    if space_dir.exists():
        shutil.rmtree(space_dir)
    
    return {"status": "success", "message": f"Space {space_id} deleted"}

@app.get("/api/spaces/{space_id}/chats", response_model=List[ChatInfo])
async def get_chats(space_id: str):
    """Get all chats for a space"""
    chats = load_chats_for_space(space_id)
    
    result = []
    for chat in chats:
        messages = chat.get('messages', [])
        result.append(ChatInfo(
            id=chat['id'],
            title=get_chat_title(messages),
            preview=messages[0]['content'][:100] if messages else "",
            created_at=chat.get('created_at', ''),
            updated_at=chat.get('updated_at', ''),
            message_count=len(messages)
        ))
    
    return result

@app.get("/api/spaces/{space_id}/chats/{chat_id}")
async def get_chat(space_id: str, chat_id: str):
    """Get specific chat by ID"""
    chats = load_chats_for_space(space_id)
    
    for chat in chats:
        if chat['id'] == chat_id:
            return chat
    
    raise HTTPException(status_code=404, detail="Chat not found")

@app.delete("/api/spaces/{space_id}/chats/{chat_id}")
async def delete_chat(space_id: str, chat_id: str):
    """Delete a chat"""
    chats = load_chats_for_space(space_id)
    chats = [c for c in chats if c['id'] != chat_id]
    save_chats_for_space(space_id, chats)
    
    return {"status": "success", "message": f"Chat {chat_id} deleted"}

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message with RAG"""
    try:
        # Initialize space if needed
        initialize_space(request.space_id)
        
        # Create hybrid retriever with 60% vector, 40% BM25
        hybrid_retriever = HybridRetriever(vector_db, alpha=0.6)
        
        # Retrieve relevant documents
        documents, metadatas, scores = hybrid_retriever.retrieve(
            query=request.query,
            n_results=5
        )
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        
        for idx, (doc, meta, score) in enumerate(zip(documents, metadatas, scores), 1):
            context_parts.append(f"Document {idx}:\n{doc}\n")
            sources.append({
                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                "metadata": meta,
                "score": float(score)
            })
        
        context = "\n".join(context_parts)
        
        # Generate response based on workflow
        if request.workflow == "summarize":
            prompt = f"""Provide a comprehensive summary of the following documents:

{context}

Summary:"""
        else:  # chat
            prompt = f"""You are a helpful AI assistant. Answer the user's question ONLY using information from the provided documents.

Documents:
{context}

User Question: {request.query}

Instructions:
- Answer ONLY based on the documents above
- If the answer is not in the documents, say "I don't have enough information in the uploaded documents to answer that."
- Be concise and accurate
- Cite which document number you're referencing

Answer:"""
        
        # Generate response
        response = llm_generator.generate(prompt, temperature=0.3)
        
        # Create or update chat
        chat_id = request.chat_id or str(uuid.uuid4())
        chats = load_chats_for_space(request.space_id)
        
        # Find existing chat or create new
        chat = None
        for c in chats:
            if c['id'] == chat_id:
                chat = c
                break
        
        if not chat:
            chat = {
                'id': chat_id,
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            chats.append(chat)
        
        # Add messages
        timestamp = datetime.now().isoformat()
        chat['messages'].extend([
            {'role': 'user', 'content': request.query, 'timestamp': timestamp},
            {'role': 'assistant', 'content': response, 'timestamp': timestamp}
        ])
        chat['updated_at'] = timestamp
        
        # Save chats
        save_chats_for_space(request.space_id, chats)
        
        return ChatResponse(
            response=response,
            sources=sources,
            chat_id=chat_id,
            timestamp=timestamp
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/spaces/{space_id}/upload")
async def upload_files(space_id: str, files: List[UploadFile] = File(...)):
    """Upload and process files for a space"""
    try:
        # Initialize space
        initialize_space(space_id)
        
        # Save uploaded files temporarily
        space_dir = get_space_dir(space_id)
        uploads_dir = space_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        processor = DocumentProcessor()
        all_chunks = []
        processed_files = []
        
        for file in files:
            # Save file
            file_path = uploads_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Process file and extract content
            try:
                file_data = processor.process_file(file_path)
                content = file_data['content']
                
                # Chunk the content
                chunks = processor.chunk_text(content, chunk_size=512, overlap=50, semantic=True)
                
                # Format chunks for vector database
                formatted_chunks = []
                for idx, chunk in enumerate(chunks):
                    formatted_chunks.append({
                        'content': chunk,
                        'metadata': {
                            'filename': file.filename,
                            'chunk_index': idx,
                            'total_chunks': len(chunks),
                            'source_type': file_data['format']
                        }
                    })
                
                all_chunks.extend(formatted_chunks)
                processed_files.append({
                    'filename': file.filename,
                    'chunks': len(chunks),
                    'processed_at': datetime.now().isoformat()
                })
            except Exception as e:
                # Log error but continue with other files
                print(f"Error processing {file.filename}: {str(e)}")
                continue
        
        # Add to vector database in batches to avoid size limits
        if all_chunks:
            # Extract texts, metadatas, and generate IDs
            texts = [chunk['content'] for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]
            ids = [f"{space_id}_{idx}_{uuid.uuid4().hex[:8]}" for idx in range(len(all_chunks))]
            
            # Process in batches of 5000 to avoid ChromaDB batch size limit
            batch_size = 5000
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                vector_db.add_documents(batch_texts, batch_metadatas, batch_ids)
                print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        # Save processed files info
        processed_file = space_dir / "processed_files.json"
        existing = []
        if processed_file.exists():
            with open(processed_file, 'r') as f:
                existing = json.load(f)
        
        existing.extend(processed_files)
        with open(processed_file, 'w') as f:
            json.dump(existing, f, indent=2)
        
        return {
            "status": "success",
            "files_processed": len(processed_files),
            "total_chunks": len(all_chunks)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/spaces/{space_id}/files")
async def get_files(space_id: str):
    """Get processed files for a space"""
    processed_file = get_space_dir(space_id) / "processed_files.json"
    
    if processed_file.exists():
        with open(processed_file, 'r') as f:
            return json.load(f)
    
    return []

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")
