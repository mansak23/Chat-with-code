from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os
import chromadb
import shutil # For clearing the database directory
from typing import List, Optional # Ensure Optional is imported
from fastapi.middleware.cors import CORSMiddleware

from backend import rag_module 
from backend import llm_module

# Define the path to your ChromaDB data
CHROMA_DB_PATH = "data/chroma_db"
COLLECTION_NAME = "code_chunks" # Must match COLLECTION_NAME in rag_module.py

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize ChromaDB client and collection
    print("Initializing ChromaDB client...")
    os.makedirs(CHROMA_DB_PATH, exist_ok=True)
    
    rag_module.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    rag_module.collection = rag_module.client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=rag_module.EMBEDDING_FUNCTION_CHROMA # Pass the embedding function
    )
    
    print("Application startup complete. ChromaDB ready for operations.")
    yield
    # Shutdown: No specific cleanup needed for persistent ChromaDB

app = FastAPI(lifespan=lifespan)

# Add CORS middleware to allow OPTIONS requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    temperature: float = llm_module.DEFAULT_TEMPERATURE
    top_k: int = rag_module.DEFAULT_TOP_K
    similarity_threshold: float = rag_module.DEFAULT_SIMILARITY_THRESHOLD
    filter_type: Optional[str] = None # Added: Optional filter for chunk type

class CodeChunk(BaseModel):
    content: str
    source: str
    start_line: int
    type: str
    distance: Optional[float] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None  # Added: Reflects new metadata from rag_module
    struct_name: Optional[str] = None # Added: Reflects new metadata from rag_module

class QueryResponse(BaseModel):
    answer: str
    retrieved_context: List[CodeChunk]
    debug_info: dict = {}

@app.get("/")
async def read_root():
    return {"message": "Chat with Your Code API is running!"}

@app.post("/upload_code_file")
async def upload_code_file(file: UploadFile = File(...)):
    """
    Receives an uploaded code file, processes it, and stores its chunks in ChromaDB.
    """
    if not file.filename.endswith(('.c', '.cpp', '.h')):
        raise HTTPException(status_code=400, detail="Only .c, .cpp, and .h files are allowed.")
    
    try:
        file_content_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(file_content_bytes)} bytes")
        
        # Process and store the uploaded file
        rag_module.process_and_store_uploaded_file(file_content_bytes, file.filename)
        
        return {"message": f"File '{file.filename}' processed and indexed successfully."}
    except Exception as e:
        print(f"Error processing uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {e}")

@app.post("/clear_codebase")
async def clear_codebase():
    """
    Clears all data from the ChromaDB collection.
    """
    try:
        if os.path.exists(CHROMA_DB_PATH):
            shutil.rmtree(CHROMA_DB_PATH)
            print(f"Cleared ChromaDB at {CHROMA_DB_PATH}")
            
            # Re-initialize the collection after clearing
            # Need to ensure the directory exists before re-initializing
            os.makedirs(CHROMA_DB_PATH, exist_ok=True) 
            rag_module.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
            rag_module.collection = rag_module.client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=rag_module.EMBEDDING_FUNCTION_CHROMA
            )
            print("ChromaDB re-initialized after clearing.")
        else:
            print("ChromaDB directory does not exist, no clearing needed.")
        
        return {"message": "ChromaDB codebase cleared successfully."}
    except Exception as e:
        print(f"Error clearing ChromaDB: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear codebase: {e}")


@app.post("/ask/", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    try:
        # Retrieve relevant code chunks, passing filter_type
        retrieved_chunks = rag_module.retrieve_relevant_chunks(
            request.query, 
            top_k=request.top_k, 
            similarity_threshold=request.similarity_threshold,
            filter_type=request.filter_type # Pass the new filter_type
        )
        
        # Prepare list of chunk contents for the LLM
        chunks_for_llm_context = [chunk['content'] for chunk in retrieved_chunks]

        # ADD THIS BLOCK FOR DEBUGGING
        if chunks_for_llm_context:
            print("\nDEBUG: Chunks content being sent to LLM:")
            for i, content in enumerate(chunks_for_llm_context):
                print(f"  Chunk {i+1} content (first 200 chars):\n{content[:200]}...")
            print("--- End of Chunks for LLM ---\n")
        else:
            print("\nDEBUG: No chunks found to send to LLM.")
        # END DEBUG BLOCK

        if not chunks_for_llm_context: # Check if the list is empty
            answer = "I cannot answer this question based on the provided code context. No relevant code chunks were found."
        else:
            # Generate answer using the LLM
            answer = llm_module.generate_answer(
                request.query, 
                chunks_for_llm_context, # Pass the list of content strings
                temperature=request.temperature
            )

        # Format retrieved_context for the API response model
        # The CodeChunk model expects specific Optional fields to be handled
        response_chunks = []
        for chunk in retrieved_chunks:
            response_chunks.append(CodeChunk(
                content=chunk['content'],
                source=chunk['source'],
                start_line=chunk['start_line'],
                type=chunk['type'],
                distance=chunk.get('distance'), # Optional
                function_name=chunk.get('function_name') or None, # Ensure None if empty string for Pydantic Optional
                class_name=chunk.get('class_name') or None,     # Ensure None if empty string for Pydantic Optional
                struct_name=chunk.get('struct_name') or None    # Ensure None if empty string for Pydantic Optional
            ))
        
        # Optionally, enhance debug_info with retrieved chunk types/names
        debug_chunks_summary = []
        for chunk in retrieved_chunks:
            name = chunk.get('function_name') or chunk.get('class_name') or chunk.get('struct_name')
            debug_chunks_summary.append({
                "type": chunk['type'],
                "name": name if name else "N/A",
                "source": os.path.basename(chunk['source']),
                "line": chunk['start_line'],
                "distance": f"{chunk['distance']:.4f}"
            })

        return QueryResponse(
            answer=answer,
            retrieved_context=response_chunks,
            debug_info={
                "retrieved_chunk_count": len(retrieved_chunks),
                "query_top_k": request.top_k,
                "query_similarity_threshold": request.similarity_threshold,
                "llm_temperature": request.temperature,
                "filter_type_applied": request.filter_type, # Added debug info
                "retrieved_chunks_summary": debug_chunks_summary # Added detailed summary
            }
        )

    except Exception as e:
        print(f"An error occurred in /ask/: {e}")
        # Log traceback for better debugging in production
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)