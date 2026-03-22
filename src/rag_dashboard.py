"""
Script to define the API to communicate with the frontend.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# ── FastAPI ────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pinecone_idx
    pinecone_idx = get_or_create_index()
    print(f"✅  Connected to Pinecone index '{PINECONE_INDEX_NAME}'")
    print(f"📂  Documents folder: {DOCUMENTS_FOLDER.absolute()}")
    print(f"🌐  Dashboard: http://localhost:8000")
    yield


app = FastAPI(lifespan=lifespan, title="RAG Dashboard")
app.mount("/images", StaticFiles(directory=str(IMAGES_FOLDER)), name="images")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Models ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5


class IndexRequest(BaseModel):
    force: bool = False


# ── API Routes ────────────────────────────────────────────────────────────────
@app.get("/api/status")
async def status():
    reg = load_registry()
    docs = [
        p
        for p in DOCUMENTS_FOLDER.iterdir()
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    try:
        stats = pinecone_idx.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
    except:
        total_vectors = "?"
    return {
        "documents_folder": str(DOCUMENTS_FOLDER.absolute()),
        "total_files": len(docs),
        "indexed_files": len(reg),
        "total_vectors": total_vectors,
        "index_name": PINECONE_INDEX_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
    }


@app.get("/api/documents")
async def list_documents():
    reg = load_registry()
    docs = []
    for p in DOCUMENTS_FOLDER.iterdir():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            h = file_hash(p)
            docs.append(
                {
                    "name": p.name,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "indexed": reg.get(p.name) == h,
                    "ext": p.suffix.lower()[1:],
                }
            )
    return sorted(docs, key=lambda x: x["name"])


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}")
    dest = DOCUMENTS_FOLDER / file.filename
    content = await file.read()
    dest.write_bytes(content)
    return {"status": "uploaded", "file": file.filename}


@app.post("/api/index")
async def index_documents(req: IndexRequest, background_tasks: BackgroundTasks):
    if indexing_status.get("running"):
        return {"status": "already_running"}
    background_tasks.add_task(index_all_documents, req.force)
    return {"status": "indexing_started"}


@app.get("/api/index/status")
async def get_index_status():
    return indexing_status


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if not req.question.strip():
        raise HTTPException(400, "Question cannot be empty")
    chunks = retrieve(req.question, req.top_k)
    if not chunks:
        return {
            "answer": "No relevant information found in the knowledge base. Please index your documents first.",
            "sources": [],
        }
    answer = generate_answer(req.question, chunks)
    # Deduplicate sources by image_file (keep highest score per page)
    seen = {}
    for c in chunks:
        key = c["image_file"]
        if key not in seen or c["score"] > seen[key]["score"]:
            seen[key] = c
    unique_sources = sorted(seen.values(), key=lambda x: -x["score"])
    return {
        "answer": answer,
        "sources": [
            {
                "source": c["source"],
                "page": c["page"],
                "total_pages": c["total_pages"],
                "image_file": c["image_file"],
                "doc_type": c["doc_type"],
                "score": c["score"],
                "text_preview": (
                    c["text"][:250] + "…" if len(c["text"]) > 250 else c["text"]
                ),
            }
            for c in unique_sources
        ],
    }
