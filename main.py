import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile

from config import settings
from models.schemas import LearnRequest, LearnResponse, QueryRequest, QueryResponse
from prompts.support import SYSTEM_PROMPTS, build_query_prompt
from services.learn import chunk_text, fetch_url, parse_html, parse_pdf
from services.llm import generate_response
from services.vectorstore import VectorStore

app = FastAPI(title="Sleeptery Support Assistant")
store = VectorStore(settings.CHROMA_PERSIST_DIR)


@app.post("/learn", response_model=LearnResponse)
async def learn_text(request: LearnRequest):
    if request.url:
        text = await fetch_url(request.url)
        source = request.url
    elif request.content:
        text = request.content
        source = "direct_input"
    else:
        raise HTTPException(status_code=400, detail="Provide content or url")

    chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": source} for _ in chunks]

    store.add(documents=chunks, metadatas=metadatas, ids=ids)

    return LearnResponse(status="ok", chunks_added=len(chunks), source=source)


@app.post("/learn/file", response_model=LearnResponse)
async def learn_file(
    file: UploadFile = File(...),
):
    content = await file.read()
    filename = file.filename or "unknown"

    if filename.endswith(".pdf"):
        text = parse_pdf(content)
    elif filename.endswith((".html", ".htm")):
        text = parse_html(content.decode())
    else:
        text = content.decode()

    chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="No content to ingest")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [{"source": filename} for _ in chunks]

    store.add(documents=chunks, metadatas=metadatas, ids=ids)

    return LearnResponse(status="ok", chunks_added=len(chunks), source=filename)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    model = request.model or settings.DEFAULT_MODEL

    results = store.query(request.question, n_results=settings.TOP_K)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    context = (
        "\n\n---\n\n".join(documents)
        if documents
        else "Релевантная информация не найдена в базе знаний."
    )

    system_prompt = SYSTEM_PROMPTS[request.mode.value]
    length = request.length.value if request.length else None
    user_message = build_query_prompt(context, request.question, request.hint, length)
    answer = await generate_response(model, system_prompt, user_message)

    sources = [
        {"text": doc[:200], "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]

    return QueryResponse(answer=answer, sources=sources, model=model)


@app.get("/stats")
async def stats():
    return {"total_chunks": store.count()}
