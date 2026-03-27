import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from config import settings
from models.schemas import (
    AnswerLength,
    LearnRequest,
    LearnResponse,
    QueryMode,
    QueryRequest,
    QueryResponse,
)
from prompts.support import SYSTEM_PROMPTS, build_dialog_prompt, build_query_prompt
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


@app.post("/query/dialog", response_model=QueryResponse)
async def query_dialog(
    file: UploadFile = File(...),
    question: str | None = Form(None),
    hint: str | None = Form(None),
    model: str | None = Form(None),
    mode: QueryMode = Form(QueryMode.strict),
    length: AnswerLength | None = Form(None),
):
    print(f"Received dialog file: {file.filename}, question: {question}, hint: {hint}, model: {model}, mode: {mode}, length: {length}")
    content = await file.read()
    dialog = content.decode()

    if not dialog.strip():
        raise HTTPException(status_code=400, detail="Dialog file is empty")

    model_name = model or settings.DEFAULT_MODEL

    # LLM анализирует диалог и выделяет суть для RAG-поиска
    search_query = await generate_response(
        model_name,
        "Ты — помощник для анализа диалогов поддержки.",
        "Выдели суть проблемы клиента из этого диалога в 1-2 предложения. "
        "Только суть, без вступлений.\n\n" + dialog,
    )
    results = store.query(search_query, n_results=settings.TOP_K)

    documents = results["documents"][0] if results["documents"] else []
    metadatas = results["metadatas"][0] if results["metadatas"] else []

    context = (
        "\n\n---\n\n".join(documents)
        if documents
        else "Релевантная информация не найдена в базе знаний."
    )

    system_prompt = SYSTEM_PROMPTS[mode.value]
    length_val = length.value if length else None
    user_message = build_dialog_prompt(context, dialog, question, hint, length_val)
    answer = await generate_response(model_name, system_prompt, user_message)

    sources = [
        {"text": doc[:200], "metadata": meta}
        for doc, meta in zip(documents, metadatas)
    ]

    return QueryResponse(answer=answer, sources=sources, model=model_name)


@app.get("/stats")
async def stats():
    return {"total_chunks": store.count()}
