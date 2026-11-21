from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as Date
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .store import CACHE, DATA, ROOT, MailStore, load_jsonl


@asynccontextmanager
async def lifespan(app: FastAPI):
    from .retrieval import SearchEngine

    documents = load_jsonl(DATA / "documents.jsonl")
    app.state.store = MailStore(documents)
    app.state.search_engine = SearchEngine(documents, CACHE)
    yield


app = FastAPI(title="Умный поиск в почте", lifespan=lifespan, docs_url=None, redoc_url=None)


class SearchTiming:
    """Include request parsing, worker scheduling, retrieval and response serialization."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["path"] != "/api/search":
            return await self.app(scope, receive, send)
        started = perf_counter()

        async def timed_send(message):
            if message["type"] == "http.response.start":
                ms = (perf_counter() - started) * 1000
                message["headers"] = [*message["headers"], (b"server-timing", f"search;dur={ms:.3f}".encode())]
            await send(message)

        await self.app(scope, receive, timed_send)


app.add_middleware(SearchTiming)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class MailQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    mailbox: str = Field(min_length=1, max_length=128)
    date: Date
    query: str = Field(min_length=1, max_length=1000)


class SearchRequest(MailQuery):
    limit: int = Field(default=10, ge=1, le=100)


def checked_store(mailbox: str) -> MailStore:
    store: MailStore = app.state.store
    if mailbox not in store.boxes:
        raise HTTPException(404, "Почтовый ящик не найден")
    return store


def ranked(request: MailQuery, limit: int) -> list[tuple[str, float]]:
    store = checked_store(request.mailbox)
    at = request.date.isoformat()
    hits = app.state.search_engine.search(request.mailbox, request.query, request.date, limit)
    return [(doc_id, float(score)) for doc_id, score in hits if store.visible(request.mailbox, at, doc_id)]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/mailboxes")
def mailboxes():
    boxes = app.state.store.mailboxes
    default = max(boxes, key=lambda box: box["count"])
    return {"mailboxes": boxes, "default_mailbox": default["id"], "default_date": default["latest_date"]}


@app.get("/api/inbox")
def inbox(mailbox: str, date: Date, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    return checked_store(mailbox).inbox(mailbox, date.isoformat(), offset, limit)


@app.get("/api/mail/{doc_id}")
def message(doc_id: str, mailbox: str, date: Date):
    store = checked_store(mailbox)
    if not store.visible(mailbox, date.isoformat(), doc_id):
        raise HTTPException(404, "Письмо не найдено")
    return {**store.summaries[doc_id], "body": store.by_id[doc_id]["doc"]}


@app.post("/api/search")
def search(request: SearchRequest):
    started = perf_counter()
    hits = ranked(request, request.limit)
    items = [{**app.state.store.summaries[doc_id], "score": score} for doc_id, score in hits]
    return {"items": items, "elapsed_ms": round((perf_counter() - started) * 1000, 2)}


@app.post("/api/answer")
def answer(request: MailQuery):
    from .generation import answer_question

    store = checked_store(request.mailbox)
    documents = [store.by_id[doc_id] for doc_id, _ in ranked(request, 5)]
    result = answer_question(request.query, documents)
    return {**result, "sources": [store.summaries[doc_id] for doc_id in result["sources"]]}
