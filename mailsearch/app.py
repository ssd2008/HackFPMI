from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as Date
from os import getenv
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .assistant import assist, watch_matches
from .store import CACHE, DATA, ROOT, MailStore, load_jsonl


frontend_origin = getenv("MAILSEARCH_FRONTEND_ORIGIN", "").strip()
DEMO_CORPUS_SHA256 = "06feb4791a52faeb777f1a8d4c22d99efd26a279943a19bdfbd0b14bc9c8daa5"


@asynccontextmanager
async def lifespan(app: FastAPI):
    expected_sha256 = DEMO_CORPUS_SHA256 if frontend_origin else None
    path = Path(getenv("MAILSEARCH_DOCUMENTS") or DATA / "documents.jsonl")
    documents = load_jsonl(path, expected_sha256=expected_sha256)

    app.state.store = MailStore(documents)
    from .ranking import build_engine_modes
    from .live import AttachmentSearch

    (app.state.ordinary_engine, app.state.paraphrase_engine,
     app.state.search_engine) = build_engine_modes(documents, app.state.store)
    app.state.attachments = AttachmentSearch(app.state.store, getenv("MAILSEARCH_FILES") or path.parent,
                                            getenv("MAILSEARCH_TESSDATA") or ROOT / ".cache/tessdata")
    try:
        yield
    finally:
        app.state.attachments.close()


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


if frontend_origin:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[frontend_origin],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
app.add_middleware(SearchTiming)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class MailQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    include_attachments: bool = False
    mailbox: str = Field(min_length=1, max_length=128)
    date: Date
    query: str = Field(min_length=1, max_length=1000)


class RankedQuery(MailQuery):
    paraphrase: bool | None = None


class SearchRequest(RankedQuery):
    limit: int = Field(default=10, ge=1, le=100)


class ActionsRequest(BaseModel):
    mailbox: str = Field(min_length=1, max_length=128)
    date: Date
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=25, ge=1, le=100)
    marked: list[str] = Field(default_factory=list)


class WatchRequest(MailQuery):
    after: Date
    seen: list[str] = Field(default_factory=list)


def checked_store(mailbox: str) -> MailStore:
    store: MailStore = app.state.store
    if mailbox not in store.boxes:
        raise HTTPException(404, "Почтовый ящик не найден")
    return store


def ranked(request: RankedQuery, limit: int):
    store = checked_store(request.mailbox)
    at = request.date.isoformat()
    engine = app.state.search_engine
    query = request.query
    if request.paraphrase is True:
        from .ranking import paraphrase_query

        engine = app.state.paraphrase_engine
        query = paraphrase_query(query, request.date)
    elif request.paraphrase is False:
        engine = app.state.ordinary_engine
    hits = engine.search(request.mailbox, query, request.date, limit)
    evidence, partial = {}, False
    if request.include_attachments:
        hits, evidence, partial = app.state.attachments.search(request.mailbox, request.query, request.date, limit, hits)
    return [(did, float(score)) for did, score in hits if store.visible(request.mailbox, at, did)], evidence, partial


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


@app.get("/api/actions")
def actions(mailbox: str, date: Date, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    return checked_store(mailbox).action_inbox(mailbox, date, offset, limit)


@app.post("/api/actions")
def marked_actions(request: ActionsRequest):
    store = checked_store(request.mailbox)
    return store.action_inbox(request.mailbox, request.date, request.offset, request.limit, request.marked)


@app.get("/api/reminder-candidates")
def reminder_candidates(mailbox: str, date: Date):
    return {"items": checked_store(mailbox).reminder_candidates(mailbox, date.isoformat())}


@app.post("/api/assistant")
def assistant(request: MailQuery):
    return assist(checked_store(request.mailbox), request.mailbox, request.date, request.query)


@app.post("/api/assistant/watch")
def assistant_watch(request: WatchRequest):
    return watch_matches(checked_store(request.mailbox), request.mailbox, request.date,
                         request.after, request.query, request.seen)


@app.get("/api/mail/{doc_id}")
def message(doc_id: str, mailbox: str, date: Date, include_attachments: bool = False):
    store = checked_store(mailbox)
    if not store.visible(mailbox, date.isoformat(), doc_id):
        raise HTTPException(404, "Письмо не найдено")

    row = store.by_id[doc_id]
    items = row.get("attachments", [])
    if include_attachments:
        items, _ = app.state.attachments.read(row)
    attachments = [{key: item[key] for key in ("filename", "description") if key in item}
                   for item in items]
    return {**store.summaries[doc_id], "body": row["doc"], "attachments": attachments,
            "requires_action": doc_id in store.actions_by_id}


@app.post("/api/search")
def search(request: SearchRequest):
    started = perf_counter()
    hits, _, partial = ranked(request, request.limit)
    items = [{**app.state.store.summaries[doc_id], "score": score} for doc_id, score in hits]
    return {"items": items, "partial_attachment_search": partial, "elapsed_ms": round((perf_counter() - started) * 1000, 2)}


@app.post("/api/answer")
def answer(request: RankedQuery):
    from .generation import answer_question

    store = checked_store(request.mailbox)
    hits, evidence, _ = ranked(request, 5)
    documents = [dict(store.by_id[did], attachments=evidence[did]) if did in evidence else store.by_id[did]
                 for did, _ in hits]
    from .attachments import searchable_documents

    result = answer_question(request.query, searchable_documents(documents) if request.include_attachments else documents,
                             generate=getenv("MAILSEARCH_GENERATIVE_ANSWERS") == "1")
    return {**result, "sources": [store.summaries[doc_id] for doc_id in result["sources"]]}
