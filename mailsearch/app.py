from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as Date
from os import getenv
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .assistant import assist, watch_matches
from .local_api import LocalAccess, router as local_router
from .local_mail import LocalInbox, LocalMail
from .store import CACHE, DATA, ROOT, MailStore, load_corpus


frontend_origin = getenv("MAILSEARCH_FRONTEND_ORIGIN", "").strip()
DEMO_CORPUS_SHA256 = "06feb4791a52faeb777f1a8d4c22d99efd26a279943a19bdfbd0b14bc9c8daa5"


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = getattr(app.state, "data_dir", DATA)
    documents, app.state.corpus_sha256 = load_corpus(data / "documents.jsonl")
    if frontend_origin and app.state.corpus_sha256 != DEMO_CORPUS_SHA256:
        raise ValueError("Для публичного демо нужен исходный синтетический корпус VK.")

    from .retrieval import SearchEngine

    app.state.store = MailStore(documents)
    app.state.search_engine = SearchEngine(documents, CACHE)
    local_dir = getattr(app.state, "local_dir", None)
    if local_dir is not None and frontend_origin:
        raise ValueError("Личные ящики доступны только в локальном приложении.")
    local = LocalMail(local_dir) if local_dir is not None else None
    app.state.local_mail = local
    try:
        if local:
            local.start()
        yield
    finally:
        if local:
            local.close()
        app.state.local_mail = None


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
app.add_middleware(LocalAccess, state=app.state)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.include_router(local_router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api/local/"):
        # Never echo a rejected password or letter in a validation response.
        message = ("Проверьте провайдера, адрес и пароль приложения."
                   if request.url.path == "/api/local/accounts" else "Проверьте адреса получателей, тему и текст письма.")
        return JSONResponse({"detail": message}, status_code=422)
    return await request_validation_exception_handler(request, exc)


class MailQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    mailbox: str = Field(min_length=1, max_length=128)
    date: Date
    query: str = Field(min_length=1, max_length=1000)


class SearchRequest(MailQuery):
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
    if mailbox in store.boxes:
        return store
    local = getattr(app.state, "local_mail", None)
    private_store = local.inbox(mailbox) if local else None
    if private_store is not None:
        return private_store
    raise HTTPException(404, "Почтовый ящик не найден")


def ranked(request: MailQuery, limit: int, store: MailStore | None = None) -> list[tuple[str, float]]:
    if store is None:
        store = checked_store(request.mailbox)
    at = request.date.isoformat()
    if isinstance(store, LocalInbox):
        hits = store.rank(request.query, request.date, limit)
    else:
        hits = app.state.search_engine.search(request.mailbox, request.query, request.date, limit)
    return [(doc_id, float(score)) for doc_id, score in hits if store.visible(request.mailbox, at, doc_id)]


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/mailboxes")
def mailboxes():
    boxes = app.state.store.mailboxes
    default = max(boxes, key=lambda box: box["count"])
    local = getattr(app.state, "local_mail", None)
    if local:
        boxes = boxes + local.boxes()
    return {"mailboxes": boxes, "default_mailbox": default["id"],
            "default_date": default["latest_date"] or Date.today().isoformat(),
            "corpus_sha256": app.state.corpus_sha256, "local_mail": local is not None}


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


@app.get("/api/mail/{doc_id:path}")
def message(doc_id: str, mailbox: str, date: Date):
    store = checked_store(mailbox)
    if not store.visible(mailbox, date.isoformat(), doc_id):
        raise HTTPException(404, "Письмо не найдено")
    return {**store.summaries[doc_id], "body": store.by_id[doc_id]["doc"],
            "requires_action": doc_id in store.actions_by_id}


@app.post("/api/search")
def search(request: SearchRequest):
    started = perf_counter()
    store = checked_store(request.mailbox)
    hits = ranked(request, request.limit, store)
    items = [{**store.summaries[doc_id], "score": score} for doc_id, score in hits]
    return JSONResponse({"items": items, "elapsed_ms": round((perf_counter() - started) * 1000, 2)})


@app.post("/api/answer")
def answer(request: MailQuery):
    from .generation import answer_question

    store = checked_store(request.mailbox)
    documents = [store.by_id[doc_id] for doc_id, _ in ranked(request, 5, store)]
    result = answer_question(request.query, documents)
    return {**result, "sources": [store.summaries[doc_id] for doc_id in result["sources"]]}
