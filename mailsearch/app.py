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
from .local_api import VisitorAccess, frontend_origin as parse_frontend_origin, router as visitor_router
from .local_mail import LocalArchive, LocalInbox
from .store import CACHE, DATA, ROOT, MailStore, load_corpus, load_jsonl
from .visitor_mail import VisitorError, Visitors


frontend_origin = parse_frontend_origin(getenv("MAILSEARCH_FRONTEND_ORIGIN", ""))
DEMO_CORPUS_SHA256 = "06feb4791a52faeb777f1a8d4c22d99efd26a279943a19bdfbd0b14bc9c8daa5"


def visitor_enabled(local_default):
    value = getenv("MAILSEARCH_VISITOR_MAIL")
    if value is not None:
        return value.strip().casefold() in {"1", "true", "yes", "on"}
    return bool(frontend_origin or local_default)


@asynccontextmanager
async def lifespan(app: FastAPI):
    data = getattr(app.state, "data_dir", DATA)
    enabled = getattr(app.state, "visitor_enabled", visitor_enabled(False))
    public = bool(frontend_origin or (enabled and getattr(app.state, "local_dir", None) is None))
    app.state.public_profile = public
    if public:
        # Authenticate the file before parsing any message, even when the supplied --data is wrong.
        documents = load_jsonl(data / "documents.jsonl", expected_sha256=DEMO_CORPUS_SHA256)
        app.state.corpus_sha256 = DEMO_CORPUS_SHA256
    else:
        documents, app.state.corpus_sha256 = load_corpus(data / "documents.jsonl")

    from .retrieval import SearchEngine

    app.state.store = MailStore(documents)
    app.state.search_engine = SearchEngine(documents, CACHE)
    local_dir = getattr(app.state, "local_dir", None)
    app.state.local_mail = LocalArchive(local_dir) if local_dir is not None and not public else None
    visitors = Visitors(app.state.search_engine.dense) if enabled else None
    app.state.visitors = visitors
    try:
        if visitors:
            visitors.start()
        yield
    finally:
        if visitors:
            visitors.close()
        app.state.visitors = None
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


class MailCORS(CORSMiddleware):
    def preflight_response(self, request_headers):
        response = super().preflight_response(request_headers)
        response.headers["Cache-Control"] = "private, no-store"
        return response


def configure_access(application, origin):
    application.add_middleware(SearchTiming)
    application.add_middleware(VisitorAccess, state=application.state, origin=origin)
    if origin:
        application.add_middleware(
            MailCORS, allow_origins=[origin], allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "Authorization", "X-Mailsearch-Visitor"], allow_credentials=False,
        )


configure_access(app, frontend_origin)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.include_router(visitor_router)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/api/visitor/"):
        return JSONResponse({"detail": "Проверьте провайдера, адрес и пароль приложения."}, status_code=422)
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(VisitorError)
async def visitor_error(request: Request, exc: VisitorError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status)


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


def checked_store(http: Request, mailbox: str) -> MailStore:
    store: MailStore = app.state.store
    if mailbox in store.boxes:
        return store
    local = getattr(app.state, "local_mail", None) if http.state.archive_allowed else None
    if local and (archive := local.inbox(mailbox)) is not None:
        return archive
    if http.state.visitor is not None:
        private = http.state.visitors.inbox(http.state.visitor, mailbox)
        if private is not None:
            return private
    raise HTTPException(404, "Почтовый ящик не найден")


def ranked(request: MailQuery, limit: int, store: MailStore) -> list[tuple[str, float]]:
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
def mailboxes(http: Request):
    boxes = app.state.store.mailboxes
    default = max(boxes, key=lambda box: box["count"])
    local = getattr(app.state, "local_mail", None) if http.state.archive_allowed else None
    if local:
        boxes = boxes + local.boxes()
    if http.state.visitor is not None:
        boxes = boxes + http.state.visitors.boxes(http.state.visitor)
    return {"mailboxes": boxes, "default_mailbox": default["id"],
            "default_date": default["latest_date"] or Date.today().isoformat(),
            "corpus_sha256": app.state.corpus_sha256, "local_mail": local is not None,
            "visitor_mail": http.state.visitors is not None}


@app.get("/api/inbox")
def inbox(http: Request, mailbox: str, date: Date, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    return checked_store(http, mailbox).inbox(mailbox, date.isoformat(), offset, limit)


@app.get("/api/actions")
def actions(http: Request, mailbox: str, date: Date, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100)):
    return checked_store(http, mailbox).action_inbox(mailbox, date, offset, limit)


@app.post("/api/actions")
def marked_actions(http: Request, request: ActionsRequest):
    store = checked_store(http, request.mailbox)
    return store.action_inbox(request.mailbox, request.date, request.offset, request.limit, request.marked)


@app.get("/api/reminder-candidates")
def reminder_candidates(http: Request, mailbox: str, date: Date):
    return {"items": checked_store(http, mailbox).reminder_candidates(mailbox, date.isoformat())}


@app.post("/api/assistant")
def assistant(http: Request, request: MailQuery):
    return assist(checked_store(http, request.mailbox), request.mailbox, request.date, request.query)


@app.post("/api/assistant/watch")
def assistant_watch(http: Request, request: WatchRequest):
    return watch_matches(checked_store(http, request.mailbox), request.mailbox, request.date,
                         request.after, request.query, request.seen)


@app.get("/api/mail/{doc_id:path}")
def message(http: Request, doc_id: str, mailbox: str, date: Date):
    store = checked_store(http, mailbox)
    if not store.visible(mailbox, date.isoformat(), doc_id):
        raise HTTPException(404, "Письмо не найдено")
    row = store.by_id[doc_id]
    return {**store.summaries[doc_id], "body": row["doc"], "requires_action": doc_id in store.actions_by_id,
            **({"truncated": True} if row.get("truncated") else {})}


@app.post("/api/search")
def search(http: Request, request: SearchRequest):
    started = perf_counter()
    store = checked_store(http, request.mailbox)
    hits = ranked(request, request.limit, store)
    items = [{**store.summaries[doc_id], "score": score} for doc_id, score in hits]
    return JSONResponse({"items": items, "elapsed_ms": round((perf_counter() - started) * 1000, 2)})


@app.post("/api/answer")
def answer(http: Request, request: MailQuery):
    from .generation import answer_question

    store = checked_store(http, request.mailbox)
    documents = [store.by_id[doc_id] for doc_id, _ in ranked(request, 5, store)]
    result = answer_question(request.query, documents)
    return {**result, "sources": [store.summaries[doc_id] for doc_id in result["sources"]]}
