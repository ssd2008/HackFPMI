from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date as Date
from os import getenv
from pathlib import Path
from time import perf_counter

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .assistant import assist, watch_matches
from .visitor_api import VisitorAccess, frontend_origin as parse_frontend_origin, router as visitor_router
from .visitor_inbox import VisitorInbox
from .store import CACHE, DATA, ROOT, MailStore, load_jsonl
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
    enabled = getattr(app.state, "visitor_enabled", visitor_enabled(False))
    public = bool(frontend_origin or enabled)
    app.state.public_profile = public
    expected_sha256 = DEMO_CORPUS_SHA256 if public else None
    corpus = DATA / "documents.jsonl"
    if not public and getenv("MAILSEARCH_DOCUMENTS"):
        corpus = Path(getenv("MAILSEARCH_DOCUMENTS"))
    documents = load_jsonl(corpus, expected_sha256=expected_sha256)
    app.state.corpus_sha256 = expected_sha256

    from .retrieval import SearchEngine

    app.state.store = MailStore(documents)
    app.state.search_engine = SearchEngine(documents, CACHE)
    from .live import AttachmentSearch
    app.state.attachments = AttachmentSearch(app.state.store, getenv("MAILSEARCH_FILES") or corpus.parent,
                                            getenv("MAILSEARCH_TESSDATA") or ROOT / ".cache/tessdata")
    visitors = Visitors(app.state.search_engine.dense) if enabled else None
    app.state.visitors = visitors
    try:
        if visitors:
            visitors.start()
        yield
    finally:
        if visitors:
            visitors.close()
        app.state.attachments.close()
        app.state.visitors = None


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
        return JSONResponse({"detail": "Для подключения укажите адрес и пароль приложения Яндекс Почты."}, status_code=422)
    return await request_validation_exception_handler(request, exc)


@app.exception_handler(VisitorError)
async def visitor_error(request: Request, exc: VisitorError):
    return JSONResponse({"detail": str(exc)}, status_code=exc.status)


class MailQuery(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    mailbox: str = Field(min_length=1, max_length=128)
    date: Date
    query: str = Field(min_length=1, max_length=1000)


class RangedMailQuery(MailQuery):
    include_attachments: bool = False
    date_from: Date | None = None
    date_to: Date | None = None


class SearchRequest(RangedMailQuery):
    limit: int = Field(default=10, ge=1, le=100)


class ActionsRequest(BaseModel):
    mailbox: str = Field(min_length=1, max_length=128)
    date: Date
    date_from: Date | None = None
    date_to: Date | None = None
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
    if http.state.visitor is not None:
        private = http.state.visitors.inbox(http.state.visitor, mailbox)
        if private is not None:
            return private
    raise HTTPException(404, "Почтовый ящик не найден")


def iso_date_range(date_from: Date | None, date_to: Date | None) -> dict[str, str]:
    return {
        key: value.isoformat()
        for key, value in (("date_from", date_from), ("date_to", date_to))
        if value is not None
    }


def ranked(request: RangedMailQuery, limit: int, store: MailStore) -> list[tuple[str, float]]:
    at = request.date.isoformat()
    date_from, date_to = request.date_from, request.date_to
    if isinstance(store, VisitorInbox):
        hits = store.rank(request.query, request.date, limit, date_from=date_from, date_to=date_to)
    else:
        hits = app.state.search_engine.search(
            request.mailbox, request.query, request.date, limit, date_from=date_from, date_to=date_to
        )
    bounds = iso_date_range(date_from, date_to)
    return [(doc_id, float(score)) for doc_id, score in hits if store.visible(request.mailbox, at, doc_id, **bounds)]


def ranked_with_files(request, limit, store):
    hits = ranked(request, limit, store)
    if not request.include_attachments:
        return hits, {}, False
    engine = store.engine if isinstance(store, VisitorInbox) else app.state.search_engine
    if engine is None or request.mailbox not in engine.boxes:
        return hits, {}, False
    text, rows, restricted = engine.fields.restrict(request.query, engine.boxes[request.mailbox])
    allowed = {engine.ids[i] for i in rows} if restricted else None
    if not text.strip() or request.query.strip() in store.by_id:
        return hits, {}, False
    return app.state.attachments.search(
        request.mailbox, text, request.date, limit, hits, store=store, allowed=allowed,
        date_from=request.date_from, date_to=request.date_to,
    )


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-cache"})


@app.get("/api/mailboxes")
def mailboxes(http: Request):
    boxes = app.state.store.mailboxes
    default = max(boxes, key=lambda box: box["count"])
    if http.state.visitor is not None:
        boxes = boxes + http.state.visitors.boxes(http.state.visitor)
    return {"mailboxes": boxes, "default_mailbox": default["id"],
            "default_date": default["latest_date"] or Date.today().isoformat(),
            "corpus_sha256": app.state.corpus_sha256,
            "visitor_mail": http.state.visitors is not None, "attachment_search": True}


@app.get("/api/inbox")
def inbox(http: Request, mailbox: str, date: Date, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100),
          date_from: Date | None = None, date_to: Date | None = None):
    return checked_store(http, mailbox).inbox(
        mailbox, date.isoformat(), offset, limit, **iso_date_range(date_from, date_to)
    )


@app.get("/api/actions")
def actions(http: Request, mailbox: str, date: Date, offset: int = Query(0, ge=0), limit: int = Query(25, ge=1, le=100),
            date_from: Date | None = None, date_to: Date | None = None):
    return checked_store(http, mailbox).action_inbox(
        mailbox, date, offset, limit, **iso_date_range(date_from, date_to)
    )


@app.post("/api/actions")
def marked_actions(http: Request, request: ActionsRequest):
    store = checked_store(http, request.mailbox)
    return store.action_inbox(
        request.mailbox, request.date, request.offset, request.limit, request.marked,
        **iso_date_range(request.date_from, request.date_to),
    )


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


@app.get("/api/mail/{doc_id}")
def message(http: Request, doc_id: str, mailbox: str, date: Date, include_attachments: bool = False):
    store = checked_store(http, mailbox)
    if not store.visible(mailbox, date.isoformat(), doc_id):
        raise HTTPException(404, "Письмо не найдено")
    row = store.by_id[doc_id]
    files = {}
    if include_attachments:
        items, complete = app.state.attachments.read(row, store)
        files = {"attachments": [{"filename": i.get("filename", ""), "description": i.get("description", "")} for i in items],
                 "partial_attachment_search": not complete}
    return {**store.summaries[doc_id], **files, "body": row["doc"], "requires_action": doc_id in store.actions_by_id,
            **({"truncated": True} if row.get("truncated") else {})}


@app.post("/api/search")
def search(http: Request, request: SearchRequest):
    started = perf_counter()
    store = checked_store(http, request.mailbox)
    hits, _, partial = ranked_with_files(request, request.limit, store)
    items = [{**store.summaries[doc_id], "score": score} for doc_id, score in hits]
    return {"items": items, **({"partial_attachment_search": partial} if request.include_attachments else {}), "elapsed_ms": round((perf_counter() - started) * 1000, 2)}


@app.post("/api/answer")
def answer(http: Request, request: RangedMailQuery):
    from .generation import answer_question

    store = checked_store(http, request.mailbox)
    hits, evidence, _ = ranked_with_files(request, 5, store)
    documents = [store.by_id[doc_id] for doc_id, _ in hits]
    if evidence:
        from .attachments import searchable_documents
        documents = searchable_documents([{**d, "attachments": evidence.get(d["doc_id"], [])} for d in documents])
    result = answer_question(request.query, documents)
    return {**result, "sources": [store.summaries[doc_id] for doc_id in result["sources"]]}
