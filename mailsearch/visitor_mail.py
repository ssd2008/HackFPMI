"""Process-local ownership and read-only synchronization for visitor mailboxes."""

from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from hashlib import sha256
import logging
from secrets import token_urlsafe
from sys import getsizeof
from threading import BoundedSemaphore, Event, Lock, RLock, Thread
from time import monotonic

from . import mail_transport as transport
from .visitor_inbox import VisitorInbox

_LOGGER = logging.getLogger(__name__)

SYNC_SECONDS = 300
SESSION_SECONDS = 3600
MAILBOX_LIMIT = 500
PROCESS_ROWS = 8_000
PROCESS_TEXT_BYTES = 48 * 1024 * 1024
RELOGIN = "Сессия закончилась. Откройте подключение почты заново."


class VisitorError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status = status


@dataclass(eq=False)
class _Account:
    provider: str
    address: str
    password: str = field(repr=False)
    folder: str = "inbox"
    id: str = field(default_factory=lambda: "account-" + token_urlsafe(18))
    validity: str | None = None
    rows: dict = field(default_factory=dict, repr=False)
    view: VisitorInbox | None = field(default=None, repr=False)
    index_lock: Lock = field(default_factory=Lock, repr=False)
    revision: int = 0
    last_synced: str | None = None
    latest: str | None = None
    error: str | None = None
    pending: int = 0
    total_available: int | None = None
    text_bytes: int = 0
    syncing: bool = False
    next_sync: float = 0

    def summary(self):
        return dict(id=self.id, provider=self.provider, address=self.address, folder=self.folder, last_synced=self.last_synced,
                    status="syncing" if self.syncing else "error" if self.error else "idle",
                    error=self.error, pending=self.pending, count=len(self.rows),
                    sync_limit=MAILBOX_LIMIT, total_available=self.total_available)


@dataclass(eq=False)
class Session:
    expires: float
    accounts: dict[str, _Account] = field(default_factory=dict, repr=False)
    failures: deque = field(default_factory=deque, repr=False)
    logging_in: set = field(default_factory=set, repr=False)
    revoked: bool = False


class Visitors:
    def __init__(self, encoder, *, remote=transport, clock=monotonic, ttl=SESSION_SECONDS, capacity=128):
        self.encoder = encoder
        self.remote = remote
        self.clock = clock
        self.ttl = ttl
        self.capacity = capacity
        self.sessions = {}
        self.issues = OrderedDict()
        self.row_count = 0
        self.text_bytes = 0
        self.lock = RLock()
        self.verifications = BoundedSemaphore(6)
        self.workers = ThreadPoolExecutor(max_workers=4, thread_name_prefix="visitor-imap")
        self.stop = Event()
        self.scheduler = Thread(target=self._schedule, name="visitor-cleanup", daemon=True)

    def start(self):
        self.scheduler.start()

    def _clear(self, session):
        session.revoked = True
        for account in session.accounts.values():
            self.row_count -= len(account.rows)
            self.text_bytes -= account.text_bytes
            account.password = ""
            account.rows.clear()
            account.view = None
        session.accounts.clear()
        session.failures.clear()
        session.logging_in.clear()

    def _prune(self, now):
        for key, session in list(self.sessions.items()):
            if session.expires <= now:
                self._clear(session)
                del self.sessions[key]
        while self.issues and next(iter(self.issues.values()))[-1] <= now - 60:
            self.issues.popitem(last=False)

    def create(self, peer):
        with self.lock:
            now = self.clock()
            self._prune(now)
            if self.stop.is_set() or len(self.sessions) >= self.capacity:
                raise VisitorError(429, "Сейчас открыто много подключений. Попробуйте позже.")
            recent = self.issues.get(peer, deque())
            while recent and recent[0] <= now - 60:
                recent.popleft()
            if len(recent) >= 30:
                raise VisitorError(429, "Слишком много новых сессий. Повторите через минуту.")
            recent.append(now)
            self.issues[peer] = recent
            self.issues.move_to_end(peer)
            if len(self.issues) > 1024:
                self.issues.popitem(last=False)
            token = token_urlsafe(32)
            self.sessions[sha256(token.encode()).digest()] = Session(now + self.ttl)
            return {"token": token, "expires_in": self.ttl}

    def resolve(self, token):
        with self.lock:
            if len(token) != 43 or not token.isascii():
                raise VisitorError(401, RELOGIN)
            key = sha256(token.encode()).digest()
            session = self.sessions.get(key)
            if session is None:
                raise VisitorError(401, RELOGIN)
            if session.expires <= self.clock():
                self._clear(session)
                del self.sessions[key]
                raise VisitorError(401, RELOGIN)
            return session

    def active(self, session):
        return not session.revoked and session.expires > self.clock() and not self.stop.is_set()

    def _require(self, session):
        if not self.active(session):
            raise VisitorError(401, RELOGIN)

    def logout(self, session):
        with self.lock:
            self._clear(session)
            for key, stored in list(self.sessions.items()):
                if stored is session:
                    del self.sessions[key]
                    break

    def close(self):
        with self.lock:
            self.stop.set()
            for session in self.sessions.values():
                self._clear(session)
            self.sessions.clear()
            self.issues.clear()
        if self.scheduler.is_alive():
            self.scheduler.join()
        self.workers.shutdown(wait=True, cancel_futures=True)

    def _schedule(self):
        while not self.stop.wait(min(15, self.ttl)):
            with self.lock:
                now = self.clock()
                self._prune(now)
                for session in self.sessions.values():
                    for account in session.accounts.values():
                        if account.next_sync <= now:
                            self._queue(session, account)

    def accounts(self, session):
        with self.lock:
            self._require(session)
            return [a.summary() for a in session.accounts.values()]

    def boxes(self, session):
        with self.lock:
            self._require(session)
            return [dict(id=a.id, label=a.address + (" (Отправленные)" if a.folder == "sent" else ""),
                         folder=a.folder, kind="live", count=len(a.rows), latest_date=a.latest,
                         examples=[], today=date.today().isoformat()) for a in session.accounts.values()]

    def connect(self, session, provider, address, password, folder="inbox"):
        key = (provider, address, folder)
        with self.lock:
            self._require(session)
            now = self.clock()
            while session.failures and session.failures[0] <= now - 60:
                session.failures.popleft()
            if len(session.failures) + len(session.logging_in) >= 6 or key in session.logging_in:
                raise VisitorError(429, "Подождите минуту перед следующей попыткой подключения.")
            existing = next((a for a in session.accounts.values() if (a.provider, a.address, a.folder) == key), None)
            if existing is None and len(session.accounts) + len(session.logging_in) >= 4:
                raise VisitorError(429, "В одной сессии можно подключить до четырёх ящиков.")
            if not self.verifications.acquire(blocking=False):
                raise VisitorError(429, "Одновременно проверяется много подключений. Попробуйте чуть позже.")
            session.logging_in.add(key)
        try:
            try:
                self.remote.verify_account(provider, address, password, folder)
            except transport.AuthenticationError as error:
                with self.lock:
                    if self.active(session):
                        session.failures.append(self.clock())
                _LOGGER.warning("IMAP provider=%s stage=authentication result=%s", provider, error.reason)
                raise VisitorError(403, str(error)) from None
            except transport.TransportError:
                # The public tunnel can replace a 502 with an HTML page without CORS headers.
                raise VisitorError(424, "Сервер демо не смог подключиться к выбранной папке по IMAP.") from None
            except Exception:
                raise VisitorError(502, "Не удалось проверить подключение к почте.") from None
            with self.lock:
                self._require(session)
                # Replace the object on reconnection: a worker with the previous credentials cannot publish.
                account = _Account(provider, address, password, folder)
                if existing is not None:
                    account.id, account.rows, account.validity = existing.id, existing.rows, existing.validity
                    account.last_synced, account.latest = existing.last_synced, existing.latest
                    account.total_available, account.text_bytes = existing.total_available, existing.text_bytes
                    account.view, account.revision, account.index_lock = existing.view, existing.revision, existing.index_lock
                    existing.password = ""
                session.accounts[account.id] = account
                self._queue(session, account)
                return account.summary()
        finally:
            with self.lock:
                session.logging_in.discard(key)
            self.verifications.release()

    def inbox(self, session, mailbox):
        with self.lock:
            self._require(session)
            account = session.accounts.get(mailbox)
            if account is None:
                return None
            if account.view is not None:
                return account.view
            revision = account.revision
            rows = list(account.rows.values())
        view = VisitorInbox(mailbox, rows, encoder=self.encoder, index_lock=account.index_lock)
        with self.lock:
            self._require(session)
            if session.accounts.get(mailbox) is account and account.revision == revision:
                if account.view is None:
                    account.view = view
                return account.view
        return view

    def sync(self, session, mailbox):
        with self.lock:
            self._require(session)
            account = session.accounts.get(mailbox)
            if account is None:
                raise VisitorError(404, "Подключённый ящик не найден.")
            self._queue(session, account)
            return account.summary()

    def _queue(self, session, account):
        if not account.syncing:
            account.syncing = True
            self.workers.submit(self._sync, session, account)

    def _current(self, session, account):
        return self.active(session) and session.accounts.get(account.id) is account

    def _sync(self, session, account):
        with self.lock:
            if not self._current(session, account):
                return
            known = set(account.rows)
            validity, password = account.validity, account.password
        try:
            batch = self.remote.fetch_inbox(account.provider, account.address, password, mailbox=account.id,
                                            known_validity=validity, known_uids=known, max_messages=MAILBOX_LIMIT,
                                            folder=account.folder)
        except Exception as exc:
            # Never forward a provider response, password, address or message into a log or HTTP error.
            with self.lock:
                if self._current(session, account):
                    account.syncing = False
                    account.next_sync = self.clock() + SYNC_SECONDS
                    account.error = ("Проверьте пароль приложения и разрешение на чтение почты по IMAP."
                                     if isinstance(exc, transport.AuthenticationError) else "Не удалось обновить папку.")
            return
        with self.lock:
            if not self._current(session, account):
                return
            removed = known if batch.validity != validity else known - batch.current_uids
            for uid in removed:
                size = getsizeof(account.rows.pop(uid)["doc"])
                self.text_bytes -= size
                account.text_bytes -= size
            self.row_count -= len(removed)
            accepted = 0
            for uid in sorted(batch.messages, reverse=True):
                row = batch.messages[uid]
                size = getsizeof(row["doc"])
                if self.row_count >= PROCESS_ROWS or self.text_bytes + size > PROCESS_TEXT_BYTES:
                    break
                account.rows[uid] = row
                self.text_bytes += size
                account.text_bytes += size
                self.row_count += 1
                accepted += 1
            deferred = len(batch.messages) - accepted
            if removed or accepted:
                account.view = None
                account.revision += 1
                account.latest = max((r["doc_dt"] or "" for r in account.rows.values()), default="") or None
            account.validity = batch.validity
            account.pending = batch.remaining + deferred
            account.total_available = batch.total_available
            account.error = ("Сейчас недостаточно временной памяти. Уже загруженные письма доступны."
                             if deferred else None)
            account.last_synced = datetime.now(timezone.utc).isoformat()
            account.next_sync = self.clock() + SYNC_SECONDS
            account.syncing = False
            if batch.remaining and accepted and not deferred:
                # Separate submissions let a newly connected visitor run between batches of a large inbox.
                self._queue(session, account)
