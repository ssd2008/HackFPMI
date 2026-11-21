"""Private, per-mailbox stores for the loopback application."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from hashlib import sha256
import json
import logging
from pathlib import Path
import sqlite3
from threading import Event, Lock, Thread

from filelock import FileLock

from . import mail_transport as transport
from .store import MailStore, load_corpus

LOG = logging.getLogger(__name__)
SYNC_SECONDS = 300
LOCAL_DATA = Path(__file__).resolve().parents[1] / ".cache" / "local-mail"


class LocalMailError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Credentials:
    """Only this application's passwords are placed in the operating system vault."""

    def __init__(self, root: Path):
        from keyring import get_keyring
        self.backend = get_keyring()
        namespace = sha256(str(root.resolve()).encode()).hexdigest()[:16]
        self.service = "HackFPMI.local-mail." + namespace

    def get(self, account):
        try:
            value = self.backend.get_password(self.service, account)
        except Exception as exc:
            raise LocalMailError(503, "Не удалось открыть системную связку ключей.") from exc
        if not value:
            raise LocalMailError(401, "Введите пароль приложения заново в настройках этого ящика.")
        return value

    def save(self, account, password):
        try:
            self.backend.set_password(self.service, account, password)
        except Exception as exc:
            raise LocalMailError(503, "Не удалось сохранить пароль в системной связке ключей.") from exc


class LocalInbox(MailStore):
    def __init__(self, mailbox, rows, index_dir, index_lock=None):
        super().__init__(rows, empty_mailbox=mailbox)
        self.mailbox = mailbox
        self.rows = rows
        self.index_dir = index_dir
        self.engine = None
        self.index_lock = index_lock or Lock()
        self.live = index_lock is not None

    def rank(self, query, at, limit):
        if not self.rows:
            return []
        if self.engine is None:
            with self.index_lock:
                if self.engine is None:
                    from .retrieval import SearchEngine
                    self.engine = SearchEngine(self.rows, self.index_dir)
                    if self.live:
                        # Earlier snapshots can finish their requests using the already-mapped file.
                        current = Path(self.engine.dense.vectors.filename)
                        for old in self.index_dir.glob("e5-*.npy"):
                            if old != current:
                                old.unlink(missing_ok=True)
                                old.with_suffix(".npy.lock").unlink(missing_ok=True)
        return self.engine.search(self.mailbox, query, at, limit)


class LocalMail:
    def __init__(self, root: Path, *, credentials=None, remote=transport):
        self.root = root.resolve()
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)
        self.lease = FileLock(self.root / ".app.lock", timeout=0)
        self.lease.acquire()
        try:
            self.remote = remote
            self.credentials = Credentials(self.root) if credentials is None else credentials
            self.database = self.root / "mail.sqlite3"
            self.database.touch(mode=0o600, exist_ok=True)
            self.database.chmod(0o600)
            self._setup()
            self.sergey = None
            imported = self.root / "sergey" / "documents.jsonl"
            if imported.exists():
                rows, _ = load_corpus(imported)
                boxes = {row["synthetic_user"] for row in rows}
                if len(boxes) != 1:
                    raise ValueError("Ожидался один ящик в локальном архиве Сергея.")
                self.sergey = LocalInbox(boxes.pop(), rows, self.root / "indexes")
            self.views = {}
            self.index_locks = {}
            self.views_lock = Lock()
            self.lock = Lock()
            self.syncing = set()
            self.stop = Event()
            self.workers = ThreadPoolExecutor(max_workers=2, thread_name_prefix="mail-sync")
            self.scheduler = Thread(target=self._schedule, daemon=True, name="mail-poll")
        except BaseException:
            self.lease.release()
            raise

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.database, timeout=20)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def _setup(self):
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY, provider TEXT NOT NULL, address TEXT NOT NULL,
                    validity TEXT, last_synced TEXT, error TEXT, pending INTEGER NOT NULL DEFAULT 0,
                    revision INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS messages (
                    account TEXT NOT NULL, uid INTEGER NOT NULL, doc_id TEXT NOT NULL,
                    doc_dt TEXT, doc TEXT NOT NULL, PRIMARY KEY(account, uid)
                );
                CREATE INDEX IF NOT EXISTS mail_date ON messages(account, doc_dt);
                CREATE TABLE IF NOT EXISTS submissions (
                    account TEXT NOT NULL, request_id TEXT NOT NULL, digest TEXT NOT NULL,
                    status TEXT NOT NULL, PRIMARY KEY(account, request_id)
                );
                UPDATE submissions SET status='uncertain' WHERE status='sending';
            """)

    def start(self):
        self.scheduler.start()

    def close(self):
        self.stop.set()
        if self.scheduler.is_alive():
            self.scheduler.join()
        self.workers.shutdown(wait=True, cancel_futures=True)
        self.lease.release()

    def _schedule(self):
        while not self.stop.is_set():
            with self.db() as db:
                ids = [r[0] for r in db.execute("SELECT id FROM accounts")]
            for account in ids:
                self.queue_sync(account)
            self.stop.wait(SYNC_SECONDS)

    def accounts(self):
        with self.db() as db:
            rows = db.execute("""
                SELECT a.id, a.provider, a.address, a.last_synced, a.error, a.pending,
                       count(m.uid) AS count, max(m.doc_dt) AS latest_date
                FROM accounts a LEFT JOIN messages m ON m.account=a.id
                GROUP BY a.id ORDER BY a.address, a.provider
            """).fetchall()
        with self.lock:
            active = self.syncing.copy()
        result = [dict(row) for row in rows]
        for row in result:
            row["status"] = "syncing" if row["id"] in active else ("error" if row["error"] else "idle")
        return result

    def boxes(self):
        boxes = []
        if self.sergey is not None:
            boxes.append({**self.sergey.mailboxes[0], "label": "Ящик Сереги", "kind": "archive"})
        for account in self.accounts():
            boxes.append(dict(id=account["id"], label=account["address"], kind="live",
                              count=account["count"], latest_date=account["latest_date"],
                              examples=[], today=date.today().isoformat()))
        return boxes

    def inbox(self, mailbox):
        if self.sergey is not None and mailbox == self.sergey.mailbox:
            return self.sergey
        with self.views_lock, self.db() as db:
            db.execute("BEGIN")
            account = db.execute("SELECT revision FROM accounts WHERE id=?", (mailbox,)).fetchone()
            if account is None:
                return None
            cached = self.views.get(mailbox)
            if cached is None or cached[0] != account["revision"]:
                rows = [dict(row, synthetic_user=mailbox) for row in db.execute(
                    "SELECT doc_id, doc_dt, doc FROM messages WHERE account=? ORDER BY uid", (mailbox,))]
                index_lock = self.index_locks.setdefault(mailbox, Lock())
                inbox = LocalInbox(mailbox, rows, self.root / "indexes" / mailbox, index_lock)
                self.views[mailbox] = (account["revision"], inbox)
            return self.views[mailbox][1]

    def connect(self, provider, address, password):
        # A failed credential update leaves an existing working connection intact.
        self.remote.verify_account(provider, address, password)
        key = sha256((provider + "\0" + address).encode()).hexdigest()[:24]
        account = "account-" + key
        with self.lock:
            self.credentials.save(account, password)
            with self.db() as db:
                db.execute("""
                    INSERT INTO accounts(id,provider,address) VALUES(?,?,?)
                    ON CONFLICT(id) DO UPDATE SET error=NULL
                """, (account, provider, address))
        return self.sync(account)

    def account(self, account):
        with self.db() as db:
            row = db.execute("SELECT * FROM accounts WHERE id=?", (account,)).fetchone()
        if row is None:
            raise LocalMailError(404, "Подключённый аккаунт не найден.")
        return row

    def sync(self, account):
        self.account(account)
        self.queue_sync(account)
        return next(row for row in self.accounts() if row["id"] == account)

    def queue_sync(self, account):
        with self.lock:
            if account in self.syncing or self.stop.is_set():
                return
            self.syncing.add(account)
            self.workers.submit(self._sync, account)

    def _sync(self, account):
        try:
            row = self.account(account)
            password = self.credentials.get(account)
            while not self.stop.is_set():
                with self.db() as db:
                    validity = db.execute("SELECT validity FROM accounts WHERE id=?", (account,)).fetchone()[0]
                    known = {r[0] for r in db.execute("SELECT uid FROM messages WHERE account=?", (account,))}
                batch = self.remote.fetch_inbox(row["provider"], row["address"], password, mailbox=account,
                                                known_validity=validity, known_uids=known)
                with self.db() as db:
                    if validity != batch.validity:
                        removed = known
                        db.execute("DELETE FROM messages WHERE account=?", (account,))
                    else:
                        removed = known - batch.current_uids
                        db.executemany("DELETE FROM messages WHERE account=? AND uid=?",
                                       ((account, uid) for uid in removed))
                    db.executemany("INSERT OR REPLACE INTO messages(account,uid,doc_id,doc_dt,doc) VALUES(?,?,?,?,?)",
                                   ((account, uid, doc["doc_id"], doc["doc_dt"], doc["doc"])
                                    for uid, doc in batch.messages.items()))
                    db.execute("""UPDATE accounts SET validity=?, last_synced=?, error=NULL,
                                  pending=?, revision=revision+? WHERE id=?""",
                               (batch.validity, datetime.now(timezone.utc).isoformat(), batch.remaining,
                                int(bool(removed or batch.messages)), account))
                if not batch.remaining or not batch.messages:
                    break
        except (transport.TransportError, LocalMailError) as exc:
            with self.db() as db:
                db.execute("UPDATE accounts SET error=? WHERE id=?", (str(exc), account))
        except Exception as exc:
            LOG.error("Mailbox sync failed (%s)", type(exc).__name__)
            with self.db() as db:
                db.execute("UPDATE accounts SET error=? WHERE id=?", ("Не удалось обновить входящие.", account))
        finally:
            with self.lock:
                self.syncing.discard(account)

    def send(self, account, recipients, subject, body, request_id):
        row = self.account(account)
        digest = sha256(json.dumps([recipients, subject, body], ensure_ascii=False).encode()).hexdigest()
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            receipt = db.execute("SELECT digest,status FROM submissions WHERE account=? AND request_id=?",
                                 (account, request_id)).fetchone()
            if receipt:
                if receipt["digest"] != digest:
                    raise LocalMailError(409, "Письмо изменилось. Начните новую отправку.")
                if receipt["status"] == "sent":
                    return
                if receipt["status"] == "uncertain":
                    raise LocalMailError(409, "Сервер не подтвердил результат. Проверьте «Отправленные» у провайдера.")
                raise LocalMailError(409, "Это письмо уже отправляется.")
            db.execute("INSERT INTO submissions VALUES(?,?,?,'sending')", (account, request_id, digest))
        state = "uncertain"
        try:
            password = self.credentials.get(account)
            message_id = f"<{request_id}.{account}@{row['address'].rsplit('@', 1)[1]}>"
            self.remote.send_message(row["provider"], row["address"], password, to=recipients,
                                     subject=subject, body=body, message_id=message_id)
            state = "sent"
        except transport.DeliveryUncertain:
            raise
        except (transport.TransportError, LocalMailError):
            state = "failed"
            raise
        finally:
            with self.db() as db:
                if state == "failed":
                    db.execute("DELETE FROM submissions WHERE account=? AND request_id=?", (account, request_id))
                else:
                    db.execute("UPDATE submissions SET status=? WHERE account=? AND request_id=?",
                               (state, account, request_id))
