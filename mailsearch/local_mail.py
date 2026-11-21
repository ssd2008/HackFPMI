"""Read-only localhost archive and immutable mailbox views shared by the search API."""

from contextlib import nullcontext
from pathlib import Path
from threading import BoundedSemaphore, Lock

from .store import MailStore, load_corpus

LOCAL_DATA = Path(__file__).resolve().parents[1] / ".cache" / "local-mail"
_PRIVATE_INDEX_BUILDS = BoundedSemaphore(2)


class LocalInbox(MailStore):
    def __init__(self, mailbox, rows, *, index_dir=None, encoder=None, index_lock=None):
        super().__init__(rows, empty_mailbox=mailbox)
        self.mailbox = mailbox
        self.rows = rows
        self.index_dir = index_dir
        self.encoder = encoder
        self.engine = None
        self.index_lock = index_lock or Lock()

    def rank(self, query, at, limit):
        if not self.rows:
            return []
        if self.engine is None:
            with self.index_lock:
                if self.engine is None:
                    from .retrieval import SearchEngine
                    with _PRIVATE_INDEX_BUILDS if self.encoder is not None else nullcontext():
                        self.engine = SearchEngine(self.rows, self.index_dir, encoder=self.encoder)
        return self.engine.search(self.mailbox, query, at, limit)


class LocalArchive:
    """Only the separately imported archive. Never open previous connection databases or host credentials."""

    def __init__(self, root: Path):
        self.view = None
        path = root / "sergey" / "documents.jsonl"
        if path.is_file():
            rows, _ = load_corpus(path)
            boxes = {row["synthetic_user"] for row in rows}
            if len(boxes) != 1:
                raise ValueError("Ожидался один ящик в локальном архиве Сергея.")
            self.view = LocalInbox(boxes.pop(), rows, index_dir=root / "indexes")

    def boxes(self):
        return [{**self.view.mailboxes[0], "label": "Ящик Сереги", "kind": "archive"}] if self.view else []

    def inbox(self, mailbox):
        return self.view if self.view and self.view.mailbox == mailbox else None
