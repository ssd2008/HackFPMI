"""A visitor-owned, immutable view of a synchronized inbox."""

from threading import BoundedSemaphore, Lock

from .store import MailStore


_INDEX_BUILDS = BoundedSemaphore(2)


class VisitorInbox(MailStore):
    def __init__(self, mailbox, rows, *, encoder, index_lock: Lock):
        super().__init__(rows, empty_mailbox=mailbox)
        self.mailbox = mailbox
        self.rows = rows
        self.encoder = encoder
        self.index_lock = index_lock
        self.engine = None
        self.attachment_cache = {}

    def rank(self, query, at, limit, *, date_from=None, date_to=None):
        if not self.rows:
            return []
        if self.engine is None:
            with self.index_lock:
                if self.engine is None:
                    from .retrieval import SearchEngine

                    with _INDEX_BUILDS:
                        self.engine = SearchEngine(self.rows, None, encoder=self.encoder)
        return self.engine.search(self.mailbox, query, at, limit, date_from=date_from, date_to=date_to)
