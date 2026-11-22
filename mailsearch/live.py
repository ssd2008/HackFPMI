"""Read scoped files on demand and reuse extracted descriptions within a service instance."""

from pathlib import Path
from difflib import SequenceMatcher
from threading import Lock
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from itertools import islice
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError
import os

from .attachment_worker import extract
from .dates import parse_date_scope
from .retrieval import _words, _stem


class AttachmentSearch:
    def __init__(self, store, files, tessdata):
        self.store = store
        self.files = Path(files).resolve()
        self.tessdata = Path(tessdata)
        self.pool = ProcessPoolExecutor(
            max_workers=min(3, os.cpu_count() or 1), mp_context=get_context("spawn")
        )
        self.cache = OrderedDict()
        self.lock = Lock()

    def close(self):
        self.pool.shutdown()
        self.cache.clear()

    def _schedule(self, source, relative):
        path = (self.files / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(self.files):
            raise ValueError("Путь за пределами каталога вложений")
        stat = path.stat()
        item = source.get("attachments", [{}])[0]
        key = (
            path,
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
            "eml_path" in source,
            item.get("filename"),
            item.get("mime_type"),
            relative,
        )
        with self.lock:
            if key not in self.cache:
                self.cache[key] = self.pool.submit(
                    extract, source, self.files, self.tessdata
                )
                if len(self.cache) > 2048:
                    self.cache.popitem(last=False)
            self.cache.move_to_end(key)
            return key, self.cache[key]

    def _prepare(self, row):
        parts = []
        sources = [
            ({"attachments": [a]}, a.get("path")) for a in row.get("attachments", [])
        ]
        if row.get("eml_path"):
            sources.append(({"eml_path": row["eml_path"]}, row["eml_path"]))
        for source, relative in sources:
            if relative is None:
                parts.append((None, source["attachments"]))
                continue
            try:
                parts.append(self._schedule(source, relative))
            except (ValueError, OSError):
                parts.append((None, None))
        return parts

    def _collect(self, parts):
        items = []
        complete = True
        for key, part in parts:
            try:
                result = part.result() if key is not None else part
            except (
                OSError,
                ValueError,
                RuntimeError,
                KeyError,
                BadZipFile,
                ParseError,
                ImportError,
            ):
                with self.lock:
                    if self.cache.get(key) is part:
                        self.cache.pop(key)
                result = None
            if result is None:
                complete = False
            else:
                items.extend(result)
                complete &= all(
                    bool(item.get("description")) and not item.get("partial")
                    for item in result
                )
        return items, complete

    def read(self, row):
        return self._collect(self._prepare(row))

    def search(self, mailbox, query, at, limit, base_hits):
        scope = parse_date_scope(query, at)
        low = scope.start.isoformat() if scope.start else ""
        high = min(at, scope.end or at).isoformat()

        def words(text):
            return {_stem(w) for w in _words(text) if len(w) > 2}

        terms = words(scope.text)

        def match_weight(candidates):
            return sum(
                1.0
                if t in candidates
                else 0.7
                * any(
                    len(t) >= 5
                    and abs(len(t) - len(c)) <= 1
                    and t[:3] == c[:3]
                    and SequenceMatcher(None, t, c).ratio() >= 0.88
                    for c in candidates
                )
                for t in terms
            )

        ids = {w for w in _words(scope.text) if w.startswith("l-")}
        scores = dict(base_hits)
        matched = {}
        partial = False

        def visible_files():
            for row in self.store.boxes[mailbox]:
                did, day = row["doc_id"], row["doc_dt"] or ""
                if (
                    not low <= day <= high
                    or (scope.bounded and not day)
                    or (ids and did.lower() not in ids)
                ):
                    continue
                if row.get("attachments") or row.get("eml_path"):
                    yield row

        rows = visible_files()
        while batch := list(islice(rows, 12)):
            pending = [(row, self._prepare(row)) for row in batch]
            for row, parts in pending:
                did = row["doc_id"]
                items, complete = self._collect(parts)
                partial |= not complete
                summary = self.store.summaries[did]
                sender = words(summary["sender"] or "")
                # The same file sent by two people is disambiguated by the envelope.
                coverage = max(
                    (
                        match_weight(
                            words(
                                item.get("description", "")
                                + " "
                                + item.get("filename", "")
                            )
                        )
                        for item in items
                    ),
                    default=0,
                )
                if coverage:
                    matched[did] = items
                    scores[did] = (
                        scores.get(did, 0.0) + 2.0 * coverage + len(terms & sender)
                    )
        return sorted(scores.items(), key=lambda hit: -hit[1])[:limit], matched, partial
