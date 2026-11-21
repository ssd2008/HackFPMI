"""Explicit sender and subject predicates over leading message headers."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from email.utils import getaddresses
import re
import unicodedata

import numpy as np


_HEADER = re.compile(r"^\s*(от|отправитель|from|sender|тема|subject|кому|to|cc|дата|date)\s*:\s*(.*)$", re.I)
_LEAD = re.compile(r"^\s*(?:текст письма|body)\s*:", re.I)
_TOPIC = re.compile(
    r"(?<!\w)(?:тем(?:а|ой|у|е)|заголов(?:ок|ком)|subject)"
    r"(?:\s+(?:целиком|полностью|точно|буквально|совпадает|равна?|с|is|equals|exactly|matches))*"
    r"\s*[:=]?\s*([«“\"])",
    re.I,
)
_DOMAIN = r"(?:[\w-]+\.)+[^\W\d_][\w-]*"
_ADDRESS = rf"[\w.!#$%&'*+/=?^_{{|}}~-]+@{_DOMAIN}"
_FROM = re.compile(
    rf"(?<!\w)(?P<field>from|sender|отправител[ья]|от|домен(?:а)?\s+отправителя|"
    rf"(?:с|от)\s+адрес(?:ов|а)?(?:\s+на)?|sender(?:'s)?\s+domain)"
    rf"(?:\s*[:=—–-]\s*|\s+)(?P<value>{_ADDRESS}|@{_DOMAIN}|{_DOMAIN}|[«“\"])",
    re.I,
)
_DONE = frozenset("""
    find show get list locate search for in mail email emails messages message letter letters the all any
    received on dated from by is there an a whose that have with exact full entire subject title during of between and
    найди найдите найти покажи покажите найти поиск по в во на от за с из и у где есть ли
    письмо письма писем письме переписка переписку сообщение сообщения сообщений почта мне все мои
    какого каком этом которого которых которое которые этот эти этот тех те тему темой полной полной точной
    полным точным заголовком полученные полученных полученное получил присланные пришедшие отправленные
    период
""".split())
_WORD = re.compile(r"[^\W_]+", re.U)
_WHITESPACE = re.compile(r"\s+")
_LITERAL_END = {'«': '»', '“': '”', '"': '"'}


def normalize(value):
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")).strip()


def headers(doc):
    subject = doc.get("subject") or ""
    sender = doc.get("sender") or doc.get("from") or ""
    if not isinstance(subject, str):
        subject = ""
    if not isinstance(sender, str):
        sender = ""
    # Text in a forwarded/quoted body must not become the envelope sender.
    for line in doc["doc"][:10000].splitlines()[:32]:
        if not line.strip():
            if subject or sender:
                break
            continue
        if _LEAD.match(line):
            break
        match = _HEADER.match(line)
        if not match:
            break
        key, value = match[1].casefold(), match[2].strip()
        if key in {"от", "отправитель", "from", "sender"} and not sender:
            sender = value
        elif key in {"тема", "subject"} and not subject:
            subject = value
        if subject and sender:
            break
    return normalize(subject), sender


def _quote(text, opening, is_known=None):
    """Honor nested angle quotes and accept embedded quote marks in a header value."""
    start = opening + 1
    closing = _LITERAL_END[text[opening]]
    if closing == "»":
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "«":
                depth += 1
            elif text[i] == closing:
                depth -= 1
                if not depth:
                    return text[start:i], i + 1
        return None
    possible = []
    for match in re.finditer(re.escape(closing), text[start:]):
        i = start + match.start()
        if i == start or text[i-1] == '\\':
            continue
        if i + 1 == len(text) or text[i+1] in " .,!?:;)]":
            possible.append((text[start:i], i + 1))
    if is_known and len(possible) > 1:
        for pair in reversed(possible):
            if is_known(pair[0]):
                return pair
    return possible[0] if possible else None


def _inside_literal(query, offset):
    # Field words in a requested quotation describe its content.
    i = 0
    while i < offset:
        if query[i] in _LITERAL_END:
            span = _quote(query, i)
            if span:
                if span[1] > offset:
                    return True
                i = span[1]
                continue
        i += 1
    return False


def only_field_request(text):
    return all(word.casefold() in _DONE for word in _WORD.findall(text))


class MailFields:
    def __init__(self, documents):
        postings = defaultdict(list)
        self.addressed_subjects = {}
        self.first_sender: dict[str, date] = {}
        for i, doc in enumerate(documents):
            subject, sender = headers(doc)
            if subject:
                postings["subject", subject].append(i)
            if sender:
                # Undated messages are visible at every mailbox snapshot.
                received = date.fromisoformat(doc["doc_dt"]) if doc.get("doc_dt") else date.min
                mailbox = doc.get("synthetic_user")
                if mailbox is not None and received <= self.first_sender.get(mailbox, date.max):
                    self.first_sender[mailbox] = received
                terms = set()
                for name, address in getaddresses([sender]):
                    address = normalize(address)
                    if "@" in address:
                        self.addressed_subjects[i] = subject
                        terms.add(("address", address))
                        terms.add(("domain", address.rsplit("@", 1)[1]))
                    if name:
                        terms.add(("name", normalize(name)))
                for term in terms:
                    postings[term].append(i)
        self.postings = {key: np.asarray(rows, dtype=np.int32) for key, rows in postings.items()}

    def has_visible_sender(self, mailbox: str, at: date) -> bool:
        first = self.first_sender.get(mailbox)
        return first is not None and first <= at

    def restrict(self, query, box):
        """Return residual text and candidates. A bare email still follows normal retrieval."""
        spans = []
        if self.postings and any(c in query for c in '«“"'):
            for match in _TOPIC.finditer(query):
                if _inside_literal(query, match.start()):
                    continue
                val = _quote(query, match.start(1), lambda title: ("subject", normalize(title)) in self.postings)
                if val:
                    value, end = val
                    spans.append((match.start(), end, ("subject", normalize(value))))
        if self.postings and ("@" in query or re.search(r"from|sender|отправител|домен|адрес|от\s*(?:[«“\"]|[\w.-]+\.)", query, re.I)):
            for match in _FROM.finditer(query):
                if any(start <= match.start() < end for start, end, _ in spans) or _inside_literal(query, match.start()):
                    continue
                value = match["value"]
                end = match.end()
                if value in _LITERAL_END:
                    literal = _quote(query, match.start("value"))
                    if not literal:
                        continue
                    value, end = literal
                    kind = "name"
                elif "@" in value and not value.startswith("@"):
                    kind = "address"
                else:
                    kind = "domain"
                spans.append((match.start(), end, (kind, normalize(value.lstrip("@")))))
        if not spans:
            return query, box, False
        parts = []
        last = 0
        for start, end, key in sorted(spans):
            if start < last:
                continue
            parts.append(query[last:start])
            last = end
            rows = self.postings.get(key)
            if rows is None:
                return "", box[:0], True
            box = np.intersect1d(box, rows, assume_unique=True)
        parts.append(query[last:])
        return " ".join(parts), box, True
