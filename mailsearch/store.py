"""Original messages, inbox views and mailbox-scoped access to a corpus snapshot."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import date, timedelta
from hashlib import file_digest
import json
from pathlib import Path
import re

from .actions import extract_action


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CACHE = ROOT / ".cache" / "mailsearch"

_HEAD = re.compile(r"^\s*(от|отправитель|тема|текст письма)\s*:\s*(.*)$", re.I)
_ORDER = re.compile(r"\bзаказ(?:а|у)?\s*(?:номер|№|no)?\s*(SYN-[0-9a-f]{6,})\b", re.I)
_IDENTIFIER = re.compile(r"\bSYN-[0-9a-f]{6,}\b", re.I)
_AMOUNT = re.compile(r"(?:сумм[ау]|стоимость)\s*[—–:\-]?\s*[\d\s]+[.,]\d\d", re.I)


def load_jsonl(path: Path, *, expected_sha256: str | None = None) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        if expected_sha256 is not None:
            if file_digest(source.buffer, "sha256").hexdigest() != expected_sha256:
                raise ValueError("Для публичного демо нужен исходный синтетический корпус VK.")
            source.seek(0)
        return [json.loads(line) for line in source if line.strip()]


def shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 1].rstrip() + "…"


def describe(row: dict) -> dict:
    lines = row["doc"].strip().splitlines()
    sender = subject = ""
    index = 0
    # Only the leading envelope is metadata. The reader always receives the original.
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        match = _HEAD.match(line)
        if not match:
            # Some anonymised envelopes replace the "От" label with a company.
            if index == 0 and len(lines) > 1 and ":" in line and re.match(r"\s*тема\s*:", lines[1], re.I):
                candidate = line.split(":", 1)[1].strip()
                if "@" in candidate:
                    sender = candidate
                index += 1
                continue
            break
        key, value = match.groups()
        key = key.casefold()
        if key == "текст письма":
            lines[index] = value
            break
        if key == "тема":
            subject = value
        else:
            sender = value
        index += 1
    body = " ".join(lines[index:]).strip()
    if not subject:
        opening = re.split(r"(?<=[.!?])\s+(?=[А-ЯЁA-Z«])", body, maxsplit=1)[0]
        subject = shorten(opening, 100) or "Без темы"
        if len(opening) <= 100 and body[len(opening):].strip():
            body = body[len(opening):].lstrip()
    return {
        "id": row["doc_id"], "subject": subject, "sender": sender or None,
        "date": row["doc_dt"], "snippet": shorten(body, 235),
    }


def make_examples(ordered: list[dict], summaries: dict[str, dict]) -> list[dict]:
    dated = [row for row in ordered if row["doc_dt"]]
    if not dated:
        return []
    latest = dated[0]
    # These suggested searches are derived from visible corpus text. The index
    # and the examples do not need the development queries or relevance labels.
    topics = (
        (r"новый заказ|получили заказ|заказ.{0,25}нужно обработать", "Какие новые заказы пришли вчера?", "Заказы за вчера"),
        (r"чек", "Какие чеки пришли вчера?", "Вчерашние чеки"),
        (r"штраф", "Какие штрафы выставили вчера?", "Штрафы за вчера"),
        (r"рецепт", "Какие сбои с электронными рецептами были вчера?", "Сбои за вчера"),
        (r"контейнер", "Какие письма о контейнерах пришли вчера?", "Контейнеры за вчера"),
        (r"приемк|приёмк", "Какие уведомления о приёмке пришли вчера?", "Приёмка за вчера"),
        (r"оплат|операци", "Что писали вчера о платежах?", "Платежи за вчера"),
    )
    opening = latest["doc"][:650].lower()
    query, label = next(((q, label) for pattern, q, label in topics if re.search(pattern, opening)),
                        ("Какие письма пришли вчера?", "Письма за вчера"))
    examples = [{"query": query, "label": label,
                 "date": (date.fromisoformat(latest["doc_dt"]) + timedelta(days=1)).isoformat()}]
    id_counts = Counter(token.upper() for row in ordered for token in set(_IDENTIFIER.findall(row["doc"].upper())))
    for row in dated:
        match = _ORDER.search(row["doc"])
        if match and id_counts[match[1].upper()] == 1 and _AMOUNT.search(row["doc"]):
            examples.append({"query": f"Какова сумма заказа {match[1]}?", "label": "Сумма заказа",
                             "date": row["doc_dt"]})
            break
    else:
        subject = summaries[latest["doc_id"]]["subject"]
        examples.append({"query": subject, "label": "Найти по теме", "date": latest["doc_dt"]})
    return examples


def _within_dates(value: str | None, at: str, date_from: str | None, date_to: str | None) -> bool:
    if value is None:
        return date_from is None and date_to is None
    return (date_from is None or value >= date_from) and value <= (date_to or at)


class MailStore:
    def __init__(self, documents: list[dict], *, empty_mailbox: str | None = None):
        self.by_id = {row["doc_id"]: row for row in documents}
        if len(self.by_id) != len(documents):
            raise ValueError("Повторяющийся doc_id в корпусе")
        groups: dict[str, list[dict]] = defaultdict(list)
        if empty_mailbox is not None:
            groups[empty_mailbox] = []
        self.summaries = {}
        self.actions_by_id = {}
        for row in documents:
            groups[row["synthetic_user"]].append(row)
            doc_id = row["doc_id"]
            self.summaries[doc_id] = describe(row)
            action = extract_action(row["doc"], row["doc_dt"])
            if action is not None:
                self.actions_by_id[doc_id] = action
        self.boxes = dict(groups)
        self.mailboxes = []
        for box, rows in sorted(self.boxes.items()):
            rows.sort(key=lambda row: (row["doc_dt"] or "", row["doc_id"]), reverse=True)
            self.mailboxes.append({
                "id": box, "count": len(rows), "latest_date": rows[0]["doc_dt"] if rows else None,
                "examples": make_examples(rows, self.summaries),
            })
        if not self.mailboxes:
            raise ValueError("Корпус пуст")

    def visible(self, box: str, at: str, doc_id: str, *, date_from=None, date_to=None) -> bool:
        row = self.by_id.get(doc_id)
        return bool(row and row["synthetic_user"] == box and _within_dates(row["doc_dt"], at, date_from, date_to))

    def inbox(self, box: str, at: str, offset: int, limit: int, *, date_from=None, date_to=None) -> dict:
        rows = [row for row in self.boxes[box] if _within_dates(row["doc_dt"], at, date_from, date_to)]
        return {"items": [self.summaries[row["doc_id"]] for row in rows[offset:offset + limit]],
                "total": len(rows), "offset": offset, "limit": limit}

    def action_inbox(self, box: str, at: date, offset: int, limit: int, marked=(), *, date_from=None, date_to=None) -> dict:
        cutoff = at.isoformat()
        marks = set(marked)
        items = []
        upcoming = []
        for row in self.boxes[box]:
            doc_id = row["doc_id"]
            action = self.actions_by_id.get(doc_id)
            manual = doc_id in marks
            if (action is None and not manual) or not _within_dates(row["doc_dt"], cutoff, date_from, date_to):
                continue
            item = {**self.summaries[doc_id], "evidence": action.evidence if action else None,
                    "deadline": action.deadline if action else None,
                    "deadline_time": action.deadline_time if action else None, "manual": manual}
            items.append(item)
            if item["deadline"] and 0 <= (date.fromisoformat(item["deadline"]) - at).days <= 1:
                upcoming.append(item)

        upcoming.sort(key=lambda item: (item["deadline"], item["deadline_time"] or "23:59"))
        return {"items": items[offset:offset + limit], "total": len(items),
                "offset": offset, "limit": limit, "urgent": upcoming[:3]}

    def reminder_candidates(self, box: str, at: str) -> list[dict]:
        return [
            {"id": row["doc_id"], "subject": self.summaries[row["doc_id"]]["subject"]}
            for row in self.boxes[box]
            if row["doc_id"] in self.actions_by_id and self.visible(box, at, row["doc_id"])
        ]
