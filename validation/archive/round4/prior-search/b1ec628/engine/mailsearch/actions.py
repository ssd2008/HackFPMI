"""Extract recipient requests and explicit deadlines with deterministic rules."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, time, timedelta
import re

from .dates import _MONTH, _MONTH_NUMBER


_INFINITIVE = (
    r"обработать|отгрузить|принять|рассмотреть|подтвердить|согласовать|"
    r"отправить|прислать|предоставить|заполнить|подписать|оплатить|погасить|"
    r"проверить|исправить|ответить|доставить|забрать|загрузить|обновить|выбрать|"
    r"передать|сообщить|завершить"
)
_IMPERATIVE = (
    r"обработайте|отгрузите|примите|рассмотрите|подтвердите|согласуйте|"
    r"отправьте|пришлите|предоставьте|заполните|подпишите|оплатите|погасите|"
    r"проверьте|исправьте|ответьте|доставьте|заберите|загрузите|обновите|"
    r"передайте|сообщите|завершите"
)
_REQUEST = re.compile(
    rf"\b(?:(?:просим|прошу|нужно|необходимо|требуется|следует|вам\s+надо)\s+(?:{_INFINITIVE})|"
    rf"(?<!не )(?<!уже )(?:{_IMPERATIVE}))\b",
    re.I,
)
_OPTIONAL = re.compile(
    r"\b(?:если|при\s+необходимости|при\s+желании|по\s+желанию|можно|можете|"
    r"рекоменду(?:ем|ется)|советуем|желательно|по\s+возможности|"
    r"не\s+(?:просим|прошу|нужно|надо|требуется|обязательно|следует|необходимо))\b",
    re.I,
)
_NEGATED = re.compile(r"\bне\s+$", re.I)
_ENVELOPE = re.compile(r"^(?:текст письма|тема)\s*:\s*", re.I)
_BOUNDARY = re.compile(r"[\r\n]+|(?<=[.!?;])\s+")
_NUMERIC = r"(?P<day>\d{1,2})[./](?P<month>\d{1,2})[./](?P<year>\d{4})"
_WORD_DATE = rf"(?P<word_day>\d{{1,2}})\s+(?P<word_month>{_MONTH})(?:\s+(?P<word_year>\d{{4}}))?"
_CLOCK = r"\d{1,2}:\d{2}"
_DEADLINE = re.compile(
    rf"\b(?:до|не\s+позднее|к)\s+"
    rf"(?:(?P<early_clock>{_CLOCK})\s*(?:мск|MSK)?,?\s+)?"
    rf"(?:(?P<iso>\d{{4}}-\d{{2}}-\d{{2}})|{_NUMERIC}|{_WORD_DATE}|"
    rf"(?P<relative>сегодня|завтра|послезавтра|конца\s+дня))"
    rf"(?:[\s,]+(?:в\s+|до\s+)?(?P<late_clock>{_CLOCK})\s*(?:мск|MSK)?)?(?!\w|:\d)",
    re.I,
)
_RELATIVE = {"сегодня": 0, "завтра": 1, "послезавтра": 2, "конца дня": 0}


@dataclass(frozen=True)
class MailAction:
    evidence: str
    deadline: str | None = None
    deadline_time: str | None = None


def _read_deadline(match: re.Match, received: str | None) -> tuple[str | None, str | None]:
    try:
        anchor = date.fromisoformat(received) if received else None
        clock = match["early_clock"] or match["late_clock"]
        if match["iso"]:
            due = date.fromisoformat(match["iso"])
        elif match["year"]:
            due = date(int(match["year"]), int(match["month"]), int(match["day"]))
        elif match["word_month"]:
            year = int(match["word_year"]) if match["word_year"] else anchor.year if anchor else None
            due = date(year, _MONTH_NUMBER[match["word_month"].casefold()], int(match["word_day"])) if year else None
        elif anchor is not None:
            relative = " ".join((match["relative"] or "сегодня").casefold().split())
            due = anchor + timedelta(days=_RELATIVE[relative])
        else:
            due = None
        if due is None:
            return None, None
        due_time = None
        if clock:
            hours, minutes = re.match(_CLOCK, clock)[0].split(":")
            due_time = time(int(hours), int(minutes)).isoformat(timespec="minutes")
        return due.isoformat(), due_time
    except (ValueError, OverflowError):
        return None, None


def extract_action(text: str, received: str | None) -> MailAction | None:
    """Return one request per letter, preferring the one with a supported deadline."""
    first = None
    for clause in _BOUNDARY.split(text):
        clause = _ENVELOPE.sub("", clause.strip())
        if not clause or clause.startswith(">") or re.match(r"(?:от|отправитель)\s*:", clause, re.I):
            continue
        request = _REQUEST.search(clause)
        if request is None:
            continue
        lead = clause[:request.start()]
        if _OPTIONAL.search(lead) or _NEGATED.search(lead):
            continue
        # Keep the date next to its request and avoid conditional or optional continuations.
        remainder = clause[request.end():request.start() + 230]
        optional = _OPTIONAL.search(remainder)
        if optional:
            remainder = remainder[:optional.start()]
        match = _DEADLINE.search(remainder)
        if match:
            deadline, clock = _read_deadline(match, received)
            if deadline:
                evidence = clause[max(0, request.start() - 25):request.end() + match.end()].strip()
                return MailAction(evidence, deadline, clock)
        if first is None:
            evidence = clause[max(0, request.start() - 25):request.end() + len(remainder)].strip(" ,;")
            first = MailAction(evidence)
    return first
