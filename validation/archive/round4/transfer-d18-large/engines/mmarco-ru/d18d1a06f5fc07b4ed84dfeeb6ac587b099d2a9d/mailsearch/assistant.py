"""Mailbox-only assistance with lexical matching and explicit invitation dates."""

from datetime import date
import re

from .dates import _ATOM, _read_day


_WATCH = re.compile(r"\b(?:когда\s+(?:прид[её]т|поступит|появится)|как\s+только|отследи|подпишись|(?:сообщи|уведоми)\s+о\s+новом)\b", re.I)
_EVENT_QUESTION = re.compile(r"мероприят|(?:на|куда).{0,55}приглас|ближайш.{0,30}событи", re.I)
_INVITE = re.compile(r"приглаша(?:ем|ю)|приглашение\s+на|будем\s+рады\s+видеть", re.I)
_EVENT_DATE = re.compile(rf"\b(?:{_ATOM})(?!\w)", re.I)
_CLOCK = re.compile(r"\s*(?:в\s+)?((?:[01]?\d|2[0-3]):[0-5]\d)\b", re.I)
_NOT_EVENT_DATE = re.compile(r"(?:\bдо|не\s+позднее|срок[:\s]*)\s*$", re.I)
_WORD = re.compile(r"[а-яёa-z0-9]+", re.I)
_STOP = set("найди найдите ищи покажи покажите когда пока появится придет поступит письмо письма почта почте уведомление уведомить уведоми сообщи напомни подпишись отследи как только какое какие где которое который меня мне мой моя могу можно ближайшее время пожалуйста нас кто что если есть писал писали про подсказка расскажи расскажите узнать тема теме".split())


def _terms(query):
    words = dict.fromkeys(_WORD.findall(query.casefold().replace("ё", "е")))
    return tuple(word[:6] if len(word) > 6 else word for word in words if len(word) >= 3 and word not in _STOP)


def _matches(store, box, cutoff, query, after=None, seen=()):
    terms = _terms(query)
    if not terms:
        return []
    patterns = [re.compile(r"\b" + term) for term in terms]
    ignored = set(seen)
    matches = []
    for row in store.boxes[box]:
        doc_id, received = row["doc_id"], row["doc_dt"]
        if doc_id in ignored or not store.visible(box, cutoff, doc_id):
            continue
        if after and (not received or received < after):
            continue
        summary = store.summaries[doc_id]
        subject = summary["subject"].casefold().replace("ё", "е")
        text = row["doc"].casefold().replace("ё", "е")
        if all(pattern.search(text) for pattern in patterns):
            score = sum(bool(pattern.search(subject)) for pattern in patterns)
            matches.append((score, summary))
    return [summary for _, summary in sorted(matches, key=lambda match: match[0], reverse=True)]


def _events(store, box, at):
    cutoff = at.isoformat()
    end = min(date.max.toordinal(), at.toordinal() + 30)
    events = []
    for row in store.boxes[box]:
        if not store.visible(box, cutoff, row["doc_id"]) or not _INVITE.search(row["doc"]):
            continue
        for line in row["doc"].splitlines():
            if not _INVITE.search(line):
                continue
            for match in _EVENT_DATE.finditer(line):
                if _NOT_EVENT_DATE.search(line[:match.start()]):
                    continue
                raw = match.group().casefold()
                try:
                    # An event announced without a year belongs to the receipt year.
                    reference = date.fromisoformat(row["doc_dt"]) if row["doc_dt"] else at
                    if re.fullmatch(r"\d{1,2}\s+[а-я]+", raw):
                        raw += f" {reference.year}"
                    day = _read_day(raw, reference)
                except (ValueError, OverflowError):
                    continue
                if not at.toordinal() <= day.toordinal() <= end:
                    continue
                clock = _CLOCK.match(line[match.end():])
                evidence = line[max(0, match.start() - 150):match.end() + (clock.end() if clock else 0)].strip()
                item = {**store.summaries[row["doc_id"]], "event_date": day.isoformat(),
                        "event_time": clock[1].zfill(5) if clock else None, "evidence": evidence}
                events.append(item)
                break
            else:
                continue
            break
    return sorted(events, key=lambda item: (item["event_date"], item["event_time"] or "23:59"))[:8]


def assist(store, mailbox, at, query):
    if _WATCH.search(query):
        if not _terms(query):
            return {"kind": "mail", "items": [], "message": "Укажите тему письма, за которым хотите следить."}
        return {"kind": "watch", "items": [], "message": "Могу следить за такими письмами в выбранном ящике."}
    if _EVENT_QUESTION.search(query):
        return {"kind": "events", "items": _events(store, mailbox, at)}
    return {"kind": "mail", "items": _matches(store, mailbox, at.isoformat(), query)[:6]}


def watch_matches(store, mailbox, at, after, query, seen):
    items = _matches(store, mailbox, at.isoformat(), query, after.isoformat(), seen)
    return {"items": items[:5], "matched_ids": [item["id"] for item in items]}
