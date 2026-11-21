"""Interpret receipt-date filters relative to the mailbox snapshot."""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, timedelta
import re
import unicodedata


_MONTHS = (
    ("январь", "января", "январе"), ("февраль", "февраля", "феврале"),
    ("март", "марта", "марте"), ("апрель", "апреля", "апреле"),
    ("май", "мая", "мае"), ("июнь", "июня", "июне"),
    ("июль", "июля", "июле"), ("август", "августа", "августе"),
    ("сентябрь", "сентября", "сентябре"), ("октябрь", "октября", "октябре"),
    ("ноябрь", "ноября", "ноябре"), ("декабрь", "декабря", "декабре"),
)
_MONTH_NUMBER = {form: i for i, forms in enumerate(_MONTHS, 1) for form in forms}
_MONTH = "(?:" + "|".join(sorted(_MONTH_NUMBER, key=len, reverse=True)) + ")"
_YEAR_SUFFIX = r"(?:\s*(?:года?|г\.?))?"
_WORD_DATE = rf"\d{{1,2}}\s+{_MONTH}(?:\s+\d{{4}}{_YEAR_SUFFIX})?"
_ATOM = rf"(?:\d{{4}}-\d{{2}}-\d{{2}}|\d{{1,2}}[./]\d{{1,2}}[./]\d{{4}}|{_WORD_DATE})"
_DAYS = {"сегодня": 0, "вчера": 1, "позавчера": 2}
_ISO = r"\d{4}-\d{2}-\d{2}"
_EN_DAY = rf"(?:{_ISO}|yesterday|today)"
_EN_RECEIPT = re.compile(
    r"\b(?:receiv(?:e|ed)|arrived|(?:e-?mails?|messages?|letters?)\s+(?:dated|delivered)|"
    r"e-?mails?|messages?|letters?)\s+"
    rf"(?:(?:between\s+(?P<between>{_EN_DAY})\s+and|from\s+(?P<from>{_EN_DAY})\s+(?:to|through))"
    rf"\s+(?P<end>{_EN_DAY})(?:,?\s+inclusive)?|"
    rf"(?:(?:during|in)\s+)?(?:the\s+)?week\s+of\s+(?P<week>{_ISO})|"
    rf"(?:(?P<bound>on|before|after|since|through)\s+)?(?P<day>{_EN_DAY}))(?!\w)"
)
_FACT_QUESTION = re.compile(
    r"^\s*(?:(?:в|на|у|из|к|для|от|с)\s+)?(?:кто|кого|кому|кем|что|чем|чего|"
    r"как(?:ой|ая|ое|ие|ого|ую|их|ом|им|ими)?|где|откуда|когда|сколько|почему|"
    r"чей|чья|чьи|каков[а-я]*|who|whom|whose|what|when|where|why|how|which)\b"
)
_MAIL_CONTEXT = re.compile(
    r"\b(?:письм[а-я]*|сообщени[а-я]*|рассылк[а-я]*|уведомлени[а-я]*|входящ[а-я]*|"
    r"квитанци[а-я]*|чеки?|счета?|заказы?|пришедш[а-я]*|присла[а-я]*|полученн[а-я]*|"
    r"(?:мне|нам|я|мы)\s+(?:приш[а-я]*|получи[а-я]*)|отправител[а-я]*|почт[аеуы]|"
    r"e-?mails?|messages?|letters?|inbox)\b"
)
_RELATIVE_RECEIPT = re.compile(
    r"\b(?:вчера|сегодня|позавчера|за\s+(?:(?:последн|прошл|минувш)[а-я]*|период)|"
    r"yesterday|today|last\s+(?:week|month))\b"
)
# Deadlines and dates quoted from the subject/body can coexist with a receipt-date filter.
_EVENT = (
    r"(?:(?:срок(?:ом|и)?|дат[аыуе])\s+(?:оплаты|отгрузки|доставки|вылета|выезда|встречи)(?:\s+заказа)?|"
    r"оплатить|отгрузить|доставить|вылететь|отправить\s+заказ|"
    r"(?:доставк(?:а|ой|у)|вылет|выезд|встреч[ау])(?=\s+(?:на|к|до)\s))"
)
_FIELD = r"в\s+(?:теме|заголовке|тексте|теле)"
_CONTENT_DATE = re.compile(
    rf"\b(?:{_EVENT}\s*[:—–-]?\s*(?:(?:назначен[ао]?\s+)?(?:на|к|до|после)\s+)?{_ATOM}|"
    rf"{_FIELD}\s+(?:письм[ае]\s+)?(?:(?:указан[ао]?|упомянут[ао]?|содержится|есть|стоит)\s+)?"
    rf"(?:дат[ауы]\s*)?[:—–-]?\s*{_ATOM}|(?:дат(?:а|у|ой)\s+)?{_ATOM}\s+{_FIELD})(?!\w)"
)
_YEAR = re.compile(
    r"\b(?:за|в)\s+(\d{4})\b(?![.,]\d|\s*(?:[₽€$%]|руб\w*|доллар\w*|евро|rub\b|usd\b|eur\b))"
    r"(?:\s*(?:год(?:у|а)?\b|г\.))?"
)


@dataclass(frozen=True)
class DateScope:
    text: str
    start: date | None = None
    end: date | None = None

    @property
    def bounded(self) -> bool:
        return self.start is not None or self.end is not None


def _last_day(day: int, month: int, reference: date) -> date:
    # An unqualified leap day can be several years back.
    for year in range(reference.year, max(0, reference.year - 9), -1):
        if day <= calendar.monthrange(year, month)[1]:
            candidate = date(year, month, day)
            if candidate <= reference:
                return candidate
    raise ValueError("No valid matching date")


def _read_day(raw: str, reference: date) -> date:
    if raw in _DAYS:
        return reference - timedelta(days=_DAYS[raw])
    if raw in {"today", "yesterday"}:
        return reference - timedelta(days=int(raw == "yesterday"))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw):
        return date.fromisoformat(raw)
    if "." in raw[:3] or "/" in raw[:3]:
        day, month, year = map(int, re.split(r"[./]", raw))
        return date(year, month, day)
    match = re.fullmatch(rf"(\d+)\s+({_MONTH})(?:\s+(\d{{4}}){_YEAR_SUFFIX})?", raw)
    if match is None:
        raise ValueError("Unrecognized date")
    day, month = int(match[1]), _MONTH_NUMBER[match[2]]
    return date(int(match[3]), month, day) if match[3] else _last_day(day, month, reference)


def parse_date_scope(query: str, reference: date) -> DateScope:
    original = " ".join(unicodedata.normalize("NFKC", query).lower().replace("ё", "е").split())
    # Preserve offsets so only a receipt-date expression can be removed from the original terms.
    text = _CONTENT_DATE.sub(lambda m: " " * len(m[0]), original)

    def scope(match: re.Match, start: date | None, end: date | None) -> DateScope:
        return DateScope((original[:match.start()] + " " + original[match.end():]).strip(), start, end)

    try:
        match = _EN_RECEIPT.search(text)
        if match:
            if match["end"]:
                return scope(match, _read_day(match["between"] or match["from"], reference),
                             _read_day(match["end"], reference))
            if match["week"]:
                anchor = date.fromisoformat(match["week"])
                monday = anchor - timedelta(days=anchor.weekday())
                return scope(match, monday, monday + timedelta(days=6))
            day = _read_day(match["day"], reference)
            bound = match["bound"]
            if bound in {"since", "after"}:
                return scope(match, day + timedelta(days=int(bound == "after")), None)
            if bound in {"before", "through"}:
                return scope(match, None, day - timedelta(days=int(bound == "before")))
            return scope(match, day, day)

        # A year in "Who won the championship in 2012?" identifies the subject.
        # "Which messages arrived in 2012?" asks about the mailbox timeline.
        if _FACT_QUESTION.match(text) and not _MAIL_CONTEXT.search(text) and not _RELATIVE_RECEIPT.search(text):
            return DateScope(original)

        match = re.search(rf"\bс\s+(\d{{1,2}})\s+(?:по|до)\s+({_WORD_DATE})(?!\w)", text)
        if match:
            end = _read_day(match[2], reference)
            day = int(match[1])
            month = end if day <= end.day else end.replace(day=1) - timedelta(days=1)
            return scope(match, date(month.year, month.month, day), end)

        atom = rf"(?:{_ATOM}|сегодня|позавчера|вчера)"
        match = re.search(rf"(?<!\w)(?:с\s+)?({atom})\s*(?:по|до|[–—])\s*({atom})(?!\w)", text)
        if match:
            end = _read_day(match[2], reference)
            start = _read_day(match[1], end if match[1] not in _DAYS else reference)
            return scope(match, start, end)

        match = re.search(rf"(?<!\w)(?:(до|раньше|после|с|по)\s+)?({atom})(?!\w)", text)
        if match:
            day = _read_day(match[2], reference)
            if match[1] in {"до", "раньше"}:
                return scope(match, None, day - timedelta(days=1))
            if match[1] == "по":
                return scope(match, None, day)
            if match[1] in {"после", "с"}:
                return scope(match, day + timedelta(days=match[1] == "после"), None)
            return scope(match, day, day)

        match = re.search(r"\b(?:вчерашн|сегодняшн|позавчерашн)[а-я]+\b", text)
        if match:
            days = 2 if match[0].startswith("поза") else int(match[0].startswith("вчера"))
            day = reference - timedelta(days=days)
            return scope(match, day, day)

        match = re.search(r"\b(\d{1,4})\s+(?:день|дня|дней)\s+назад\b", text)
        if match:
            day = reference - timedelta(days=int(match[1]))
            return scope(match, day, day)

        match = re.search(
            r"\b(?:за\s+)?(?:последни[ехйю]\s+)?(\d{1,4})\s+"
            r"(день|дня|дней|недели|недель|неделю|месяца|месяцев|месяц)\b", text
        )
        if match:
            unit = 7 if match[2].startswith("недел") else 30 if match[2].startswith("месяц") else 1
            count = int(match[1]) * unit
            return scope(match, reference - timedelta(days=count - 1), reference)

        match = re.search(
            r"\b(?:за|на|в)\s+(?:(?P<period>прошл(?:ую|ой|ый|ом)|минувш(?:ую|ей|ий|ем)|"
            r"эт(?:у|ой|от|ом)|текущ(?:ую|ей|ий|ем)|последн(?:юю|ей|ий|ем))\s+)?"
            r"(?:(?P<calendar>календарн(?:ую|ой|ый|ом))\s+)?(?P<unit>недел[юеи]|месяце?)\b", text
        )
        if match:
            period = match["period"] or ""
            calendar_period = bool(match["calendar"])
            previous = period.startswith(("прошл", "минувш")) or (calendar_period and period.startswith("последн"))
            current = period.startswith(("эт", "текущ")) or (calendar_period and not period)
            if match["unit"].startswith("недел"):
                monday = reference - timedelta(days=reference.weekday())
                if previous:
                    return scope(match, monday - timedelta(days=7), monday - timedelta(days=1))
                return scope(match, monday if current else reference - timedelta(days=6), reference)
            first = reference.replace(day=1)
            if previous:
                last = first - timedelta(days=1)
                return scope(match, last.replace(day=1), last)
            return scope(match, first if current else reference - timedelta(days=29), reference)

        match = re.search(rf"\b(?:за|в)\s+({_MONTH})(?:\s+(\d{{4}}){_YEAR_SUFFIX})?\b", text)
        if match:
            month = _MONTH_NUMBER[match[1]]
            year = int(match[2]) if match[2] else reference.year - int(month > reference.month)
            return scope(match, date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1]))

        match = _YEAR.search(text)
        if match:
            year = int(match[1])
            return scope(match, date(year, 1, 1), date(year, 12, 31))
    except (ValueError, OverflowError):
        # An invalid explicit date is an unsatisfiable request.
        return DateScope(original, date.max, date.min)
    return DateScope(original)
