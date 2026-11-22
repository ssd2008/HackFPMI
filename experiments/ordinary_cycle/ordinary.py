"""Distinguish completed events from a permission, request or plan for the same event."""
from functools import lru_cache
import json
from math import inf, nextafter
import re

from mailsearch.dates import parse_date_scope
from mailsearch.retrieval import _features, _words
from mailsearch.store import ROOT


CONFIG = ROOT / "results/ordinary_cycle/ordinary-weights.json"
COMMAND = re.compile(r"^\s*(?:покаж\w*|найди\w*|отыщи\w*|выведи\w*|перечисли\w*|какие)\b", re.I)
REPORT = re.compile(r"\b(?:уведомлени\w*|извещени\w*|отч[её]т\w*|подтверждени\w*)\s+об?\s|"
                    r"\b(?:состоял\w*|выполнен\w*|заверш[её]н\w*)\b", re.I)
BROAD = re.compile(r"\b(?:переписк\w*|упоминани\w*|связан\w*|касавш\w*|касающ\w*|"
                   r"любые|по\s+теме|вс[её]\s+(?:по|про|что))\b", re.I)
STRUCTURED = re.compile(r'[@"«»]|\b(?:тем[а-я]*\s*[:=]|с темой|артикул\w*|отправител\w*|номер\w*\s+\d|syn-|l-)', re.I)
WORD = r"[а-яё]+"
# The permission-giving institution can stand between the cue and its preposition.
PERMISSION = rf"(?:разрешени[а-яё]*|допуск[а-яё]*|согласи(?:е|я|ю|ем|и|й|ям|ями|ях))\s+(?:{WORD}\s+){{0,2}}?(?:на|к)"
PLAN = rf"(?:заявк[а-яё]*|запрос[а-яё]*)\s+(?:{WORD}\s+){{0,2}}?на|(?:запланирован[а-яё]*|планируется|предстоит)"
CUE = re.compile(rf"\b(?:(?P<permission>{PERMISSION})|(?P<plan>{PLAN}))\s+(?P<event>{WORD})\b", re.I)
EXPLICIT = re.compile(r"\b(?:разреш\w*|допуск\w*|согласи\w*|план\w*|предсто\w*|заявк\w*|запрос\w*|можно|может|ли)\b", re.I)
CONFIRMED = re.compile(r"\b(?:выполнен\w*|заверш[её]н\w*|закончен\w*|состоял\w*|произвед[её]н\w*|исполнен\w*)\b", re.I)
CLAUSE = re.compile(r"[.!?;\n]+|,\s*(?:но|затем|после чего)\b", re.I)
NEGATED = re.compile(r"\b(?:не|если|будет|может|пока)\b", re.I)


def referenced_events(text):
    """Find the head after each modal cue; keep separately affirmed events intact."""
    matches = list(CUE.finditer(text))
    if not matches:
        return None
    groups = {"permission": set(), "plan": set()}
    for match in matches:
        head = _features(_words(match["event"]), query=True)
        groups["permission" if match["permission"] else "plan"].update(head)
    for clause in CLAUSE.split(text):
        if CONFIRMED.search(clause) and not NEGATED.search(clause):
            factual = set(_features(_words(CUE.sub(" ", clause)), query=True))
            for heads in groups.values():
                heads.difference_update(factual)
    return tuple(frozenset(group) for group in groups.values())


def query_events(query, at):
    if not COMMAND.match(query) or STRUCTURED.search(query):
        return frozenset()
    text = parse_date_scope(query, at).text
    if EXPLICIT.search(text) or BROAD.search(text) or not REPORT.search(text):
        return frozenset()
    return frozenset(_features(_words(text), query=True))


class EventRanker:
    def __init__(self, base, docs, weights):
        self.base = base
        self.weights = tuple(float(weights[k]) for k in ("permission", "plan"))
        self.events = {}
        self.text = {doc["doc_id"]: doc["doc"] for doc in docs}
        self.document_terms = lru_cache(maxsize=512)(self._document_terms)
        for row in docs:
            events = referenced_events(row["doc"])
            if events and any(events):
                self.events[row["doc_id"]] = events

    def _document_terms(self, did):
        return frozenset(_features(_words(self.text[did])))

    def evidence(self, hits, topics):
        """Cap demotions at the weakest ordinary hit independently mentioning the same event."""
        roles = {}
        active_heads = set()
        for did, _ in hits:
            if did in self.events:
                permission, plan = (topics.intersection(h) for h in self.events[did])
                if permission or plan:
                    roles[did] = permission, plan
                    active_heads.update(permission | plan)
        if not active_heads:
            return [(did, float(score), 0., 0.) for did, score in hits]
        floors = {}
        for did, score in hits:
            qualified = set().union(*roles.get(did, ()))
            for head in (self.document_terms(did) & active_heads) - qualified:
                floors[head] = min(floors.get(head, inf), float(score))
        rows = []
        for did, score in hits:
            bounds = []
            for heads in roles.get(did, ((), ())):
                bound = min((floors[h] for h in heads if h in floors), default=inf)
                bounds.append(max(0., float(score) - nextafter(bound, -inf)))
            rows.append((did, float(score), *bounds))
        return rows

    def search(self, mailbox, query, at, limit):
        if (not self.events or not any(self.weights) or limit < 1
                or not (topics := query_events(query, at))):
            return self.base.search(mailbox, query, at, limit)
        hits = self.base.search(mailbox, query, at, max(32, limit))
        evidence = self.evidence(hits, topics)
        scored = [(did, score - max(min(self.weights[0], permission), min(self.weights[1], plan)))
                  for did, score, permission, plan in evidence]
        return sorted(scored, key=lambda item: -item[1])[:limit]



def wrap_ordinary(base, docs, store):
    weights = json.loads(CONFIG.read_text())["weights"]
    return EventRanker(base, docs, weights)
