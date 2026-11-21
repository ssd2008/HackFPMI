"""Conservative Russian topic search with lexical confirmation in English mail."""

from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
import gzip
from pathlib import Path
import re
import threading

import numpy as np
import snowballstemmer


_TABLE = Path(__file__).with_name("resources") / "ru_en.tsv.gz"
_SELECT = re.compile(r"^\s*(?:(?:найди(?:те)?|покажи(?:те)?|отыщи(?:те)?|ищу|(?:мне\s+)?нужн[аы])\b|"
                     r"(?:письма|сообщения|переписка)\s+(?:о|об|про)\b)", re.I)
# Reconsider an existing hit only for an explicit "mail about X" request.
# Named senders, project names and other descriptive retrieval requests retain
# the hybrid ranking; several English keywords are substantive lexical evidence.
_TOPIC = re.compile(
    r"^\s*(?:(?:найди(?:те)?|покажи(?:те)?|отыщи(?:те)?|ищу)\s+(?:мне\s+)?"
    r"(?:письма|сообщения|переписку)|письма|сообщения|переписка)\s+(?:о|об|про)\b", re.I,
)
_WORD = re.compile(r"[a-z]+|[а-яё]+", re.I)
_ENGLISH = re.compile("[a-z]{2,}", re.I)
_BODY = re.compile(r"^\s*(?:текст письма|body)\s*:", re.M | re.I)
_LOCAL = threading.local()
_CYRILLIC = re.compile("[а-яё]{2,}", re.I)
_BOILERPLATE = frozenset("переписка переписку переписке письма письме сообщения уведомления найди покажи ищу "
                         "пришедшие получили полученные мне нужны нужна мой моей на эту тему о об с по про или либо "
                         "упомянутых тему тематике котором которых как".split())
_UNSUPPORTED = re.compile(r"\d|\b(?:не|без|кроме|исключая)\b", re.I)
_FUNCTION = frozenset("a an the to of on for at in with or and by from".split())


@lru_cache(maxsize=1)
def dictionary():
    with gzip.open(_TABLE, "rt", encoding="utf-8") as file:
        return {ru: en.split("|") for ru, en in (line.rstrip("\n").split("\t") for line in file)}


@lru_cache(maxsize=64000)
def _en_stem(word):
    if not hasattr(_LOCAL, "stemmer"):
        _LOCAL.stemmer = snowballstemmer.stemmer("english")
    return _LOCAL.stemmer.stemWord(word)


class BilingualMail:
    def __init__(self, documents, mail_fields, stem):
        # Only messages with a real mail envelope can join this fallback. Each row's
        # language is determined from its own text, independently of other users and dates.
        self.cyrillic = {}
        self.content = {}
        self.subjects = {}
        self.postings = defaultdict(set)
        for row, subject in mail_fields.addressed_subjects.items():
            text = documents[row]["doc"]
            marker = _BODY.search(text, 0, 3600)
            body = text[marker.end():] if marker else text
            preview = subject + " " + body[:700]
            latin = len(_ENGLISH.findall(preview))
            cyrillic = len(_CYRILLIC.findall(preview))
            # A short all-English subject or note is still English mail.
            if not latin or (cyrillic and (latin < 5 or latin <= 3 * cyrillic)):
                continue
            original = {stem(word) for word in _CYRILLIC.findall((subject + " " + body).casefold().replace("ё", "е"))}
            if original:
                self.cyrillic[row] = original
            words = [_en_stem(w) for w in _ENGLISH.findall(body[:3600].lower())]
            title = [_en_stem(w) for w in _ENGLISH.findall(subject)]
            self.content[row] = " ".join(words)
            self.subjects[row] = " ".join(title)
            for word in set(words) | set(title):
                self.postings[word].add(row)
        if self.content:
            dictionary()  # Local resource, loaded once during startup.

    def ungrounded(self, query, selected, stem, stop, request):
        """Identify English hits supported only by the request wording or a stray English keyword."""
        if not len(selected) or not self.content or not _TOPIC.search(query) or _UNSUPPORTED.search(query):
            return set()
        if len({w for w in _ENGLISH.findall(query.lower()) if w not in _FUNCTION}) > 1:
            return set()
        terms = {stem(word) for word in _CYRILLIC.findall(query.casefold().replace("ё", "е"))
                 if len(word) >= 3 and word not in stop and word not in _BOILERPLATE and stem(word) not in request}
        if not terms:
            return set()
        return {int(row) for row in selected if row in self.content and terms.isdisjoint(self.cyrillic.get(row, ()))}

    def groups(self, query, stem, stop, request):
        groups = []
        alternative = False
        words = _WORD.findall(query.casefold().replace("ё", "е"))
        for word in words:
            if word in {"или", "либо"}:
                alternative = True
                continue
            key = stem(word)
            if len(word) < (2 if word.isascii() else 3) or word in _BOILERPLATE or word in stop or key in request:
                continue
            senses = [word] if word.isascii() else dictionary().get(key)
            if not senses:
                return []
            forms = [tuple(_en_stem(w) for w in _ENGLISH.findall(phrase) if w not in _FUNCTION) for phrase in senses]
            if alternative and groups:
                groups[-1].update(form for form in forms if form)
            else:
                groups.append({form for form in forms if form})
            alternative = False
        return groups

    def search(self, query, rows, dense, limit, stem, stop, request):
        if not self.content or not _SELECT.search(query) or not _CYRILLIC.search(query) or _UNSUPPORTED.search(query):
            return []
        groups = self.groups(query, stem, stop, request)
        if not groups:
            return []
        candidates = set(map(int, rows))
        patterns = []
        for alternatives in groups:
            possible = set()
            names = []
            for tokens in alternatives:
                if all(t in self.postings for t in tokens):
                    matches = candidates.intersection(self.postings[tokens[0]])
                    for word in tokens[1:]:
                        matches.intersection_update(self.postings[word])
                    possible.update(matches)
                    names.append(" ".join(tokens))
            candidates.intersection_update(possible)
            if not candidates:
                return []
            patterns.append(re.compile(r"\b(?:" + "|".join(sorted(names, key=len, reverse=True)) + r")\b"))

        eligible = np.array(sorted(candidates), dtype=np.int32)
        topic = " ".join(w for w in _WORD.findall(query.lower()) if w not in _BOILERPLATE
                         and w not in stop and stem(w) not in request)
        cosine = dense.vectors[eligible] @ dense.query(topic)
        # Translations can be polysemous. A mutually attested gloss alone is
        # insufficient: require both multilingual semantic and local textual evidence.
        results = []
        for i in np.argsort(-cosine, kind="stable")[:max(96, 4 * limit)]:
            if cosine[i] < .78:
                break
            row = int(eligible[i])
            whole_title = all(pattern.search(self.subjects[row]) for pattern in patterns)
            if whole_title or self.nearby(patterns, self.content[row]):
                results.append((row, float(cosine[i]) + .035 * whole_title))
        results.sort(key=lambda pair: -pair[1])
        return results[:limit]

    @staticmethod
    def nearby(patterns, text):
        # The normalized text contains only stemmed English words. Distinct
        # concepts in a long message need a common passage, not remote mentions.
        locations = sorted((m.start(), group) for group, p in enumerate(patterns) for m in p.finditer(text))
        last = [-1000] * len(patterns)
        for position, group in locations:
            last[group] = position
            if position - min(last) <= 96:
                return True
        return False
