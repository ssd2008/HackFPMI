"""Mailbox-scoped hybrid retrieval, independent of public queries and judgments."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
import re
import threading

import numpy as np
from scipy.sparse import csr_matrix
import snowballstemmer

from .dates import parse_date_scope
from .dense import DenseIndex


_EMAIL = r"[a-z0-9_.+%-]+@[a-z0-9.-]+\.[a-z]{2,}"
_OPAQUE = r"(?:syn|l)-[a-z0-9]+(?:-[a-z0-9]+)*"
_TOKEN = re.compile(rf"{_EMAIL}|{_OPAQUE}|\d+(?:[.,:/-]\d+)+|[а-я]+|[a-z_][a-z_\d]*|\d+")
_SUBJECT = re.compile(r"(?im)^\s*(?:тема|subject)\s*:\s*([^\r\n]+)")
_LOCAL = threading.local()


@lru_cache(maxsize=80_000)
def _stem(word: str) -> str:
    if word.isascii() or len(word) < 3:
        return word
    if not hasattr(_LOCAL, "stemmer"):
        _LOCAL.stemmer = snowballstemmer.stemmer("russian")
    return _LOCAL.stemmer.stemWord(word)


# Snowball removes endings; these common nominalizations, synonyms and aspect
# pairs need a shared feature for sparse matching. E5 still sees the original.
_FAMILIES = (
    "оплата платить оплачивать оплатить оплачено платеж уплатить",
    "успешный успех одобрен одобрено одобрение прошел проведен",
    "отказ отказано отклонен отклоненный отклонена отклонение отклонить отклоняет",
    "возврат вернуть возвращен вернет возвращать возвращение",
    "ошибка сбой неисправность неполадка",
    "завершение завершен окончание окончен закончен закончить",
    "прибытие прибыли прибыла прибывает приезд приехал приехала",
    "автомашина автомашины автомобиль машина грузовик",
    "погрузка погрузить грузить погружен погрузили погружали загружен загрузка",
    "выгрузка выгрузить выгружен выгрузили выгружать разгрузка разгрузили разгружен",
    "создание создан создавать создали",
    "покупатель покупательница",
    "доставка доставить доставлен доставлять доставили",
    "штраф оштрафовать оштрафован",
)
_ALIAS = {
    _stem(word): _stem(group.split()[0]) for group in _FAMILIES for word in group.split()
}
_STOP = {
    _stem(w) for w in (
        "и а но или на над о об обо к ко из за у под по с со от в во для при до сам "
        "этот это тот тот же уже как где что чем чей мой моей мне пожалуйста из-за "
        "он она оно они ты вы мы свой такой который какой какие когда есть надо "
        "между про по ли там то бы все всех всю по через да ну нас у меня я"
    ).split()
}
_REQUEST_WORDS = {
    _stem(w) for w in (
        "покажи покажите показать показывай найти найди найдите ищи ищу поиск "
        "письма сообщение уведомление почта входящие почтовый выведи отыщи "
        "полученные получено получение пришедшие пришли приходили получал поступившие "
        "присланные пришедшее увидеть хочу нужен нужно найдем пожалуйста"
    ).split()
}


def _words(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("ё", "е"))


def _features(words: list[str], *, query: bool = False) -> list[str]:
    result = []
    for word in words:
        if len(word) == 1 and not word.isdigit():
            continue
        stem = _stem(word)
        if stem in _STOP or (query and stem in _REQUEST_WORDS):
            continue
        result.append(_ALIAS.get(stem, stem).replace(",", "."))
    return result


class LexicalIndex:
    """Precompute subject-weighted BM25 term frequencies in a sparse matrix."""

    def __init__(self, documents: list[dict]) -> None:
        self.vocabulary: dict[str, int] = {}
        columns, counts, pointers = array("i"), array("f"), array("i", [0])
        lengths = []
        for doc in documents:
            body = _features(_words(doc["doc"]))
            lengths.append(len(body))
            weights = Counter(body)
            subject = _SUBJECT.search(doc["doc"])
            if subject:
                for term in _features(_words(subject[1][:500])):
                    weights[term] += 2
            for term, count in weights.items():
                index = self.vocabulary.setdefault(term, len(self.vocabulary))
                columns.append(index)
                counts.append(count)
            pointers.append(len(counts))

        size = len(documents)
        lengths = np.asarray(lengths, dtype=np.float32)
        offsets = np.asarray(pointers, dtype=np.int32)
        indices = np.asarray(columns, dtype=np.int32)
        values = np.asarray(counts, dtype=np.float32)
        mean_length = max(1, lengths.mean()) if size else 1
        norms = 1.15 * (0.65 + 0.35 * lengths / mean_length)
        values /= values + np.repeat(norms, np.diff(offsets))
        doc_frequency = np.bincount(indices, minlength=len(self.vocabulary))
        self.idf = np.log1p((size - doc_frequency + 0.5) / (doc_frequency + 0.5))
        self.weights = csr_matrix(
            (values, indices, offsets), shape=(size, len(self.vocabulary)), dtype=np.float32
        )
        self.weights.sort_indices()

    def score(self, features: list[str], candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Repeating a term in a conversational query should not multiply its weight.
        terms = set(features)
        cols = [self.vocabulary[t] for t in sorted(terms) if t in self.vocabulary]
        if not cols:
            zeros = np.zeros(len(candidates), dtype=np.float32)
            return zeros, zeros
        weights = self.idf[cols].astype(np.float32)
        weights /= weights.sum()
        hits = self.weights[candidates][:, cols]
        coverage = np.asarray((hits > 0) @ weights).ravel()
        frequency = np.asarray(hits @ weights).ravel()
        return 0.48 * coverage + 0.52 * frequency, coverage


class SearchEngine:
    def __init__(self, documents: list[dict], cache_dir: Path):
        self.ids = [doc["doc_id"] for doc in documents]
        self.by_id = {doc_id.lower(): i for i, doc_id in enumerate(self.ids)}
        self.dates = np.array(
            [date.fromisoformat(d["doc_dt"]).toordinal() if d["doc_dt"] else 0 for d in documents],
            dtype=np.int32,
        )
        boxes: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(documents):
            boxes[doc["synthetic_user"]].append(i)
        self.boxes = {key: np.asarray(rows, dtype=np.int32) for key, rows in boxes.items()}
        self.lexical = LexicalIndex(documents)
        self.dense = DenseIndex(documents, Path(cache_dir))

    def search(
        self, synthetic_user: str, query: str, query_date: date, limit: int = 10
    ) -> list[tuple[str, float]]:
        box = self.boxes.get(synthetic_user)
        if box is None or limit < 1 or not query.strip():
            return []
        scope = parse_date_scope(query, query_date)
        dates = self.dates[box]
        available = dates <= query_date.toordinal()
        if scope.bounded:
            available &= dates > 0
            if scope.start:
                available &= dates >= scope.start.toordinal()
            if scope.end:
                available &= dates <= scope.end.toordinal()
        rows = box[available]
        if not len(rows):
            return []
        words = _words(scope.text)
        features = _features(words, query=True)
        if not features:
            if not words and not scope.bounded:
                return []
            order = np.argsort(-self.dates[rows], kind="stable")[:limit]
            return [(self.ids[rows[i]], 1.0) for i in order]

        # Opaque IDs, addresses and long numbers are strong literal constraints.
        identifiers = [
            word.replace(",", ".") for word in words
            if "@" in word or word.startswith(("syn-", "l-")) or (word.isdigit() and len(word) >= 5)
        ]
        for token in dict.fromkeys(identifiers):
            if token.startswith("l-"):
                rows = rows[rows == self.by_id.get(token, -1)]
            elif token in self.lexical.vocabulary:
                hits = self.lexical.weights[rows, self.lexical.vocabulary[token]].toarray().ravel()
                rows = rows[hits > 0]
            else:
                return []
            if not len(rows):
                return []
        if identifiers and len(rows) == 1:
            return [(self.ids[rows[0]], 1.0)]

        terms = " ".join(w for w in words if _stem(w) not in _REQUEST_WORDS)
        vector = self.dense.query(terms)
        cosine = self.dense.vectors[rows] @ vector
        lexical, coverage = self.lexical.score(features, rows)
        semantic = np.clip((cosine - 0.74) / 0.17, 0, 1.25)
        scores = 0.55 * lexical + 0.45 * semantic
        # Keep paraphrases without shared words when their semantic signal is strong.
        plausible = ((coverage > 0) & (scores >= 0.24)) | (cosine >= 0.835)
        if identifiers:
            plausible[:] = True
        order = np.flatnonzero(plausible)
        order = order[np.argsort(-scores[order], kind="stable")[:limit]]
        return [(self.ids[rows[i]], float(scores[i])) for i in order]
