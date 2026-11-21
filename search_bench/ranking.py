"""Experimental rankers for the public mail benchmark."""

from collections import defaultdict
from datetime import date
from functools import lru_cache
from pathlib import Path
import re

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import snowballstemmer

from mailsearch.dates import DateScope, parse_date_scope


# Each entry lists whether to apply parsed date bounds and which scores to fuse.
METHODS = {
    "date-only": (True, ()),
    "bm25": (False, ("bm25",)),
    "bm25-date": (True, ("bm25",)),
    "bm25-subject-date": (True, ("bm25-subject",)),
    "char-date": (True, ("char",)),
    "lexical-date": (True, ("bm25-subject", "char")),
    "e5-date": (True, ("e5",)),
    "hybrid-date": (True, ("bm25-subject", "e5")),
    "char-e5-date": (True, ("char", "e5")),
    "fusion-date": (True, ("bm25-subject", "char", "e5")),
}

TOKEN = re.compile(r"[a-zа-я0-9]+")
SUBJECT = re.compile(r"(?im)^\s*тема\s*:\s*([^\r\n]+)")
STOPWORDS = frozenset("а без в во для до за и из к ко на над о об от по под при про с со у это".split())
QUERY_PREFIX = re.compile(r"^(?:найди(?:те)?|покажи(?:те)?|отыщи(?:те)?)\s+")
RECEIPT_SUFFIX = re.compile(r"[,\s]+(?:полученн\w+|пришедш\w+|поступивш\w+)[,.\s]*$")
STEMMER = snowballstemmer.stemmer("russian")
stem_word = lru_cache(maxsize=65_536)(STEMMER.stemWord)


def normalize(text: str) -> str:
    return text.casefold().replace("ё", "е")


def terms(text: str) -> list[str]:
    return [stem_word(word) for word in TOKEN.findall(normalize(text)) if word not in STOPWORDS]


def query_text(query: str, scope: DateScope) -> str:
    text = RECEIPT_SUFFIX.sub("", scope.text) if scope.bounded else scope.text
    text = QUERY_PREFIX.sub("", text).strip(" ,.?!")
    return text or ("" if scope.bounded else query)


class BM25:
    def __init__(self, texts: list[str]):
        vectorizer = CountVectorizer(analyzer=terms, dtype=np.float32)
        counts = vectorizer.fit_transform(texts).tocsr()
        self.vocab = vectorizer.vocabulary_
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        df = np.bincount(counts.indices, minlength=counts.shape[1])
        self.idf = np.log1p((len(texts) - df + 0.5) / (df + 0.5)).astype(np.float32)
        norm = 1.2 * (0.25 + 0.75 * lengths / max(lengths.mean(), 1))
        counts.data = 2.2 * counts.data / (counts.data + np.repeat(norm, np.diff(counts.indptr)))
        self.weights = counts

    def score(self, text: str, candidates: np.ndarray) -> np.ndarray:
        columns = [self.vocab[word] for word in dict.fromkeys(terms(text)) if word in self.vocab]
        return self.weights[candidates][:, columns] @ self.idf[columns]


class Search:
    def __init__(self, documents: list[dict], methods: list[str], cache_dir: Path):
        self.ids = [doc["doc_id"] for doc in documents]
        boxes = defaultdict(list)
        for row, doc in enumerate(documents):
            boxes[doc["synthetic_user"]].append(row)
        self.boxes = {box: np.asarray(rows, dtype=np.intp) for box, rows in boxes.items()}
        self.days = np.asarray([
            date.fromisoformat(doc["doc_dt"]).toordinal() if doc["doc_dt"] else 0
            for doc in documents
        ])
        self.scores = {score for method in methods for score in METHODS[method][1]}
        if self.scores & {"bm25", "bm25-subject"}:
            self.bm25 = BM25([doc["doc"] for doc in documents])
        if "bm25-subject" in self.scores:
            subjects = []
            for doc in documents:
                text = doc["doc"]
                match = SUBJECT.search(text[:800])
                subjects.append(match[1] if match else text[:180])
            self.subject_bm25 = BM25(subjects)
        if "char" in self.scores:
            self.char = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True,
                preprocessor=normalize, dtype=np.float32,
            )
            self.char_matrix = self.char.fit_transform(doc["doc"] for doc in documents).tocsr()
        if "e5" in self.scores:
            from mailsearch.dense import DenseIndex

            self.dense = DenseIndex(documents, cache_dir)

    def candidates(self, mailbox: str, as_of: date, scope: DateScope | None = None) -> np.ndarray:
        rows = self.boxes.get(mailbox, np.empty(0, dtype=np.intp))
        days = self.days[rows]
        available = days <= as_of.toordinal()
        if scope and scope.bounded:
            available &= days != 0
            if scope.start:
                available &= days >= scope.start.toordinal()
            if scope.end:
                available &= days <= scope.end.toordinal()
        return rows[available]

    def rank(self, method: str, raw_query: str, mailbox: str, as_of: date, k: int) -> list[str]:
        use_date, scores = METHODS[method]
        scope = parse_date_scope(raw_query, as_of)
        rows = self.candidates(mailbox, as_of, scope if use_date else None)
        text = query_text(raw_query, scope)
        if not scores or not len(rows) or not text:
            return [self.ids[row] for row in rows[:k]]

        score_vectors = []
        for scorer in scores:
            if scorer in {"bm25", "bm25-subject"}:
                values = self.bm25.score(text, rows)
                if scorer == "bm25-subject":
                    values += self.subject_bm25.score(text, rows)
            elif scorer == "char":
                values = (self.char_matrix[rows] @ self.char.transform([text]).T).toarray().ravel()
            else:
                values = self.dense.vectors[rows] @ self.dense.query(text)
            score_vectors.append(values)

        if len(score_vectors) == 1:
            combined = score_vectors[0]
        else:
            combined = np.zeros(len(rows), dtype=np.float64)
            for values in score_vectors:
                order = np.argsort(-values, kind="stable")
                ranks = np.empty(len(rows), dtype=np.intp)
                ranks[order] = np.arange(1, len(rows) + 1)
                # Give tied values equal weight, including absent lexical matches.
                starts = np.r_[True, values[order][1:] != values[order][:-1]]
                ranks[order] = np.maximum.accumulate(np.where(starts, ranks[order], 0))
                combined += 1 / (10 + ranks)
        top = np.argsort(-combined, kind="stable")[:k]
        return [self.ids[row] for row in rows[top]]
