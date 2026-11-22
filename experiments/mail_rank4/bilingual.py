"""Learn lexical translations from aligned English/Russian MS MARCO train queries."""

from collections import Counter
from functools import lru_cache
import gzip
import threading
import json
import re

import numpy as np
import snowballstemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from mailsearch.dates import parse_date_scope
from mailsearch.retrieval import _CODE_CANDIDATE, _article
from mailsearch.store import ROOT, describe
from .model import COMMAND, STRUCTURED, content

WEIGHTS = ROOT / "results/mail_rank4/translation.json.gz"
WORD = re.compile(r"[^\W_]+", re.UNICODE)
RU = re.compile("[а-яё]", re.I)
IGNORE = set(
    "где как найти найди покажи мне это что для от кто меня было чтобы письма письмо письме сообщение присылали было про the a is and or to for how which of on in be an what when where at with from my i me are by as it do you your".split()
)


_thread = threading.local()


@lru_cache(maxsize=80000)
def stem(word):
    # Snowball mutates its buffers. Training and requests can run in multiple threads.
    lang = "russian" if RU.search(word) else "english"
    if not hasattr(_thread, lang):
        setattr(_thread, lang, snowballstemmer.stemmer(lang))
    return getattr(_thread, lang).stemWord(word)


def words(text):
    return [
        stem(w)
        for w in WORD.findall(text.lower().replace("ё", "е"))
        if len(w) > 1 and w not in IGNORE
    ]


FULL_WEIGHTS = WEIGHTS.with_name("translation-full.json.gz")


def train(full=False):
    data = ROOT / "datasets/local"
    en = dict(
        row.rstrip("\n").split("\t", 1)
        for row in (ROOT / ".cache/defense/msmarco-en-train.tsv").open()
    )
    english, russian = [], []
    with next((data / "mmarco-ru/data/google/queries/train").glob("*.tsv")).open() as f:
        for line in f:
            key, query = line.rstrip("\n").split("\t", 1)
            if key in en and (full or int(key) % 2 == 0):
                english.append(en[key])
                russian.append(query)
    matrices, names = [], []
    for texts in (russian, english):
        enc = CountVectorizer(
            analyzer=words, min_df=3, max_df=0.12, binary=True, dtype=np.float32
        )
        matrices.append(enc.fit_transform(texts))
        names.append(enc.get_feature_names_out())
    ru, en = matrices
    df_ru = np.asarray(ru.sum(0)).ravel()
    df_en = np.asarray(en.sum(0)).ravel()
    co = (ru.T @ en).tocsr()
    translations = {}
    for i, token in enumerate(names[0]):
        if not RU.search(token):
            continue
        start, end = co.indptr[i : i + 2]
        ids = co.indices[start:end]
        counts = co.data[start:end]
        scores = counts**2 / ((df_ru[i] + 4) * (df_en[ids] + 4))
        top = np.argsort(-scores)[:3]
        translations[token] = [
            [str(names[1][ids[j]]), round(float(scores[j] ** 0.5), 4)]
            for j in top
            if scores[j] > 0.012 and counts[j] >= 3
        ]
    data = json.dumps(
        dict(pairs=len(russian), terms=translations), ensure_ascii=False
    ).encode()
    target = FULL_WEIGHTS if full else WEIGHTS
    target.write_bytes(gzip.compress(data, mtime=0))
    print(len(russian), "pairs", len(translations), "terms", flush=True)


class BilingualSearch:
    def __init__(self, base, docs, dense, weights_path=WEIGHTS):
        self.base = base
        self.dense = dense
        self.terms = json.loads(gzip.decompress(weights_path.read_bytes()))["terms"]
        self.ids = np.array([d["doc_id"] for d in docs])
        self.boxes = np.array([d["synthetic_user"] for d in docs])
        self.dates = np.array([d["doc_dt"] or "" for d in docs])
        self.encoder = TfidfVectorizer(
            analyzer=words, sublinear_tf=True, dtype=np.float32
        )
        text = []
        for d in docs:
            s = describe(d)
            sender = (s["sender"] or "").split("@")[-1]
            text.append((s["subject"] + " ") * 3 + sender + " " + s["snippet"])
        try:
            self.matrix = self.encoder.fit_transform(text)
        except ValueError:
            self.matrix = None

    def search(self, mailbox, query, at, limit):
        text = content(query)
        if (
            self.matrix is None
            or not COMMAND.match(query)
            or STRUCTURED.search(query)
            or parse_date_scope(query, at).bounded
            or len(text.split()) < 3
            or any(
                not m.lastgroup and _article(m[0])
                for m in _CODE_CANDIDATE.finditer(query)
            )
        ):
            return self.base.search(mailbox, query, at, limit)
        baseline = self.base.search(mailbox, query, at, max(limit, 40))
        tokens = words(text)
        translations = Counter()
        for t in set(tokens):
            for en, weight in self.terms.get(t, []):
                translations[en] = max(translations[en], weight)
        for token in tokens:
            translations[token] = 1
        vocab = self.encoder.vocabulary_
        q = np.zeros(len(vocab), dtype=np.float32)
        for t, v in translations.items():
            if t in vocab:
                q[vocab[t]] = v * self.encoder.idf_[vocab[t]]
        idx = np.flatnonzero((self.boxes == mailbox) & (self.dates <= at.isoformat()))
        if not len(idx):
            return []
        lex = self.matrix[idx] @ q
        semantic = self.dense.vectors[idx] @ self.dense.query(text)
        if not baseline and (not np.any(lex) or semantic.max() < 0.80):
            return []
        scores = {did: 1 / (4 + pos) for pos, (did, _) in enumerate(baseline)}
        # Rank on both signals so that a single translated word is insufficient.
        mixed = lex / max(1e-6, lex.max()) + np.clip((semantic - 0.73) / 0.16, 0, 1.5)
        for pos, i in enumerate(np.argsort(-mixed, kind="stable")[: max(40, limit)]):
            if lex[i] > 0:
                did = str(self.ids[idx[i]])
                scores[did] = scores.get(did, 0) + 1.35 / (4 + pos)
        return sorted(scores.items(), key=lambda p: -p[1])[:limit]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true")
    train(parser.parse_args().full)
