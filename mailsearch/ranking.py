"""Serve the combined experiment with indexes fitted to the supplied mailbox corpus."""
from os import getenv

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from experiments.mail_rank.models import Ranked
from experiments.mail_rank3.candidates import FieldIndex, HeaderModel, RoutedRanker
from .dates import parse_date_scope
from .retrieval import SearchEngine
from .store import CACHE, ROOT


class CharacterRanker:
    def __init__(self, base, docs, store):
        self.base = base
        self.indices = {d["doc_id"]: i for i, d in enumerate(docs)}
        texts = [((store.summaries[d["doc_id"]]["subject"] + " ") * 2 + d["doc"][:1500]) for d in docs]
        self.encoder = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=min(2,len(docs)),
                                       max_features=110000, dtype=np.float32)
        try:
            self.matrix = self.encoder.fit_transform(texts)
        except ValueError:
            self.matrix = None

    def search(self, mailbox, query, at, limit):
        hits = self.base.search(mailbox, query, at, max(limit, 32))
        if len(hits) < 2 or self.matrix is None:
            return hits[:limit]
        idx = [self.indices[did] for did, _ in hits]
        extra = (self.matrix[idx] @ self.encoder.transform([query]).T).toarray().ravel()
        ranked = [(did, score + .2 * float(value)) for (did, score), value in zip(hits, extra)]
        return sorted(ranked, key=lambda hit: -hit[1])[:limit]


def build_engine_modes(docs, store, *, prepare_paraphrase=True):
    base = SearchEngine(docs, CACHE)
    mlp = Ranked(base, store, joblib.load(ROOT / "results/mail_rank/mlp.joblib"))
    char = CharacterRanker(mlp, docs, store)
    header = None
    if len(docs) >= 64:
        try:
            header = HeaderModel(docs, store, cache_dir=CACHE)
        except ValueError:
            # An export containing only identical or empty headers has no usable SVD.
            pass
    ranked = RoutedRanker(char, FieldIndex(docs, store, parse_date_scope), header, "cluster", parse_date_scope)
    ordinary_name = getenv("MAILSEARCH_ORDINARY_MODEL", "event")
    if ordinary_name == "combined":
        ordinary = ranked
    elif ordinary_name == "event":
        from experiments.ordinary_cycle.ordinary import wrap_ordinary

        ordinary = wrap_ordinary(ranked, docs, store)
    else:
        raise ValueError(f"Unknown ordinary search model: {ordinary_name!r}")

    variant = getenv("MAILSEARCH_PARAPHRASE_MODEL")
    default_paraphrase = bool(variant) or getenv("MAILSEARCH_BILINGUAL") == "1"
    paraphrase = None
    if prepare_paraphrase or default_paraphrase:
        from experiments.mail_rank4.bilingual import BilingualSearch

        paraphrase = BilingualSearch(ranked, docs, base.dense)
        if variant and variant != "bilingual":
            from experiments.recall_cycle.variants import wrap_candidate

            paraphrase = wrap_candidate(variant, paraphrase, docs)
    return ordinary, paraphrase, paraphrase if default_paraphrase else ordinary


def build_engine(docs, store):
    """Compatibility entry point for callers that only need the configured default."""
    _, _, default = build_engine_modes(docs, store, prepare_paraphrase=False)
    return default


def paraphrase_query(query, at):
    """Allow a free-form description in explicit paraphrase mode."""
    from experiments.mail_rank4.model import COMMAND
    from experiments.recall_cycle.reranker import natural

    if COMMAND.match(query):
        return query
    command = "Найди " + query
    return command if natural(command, at) else query
