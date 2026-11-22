"""Build the same optional paraphrase ranker in the service and the paired API evaluation."""

from experiments.mail_rank4.bilingual import BilingualSearch, FULL_WEIGHTS


def wrap_candidate(name, bilingual, docs):
    if name == "bilingual":
        return bilingual
    if name == "full":
        return BilingualSearch(bilingual.base, docs, bilingual.dense, FULL_WEIGHTS)
    if name == "joint":
        from .joint import JointSearch

        return JointSearch(bilingual, docs)
    if name == "interaction":
        from .interaction import InteractionSearch

        return InteractionSearch(bilingual, docs)
    from .reranker import build_candidate

    return build_candidate(name, bilingual, docs)
