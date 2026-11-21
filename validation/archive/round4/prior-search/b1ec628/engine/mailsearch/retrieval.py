"""Mailbox-scoped hybrid retrieval, independent of public queries and judgments."""

from __future__ import annotations

from array import array
from collections import Counter, defaultdict
from dataclasses import dataclass
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
from .scoring import semantic_lexical_scale
from .spelling import CorpusSpelling
from .fields import MailFields, only_field_request
from .bilingual import BilingualMail


_EMAIL = r"[a-z0-9_.+%-]+@[a-z0-9.-]+\.[a-z]{2,}"
_OPAQUE = r"(?:syn|l)-[a-z0-9]+(?:-[a-z0-9]+)*"
_WHOLE_OPAQUE = re.compile(rf"(?<![\w-]){_OPAQUE}(?![\w-])", re.I)
_TOKEN = re.compile(rf"{_EMAIL}|{_OPAQUE}|\d+(?:[.,:/-]\d+)+|[а-я]+|[a-z_][a-z_\d]*|\d+")
# Match complete candidate references before interpreting dates. At least four
# digits avoids treating everyday names such as X5 or B2B as unique references.
_CODE_CANDIDATE = re.compile(
    rf"(?P<keep>https?://[^\s<>]+|{_EMAIL}|syn-[a-z0-9]+(?:-[a-z0-9]+)*|l-[0-9a-f]{{16,}}\b)|"
    r"(?<![\w@/-])(?=(?:[a-zа-яё_/-]*[0-9]){4})"
    r"[a-zа-яё0-9]+(?:[-_/][a-zа-яё0-9]+)*(?![\w@_/-])", re.I
)
_CALENDAR_NUMBER = re.compile(
    r"(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|(?:19|20)\d\d[-/](?:19|20)\d\d)"
)
_UNIT = r"(?:руб(?:лей|ля)?|rub|rur|eur|usd|год[а-я]*|кг|мл|км|мм|см|шт)"
_MEASUREMENT = re.compile(rf"(?:{_UNIT}[-_]?\d+|\d+[-_]?{_UNIT})")
_SUBJECT = re.compile(r"(?im)^\s*(?:тема|subject)\s*:\s*([^\r\n]+)")
_CLAUSE = re.compile(r"[\r\n]+|(?<=[.!?;])(?:\s+|(?=[А-ЯЁ]))")
_PLAN_NOUN = r"(?:разрешени[а-я]*|заявк[а-я]*|запрос[а-я]*|подготовк[а-я]*|счет[а-я]*|план(?:а|у|ы|ом|ов)?)"
_PLANNED_ACTION = re.compile(
    rf"\b{_PLAN_NOUN}(?:\s+[а-я-]+){{0,2}}?\s+(?:на|к|для)\s+"
    r"(?:(?:предстоящ|будущ|планов|срочн)[а-я]*\s+)?"
    r"(?P<action>[а-я-]+(?:ни[еяю]|ци[юия]|к[ауеи]|т[ау])|возврат|ремонт)\b"
)
_PLAN_REQUEST = re.compile(rf"\b(?:{_PLAN_NOUN}|разреш[а-я]*|запланир[а-я]*|предстоящ[а-я]*)\b")
_QUOTED = re.compile(r'«([^«»]*)»|"([^"]*)"')
_NAME_PREPOSITION = re.compile(r"\b(?:об?|про|к|с|из|для|у|по|над|перед|от)\s*$", re.I)
_VERBATIM = re.compile(r"\b(?:дослов[а-я]*|букваль[а-я]*|точн[а-я]*\s+(?:фраз[а-я]*|цитат[а-я]*|текст[а-я]*))\b", re.I)
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
    "отказ отказано отклонен отклоненный отклонена отклонение отклонить отклоняет отклонили "
    "неудача неудачный неуспешный безуспешный отвергнут declined failed",
    "возврат вернуть возвращен вернет возвращать возвращение возвращали",
    "ошибка сбой неисправность неполадка поломка сломался",
    "завершение завершен окончание окончен закончен закончить закончился выполнен исполнен "
    "состоялся завершили прошел прошла прошли успех успешный одобрен одобрено одобрение проведен",
    "прибытие прибыли прибыла прибывает приезд приехал приехала",
    "швартовка пришвартован пришвартовался ошвартован швартоваться причаливание причалил причалить",
    "автомашина автомашины автомобиль машина грузовик",
    "погрузка погрузить погружен погрузили погружали загружен загрузка",
    "выгрузка выгрузить выгружен выгрузили выгружать разгрузка разгрузили разгружен",
    "отгрузка отгрузил отгружен отгрузили отгрузить отгружают",
    "создание создан создавать создали оформили оформить оформлен оформление",
    "визит автовизит заезд въезд",
    "досмотр осмотр досмотрели проверка проверять проверили",
    "приемка принятый приняли принята принято принят",
    "чек квитанция квиток",
    "кассовый касса фискальный",
    "электронный цифровой",
    "коммунальный коммуналка жку жкх",
    "покупатель покупательница",
    "доставка доставить доставлен доставлять доставили",
    "штраф оштрафовать оштрафован",
)
_ALIAS = {
    _stem(word): _stem(group.split()[0]) for group in _FAMILIES for word in group.split()
}
# These words in a message also identify the objects involved in a port call.
_PORT_CONTEXT = {
    _stem("судозаход"): (_stem("судно"),),
    _stem("швартовка"): (_stem("причал"),),
}
_STOP = frozenset("""
    и а но или на над о об обо к ко из за у под по с со от в во для при до сам
    этот это тот же уже как где что чем чей мой моей мне пожалуйста
    он она оно они ты вы мы свой такой который какой какие когда есть надо
    между про ли там то бы все всех всю через да ну нас меня я
    том тех тем этим эта той тому чтобы вот только тоже каких
    моих мою мои моего этих этого этом такая такое такие хотеть хочет хотят
""".split())
_NEGATIVE = _stem("отказ")
_NEGATABLE = {
    _stem(w) for w in (
        "прошел прошла прошли проходит проходить удался удалось получилось состоялся "
        "вышло прошедший обработан проведен выполнен одобрен зачислен списан "
        "успешный подтвержден завершен исполнен поступил хватило зачислили"
    ).split()
}
# Only explicit decisions count as facts. Infinitives and imperatives such as
# "одобрить" and "подтвердите" describe possible actions.
_APPROVED = (
    r"прош(?:ел|ла|ло|ли)|состоял(?:ся|ась|ось|ись)|одобрен[аоы]?|одобрил[аио]?|"
    r"подтвержден[аоы]?|подтвердил[аио]?|согласован[аоы]?|выполнен[аоы]?|"
    r"исполнен[аоы]?|завершен[аоы]?|проведен[аоы]?|успешн(?:о|ая|ый|ые|ой|ых|ую|ое)"
)
_DECLINED = (
    r"отклон(?:ен(?:[аоы]|н[а-я]+)?|ил[аио]?|ение)|отказ(?:ано|ал[аио]?|[аеуы]?)|"
    r"неудачн(?:о|ая|ый|ые)|безуспешно|declined|failed"
)
_NEGATION = r"не\s+(?:(?:был[аои]?|оказал[а-я]+)\s+)?"
_DECISION = re.compile(
    rf"\b(?:(?P<unclear>{_NEGATION}(?:{_DECLINED}))|"
    rf"(?P<no>{_NEGATION}(?:{_APPROVED}|удалось|получилось)|{_DECLINED})|"
    rf"(?P<yes>{_APPROVED}))\b"
)
_CONDITIONAL = re.compile(
    r"\b(?:если|бы|ли|мож(?:ет|но|ете)|возмож[а-я]*|вероятно|вдруг|чтобы|"
    r"при условии|при отказе|будет|будут)\b|\?"
)
# A completed order cancellation and a warning about one share almost every
# content word. This check is deliberately limited to an explicit order reference.
_CANCEL_ROOT = re.compile(r"\b(?:отмен|аннулир)")
_CANCEL_PAST = r"(?:отмен(?:ен(?:н[а-я]+|[аоы])?|ил[аио]?)|аннулирова(?:н(?:н[а-я]+|[аоы])?|л[аио]?))"
_CANCEL_NOUN = r"(?:отмен[аеуы]|аннулировани[еяю])"
_ORDER_MOD = r"(?:(?:уже|все[ -]таки|действительно|фактически|окончательно|был[аи]?|наш|ваш|этот|мой)\s+)*"
_ORDER_LINK = r"(?:\s*(?:номер|№|no\.?)?\s*[:#–-]?\s*<ref>)?"
_COMPLETED_ORDER = re.compile(
    rf"\bзаказ(?:ы)?\b{_ORDER_LINK}\s+{_ORDER_MOD}{_CANCEL_PAST}\b|"
    rf"\b{_CANCEL_PAST}\s+{_ORDER_MOD}(?:заказ(?:ы)?|его)\b|"
    rf"\bего\s+{_ORDER_MOD}{_CANCEL_PAST}\b"
)
_CANCEL_ORDER_NOUN = re.compile(rf"\b{_CANCEL_NOUN}\s+(?:(?:этого|моего|нашего|вашего)\s+)?заказ(?:а|ов)\b")
_EVIDENCE_REQUEST = re.compile(
    r"\b(?:подтвержд[а-я]*|фактическ[а-я]*|состоявш[а-я]*|совершивш[а-я]*|"
    r"(?:письм[оае]|уведомлени[ея]|сообщени[ея]|извещени[ея])\s+об)\b"
)
_ORDER_MENTION = re.compile(
    r"\bзаказ(?:а|у|е|ом|ы|ов)?\b\s*(?:(?:с\s+)?номер(?:ом)?|№|no\.?)?\s*[:#–-]?\s*"
    r"(?P<id><ref>|(?=[\w/-]*\d)[\w/-]{4,64})"
)
_WARNING = re.compile(
    r"\b(?:если|иначе|чтобы|придется|мож(?:ет|ем|но)|могут|буд(?:ет|ут|ем)|"
    r"возможн[а-я]*|риск[а-я]*|при\s+(?:не[а-я]*|отсутствии)|"
    r"(?:отмен|аннулиру)(?:ят|им|ит|ится|ятся|ют|ем|ет|ется|ются))\b"
)
_UNASSERTED = re.compile(r"\bли\b|\bне\s+(?:был[аои]?\s+)?(?:отмен|аннулир|подтвержд)|\?")
_CANCEL_VERB = r"(?:отмен[а-я]*|аннулир[а-я]*)"
_WARNING_ORDER = re.compile(
    rf"\b(?:заказ(?:ы)?\b{_ORDER_LINK}|его|он)\s+"
    rf"(?:(?:буд[а-я]*|мож[а-я]*|могут|быть|автоматически|придется|нам|мы|тогда)\s+){{0,5}}{_CANCEL_VERB}\b|"
    rf"\b{_CANCEL_VERB}\s+(?:(?:этот|ваш|наш|мой)\s+)?(?:заказ(?:ы)?|его)\b"
)
_CANCEL_CLAUSE = re.compile(r"[\r\n,;]+|(?<=[.!?])\s+|\b(?:но|однако|затем)\b")
_CONFIRM_CANCEL = re.compile(
    r"\b(?:отмен[ау]\s+(?:уже\s+|была\s+)?подтвер[жд][а-я]*|подтвер[жд][а-я]*\s+отмен[ауые])\b"
)
_SELECT_REQUEST = re.compile(r"^\s*(?:найди(?:те)?|покажи(?:те)?|какие|каких|перечисли)\b")
_REQUEST_WORDS = {
    _stem(w) for w in (
        "покажи покажите показать показывай найти найди найдите ищи ищу поиск "
        "письма сообщение уведомление почта входящие почтовый выведи отыщи "
        "полученные получено получение пришедшие пришли приходили получал поступившие "
        "присланные пришедшее увидеть хочу нужен нужно найдем пожалуйста"
    ).split()
}


def _words(text: str) -> list[str]:
    return _TOKEN.findall(text.lower().replace("ё", "е").replace("₽", " руб "))


def _literal_text(text: str) -> str:
    return " ".join(text.casefold().replace("ё", "е").split())


@dataclass(frozen=True)
class QuotedPhrase:
    literal: str
    exact: re.Pattern
    title_stems: tuple[str, ...]
    inflected: re.Pattern | None

    def matches(self, text: str) -> bool:
        if self.literal in text and self.exact.search(text):
            return True
        if self.inflected:
            for match in self.inflected.finditer(text):
                if tuple(_stem(word) for word in match[0].split()) == self.title_stems:
                    return True
        return False


def _extract_phrases(query: str) -> tuple[str, list[QuotedPhrase]]:
    phrases = []
    verbatim = _VERBATIM.search(query)

    def extract(match):
        original = match[match.lastindex]
        phrase = _literal_text(original)
        if phrase:
            # Preserve punctuation literally. Only alphanumeric/underscore
            # endpoints require a boundary to prevent a match inside a word.
            left = r"(?<!\w)" if re.match(r"\w", phrase[0]) else ""
            right = r"(?!\w)" if re.match(r"\w", phrase[-1]) else ""
            stems, inflected = (), None
            if (not verbatim and original[:1].isupper() and _NAME_PREPOSITION.search(query[:match.start()])
                    and re.fullmatch(r"[а-я]+(?: [а-я]+){1,7}", phrase)):
                # A proper name in running Russian prose follows its governing
                # preposition. Keep word order and check every complete stem.
                stems = tuple(_stem(word) for word in phrase.split())
                pattern = r"\s+".join(re.escape(stem) + "[а-я]*" for stem in stems)
                inflected = re.compile(r"(?<!\w)" + pattern + r"(?!\w)")
            phrases.append(QuotedPhrase(phrase, re.compile(left + re.escape(phrase) + right), stems, inflected))
        return " "

    return _QUOTED.sub(extract, query), phrases


def _article(token: str) -> bool:
    if len(token) > 64:
        return False
    if _MEASUREMENT.fullmatch(token) or _CALENDAR_NUMBER.fullmatch(token):
        return False
    if any(c.isalpha() for c in token):
        return True
    # Numeric order numbers often look like 405769-5437-5. Decimal money,
    # ordinary integers and common written dates have their own semantics.
    return len(token) >= 8 and ("-" in token or "/" in token)


def _features(words: list[str], *, query: bool = False) -> list[str]:
    result = []
    negation = False
    for word in words:
        if len(word) == 1 and not word.isdigit():
            continue
        stem = _stem(word)
        if word == "не":
            negation = True
            continue
        if negation and word in {"был", "было", "были", "была", "оказался"}:
            continue
        if negation and stem in _NEGATABLE:
            result.append(_NEGATIVE)
            negation = False
            continue
        if negation and _ALIAS.get(stem) == _NEGATIVE:
            negation = False
            continue
        negation = False
        if word in _STOP or (query and stem in _REQUEST_WORDS):
            continue
        term = _ALIAS.get(stem, stem).replace(",", ".")
        result.append(term)
        if not query and term in _PORT_CONTEXT:
            result.extend(_PORT_CONTEXT[term])
    return result


def _warning_without_confirmation(text: str, reference: re.Pattern) -> bool:
    """Remove only an identified order's warning, preserving ambiguous or mixed mail."""
    text = reference.sub("<ref>", text.lower().replace("ё", "е"))
    active = False
    warned = False
    for clause in _CANCEL_CLAUSE.split(text):
        refs = {match["id"] for match in _ORDER_MENTION.finditer(clause)}
        if refs:
            if "<ref>" in refs and len(refs) > 1:
                return False  # Two orders in one clause have ambiguous ownership.
            active = "<ref>" in refs
        if not active or not _CANCEL_ROOT.search(clause):
            continue
        if _WARNING.search(clause) and (_WARNING_ORDER.search(clause) or _CANCEL_ORDER_NOUN.search(clause)):
            warned = True
            continue
        if "?" in clause and _COMPLETED_ORDER.search(clause):
            return False
        if (_COMPLETED_ORDER.search(clause) or _CANCEL_ORDER_NOUN.search(clause) or
                _CONFIRM_CANCEL.search(clause)) and not _UNASSERTED.search(clause):
            return False
    return warned


def _completed_order_reference(query: str, identifiers: list[str]) -> re.Pattern | None:
    if len(identifiers) != 1 or "@" in identifiers[0] or not _CANCEL_ROOT.search(query):
        return None
    identifier = identifiers[0].removeprefix("code:")
    reference = re.compile(rf"(?<![\w-]){re.escape(identifier)}(?![\w-])", re.I)
    marked = reference.sub("<ref>", query)
    if _WARNING.search(marked) or re.search(r"\b(?:ли|риск[а-я]*|предупреж[а-я]+)\b", marked):
        return None
    if "?" in marked and not re.search(r"\b(?:найди|покажи|где|в каком письме)\b", marked):
        return None
    plain = marked.replace("?", "")
    if _UNASSERTED.search(plain):
        return None
    if _COMPLETED_ORDER.search(plain) or (_CANCEL_ORDER_NOUN.search(plain) and _EVIDENCE_REQUEST.search(plain)):
        return reference
    return None


def _decisions(clauses: list[str]) -> tuple[int, set[str]]:
    """Return a single asserted polarity and the words around that assertion."""
    signs = set()
    context = set()
    for clause in clauses:
        text = clause.lower().replace("ё", "е")
        if _CONDITIONAL.search(text):
            continue
        for match in _DECISION.finditer(text):
            if match.lastgroup == "unclear":
                continue
            signs.add(1 if match.lastgroup == "yes" else -1)
            if len(signs) > 1:
                return 0, set()
            before = _words(text[max(0, match.start() - 100):match.start()])[-8:]
            after = _words(text[match.end():match.end() + 100])[:8]
            context.update(_features(before + after, query=True))
    return (next(iter(signs)), context) if signs else (0, set())


_GENERIC_CONTEXT = set(_features(_words(
    "номер статус результат информация дата время описание тема текст итог раньше теперь почему"
)))


class LexicalIndex:
    """Precompute subject-weighted BM25 term frequencies in a sparse matrix."""

    def __init__(self, documents: list[dict]) -> None:
        self.vocabulary: dict[str, int] = {}
        self.sequences: list[array] = []
        self.decisions: list[tuple[int, set[str]]] = []
        planned_rows = defaultdict(list)
        columns, counts, pointers = array("i"), array("f"), array("i", [0])
        lengths = []
        for row, doc in enumerate(documents):
            passages = _CLAUSE.split(doc["doc"])
            self.decisions.append(_decisions(passages))
            clauses = [_features(_words(clause)) for clause in passages]
            weights = Counter(term for clause in clauses for term in clause)
            # A permit or order for an action can be useful context, but it does
            # not report that the action happened. Keep a document at full weight
            # if the same event is also named outside the prospective expression.
            prospective = Counter(
                term for match in _PLANNED_ACTION.finditer(doc["doc"].lower().replace("ё", "е"))
                for term in _features([match["action"]])
            )
            for term, count in prospective.items():
                if weights[term] == count:
                    planned_rows[term].append(row)
            lengths.append(sum(weights.values()))
            subject = _SUBJECT.search(doc["doc"])
            if subject:
                for term in _features(_words(subject[1][:500])):
                    weights[term] += 2
            for match in _CODE_CANDIDATE.finditer(doc["doc"]):
                token = match[0].lower().replace("ё", "е")
                if not match.lastgroup and _article(token):
                    weights["code:" + token] = 1
            for term, count in weights.items():
                index = self.vocabulary.setdefault(term, len(self.vocabulary))
                columns.append(index)
                counts.append(count)
            pointers.append(len(counts))
            sequence = array("i")
            for clause in clauses:
                sequence.extend(self.vocabulary[term] for term in clause)
                sequence.append(-1)
            self.sequences.append(sequence)

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
        sparse = csr_matrix(
            (values, indices, offsets), shape=(size, len(self.vocabulary)), dtype=np.float32
        )
        # One sorted posting list per term. A short query should not copy every
        # term from every visible letter just to look up its own terms.
        self.postings = sparse.tocsc()
        self.postings.sort_indices()
        self.planned = {term: np.asarray(rows, dtype=np.int32) for term, rows in planned_rows.items()}

    def posting(self, column: int) -> tuple[np.ndarray, np.ndarray]:
        start, end = self.postings.indptr[column:column + 2]
        return self.postings.indices[start:end], self.postings.data[start:end]

    @staticmethod
    def _locate(posted: np.ndarray, candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Every vocabulary entry occurs in at least one document.
        slots = np.minimum(posted.searchsorted(candidates), len(posted) - 1)
        return slots, posted[slots] == candidates

    def contains(self, column: int, candidates: np.ndarray) -> np.ndarray:
        posted, _ = self.posting(column)
        return self._locate(posted, candidates)[1]

    def rank_event_evidence(self, query: str, terms: set[str], rows: np.ndarray, scores: np.ndarray) -> None:
        planned_terms = terms.intersection(self.planned)
        if not planned_terms or _PLAN_REQUEST.search(query):
            return
        total = sum(self.idf[self.vocabulary[t]] for t in terms if t in self.vocabulary)
        for term in sorted(planned_terms):
            _, present = self._locate(self.planned[term], rows)
            scores[present] -= 0.28 * self.idf[self.vocabulary[term]] / total

    def score(self, features: list[str], candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Repeating a term in a conversational query should not multiply its weight.
        terms = set(features)
        cols = [self.vocabulary[t] for t in sorted(terms) if t in self.vocabulary]
        if not cols:
            zeros = np.zeros(len(candidates), dtype=np.float32)
            return zeros, zeros
        weights = self.idf[cols].astype(np.float32)
        # Words outside the corpus are still part of the request. Renormalizing
        # over known words alone turns "order for [unknown item]" into "order".
        weights *= (len(cols) / len(terms)) / weights.sum()
        # For a large scope scatter once per term, then select the requested
        # rows. For a small scope look up just those rows in each posting.
        broad = len(candidates) * 8 >= len(self.sequences)
        size = len(self.sequences) if broad else len(candidates)
        coverage = np.zeros(size, dtype=np.float32)
        frequency = np.zeros(size, dtype=np.float32)
        for column, weight in sorted(zip(cols, weights)):
            posted, strengths = self.posting(column)
            if broad:
                positions = posted
            else:
                slots, present = self._locate(posted, candidates)
                positions = np.flatnonzero(present)
                strengths = strengths[slots[positions]]
            coverage[positions] += weight
            frequency[positions] += strengths * weight
        if broad:
            coverage, frequency = coverage[candidates], frequency[candidates]
        return 0.48 * coverage + 0.52 * frequency, coverage

    def proximity(self, features: list[str], candidates: np.ndarray) -> np.ndarray:
        """Reward distinct query terms supported by one short passage, in either order."""
        cols = {self.vocabulary[t] for t in features if t in self.vocabulary}
        output = np.zeros(len(candidates), dtype=np.float32)
        if len(cols) < 2:
            return output
        total = sum(self.idf[col] for col in cols)
        weights = {col: float(self.idf[col] / total) for col in cols}
        max_span = 10
        for i, row in enumerate(candidates):
            # A gap larger than the window separates the subject, body, and sentences.
            position = 0
            matches = []
            for token in self.sequences[row]:
                if token in weights:
                    matches.append((position, token))
                position += max_span + 1 if token == -1 else 1
            best = 0.0
            for anchor, (start, first) in enumerate(matches):
                seen = {first}
                coverage = weights[first]
                for j in range(anchor + 1, len(matches)):
                    end, term = matches[j]
                    if end - start > max_span:
                        break
                    if term in seen:
                        continue
                    seen.add(term)
                    coverage += weights[term]
                    # One unrelated occurrence elsewhere in the message cannot
                    # complete a phrase. Shorter spans provide better evidence.
                    extra_words = end - start + 1 - len(seen)
                    fraction = (len(seen) - 1) / (len(cols) - 1)
                    best = max(best, coverage * fraction ** 0.5 / (1 + extra_words / 5))
            output[i] = best
        return output

    def literal_evidence(self, features: set[str], candidates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Absolute evidence, independent of how many unmatched words the user adds."""
        cols = [self.vocabulary[t] for t in sorted(features) if t in self.vocabulary]
        if not cols:
            empty = np.zeros(len(candidates), dtype=bool)
            return empty, empty
        present = self.postings[:, cols].tocsr()[candidates] > 0
        idf = self.idf[cols]
        # Subtract the surprisal of a term that appears in a quarter of the
        # corpus: matching only common words like "order" and "amount" is weak.
        evidence = present @ np.maximum(idf - np.log(4), 0)
        corroborated = (present.getnnz(axis=1) >= 2) & (evidence >= np.log(100))
        rare = np.asarray(present @ (idf >= np.log(1_000)).astype(np.int8)).ravel() > 0
        return corroborated, rare


class SearchEngine:
    def __init__(self, documents: list[dict], cache_dir: Path):
        self.ids = [doc["doc_id"] for doc in documents]
        self.by_id = {doc_id: i for i, doc_id in enumerate(self.ids)}
        self.documents = documents
        opaque_rows = defaultdict(list)
        for i, document in enumerate(documents):
            for token in {m[0].lower() for m in _WHOLE_OPAQUE.finditer(document["doc"])}:
                opaque_rows[token].append(i)
        self.opaque_rows = {token: np.asarray(rows, dtype=np.int32) for token, rows in opaque_rows.items()}
        self.fields = MailFields(documents)
        self.bilingual = BilingualMail(documents, self.fields, _stem)
        # The public format embeds its subject in doc. A separate subject field
        # in an imported mailbox is searchable without inventing adjacent words.
        self.literal_texts = [
            _literal_text(d["doc"]) + "\0" + _literal_text(d.get("subject") or "") for d in documents
        ]
        self.dates = np.array(
            [date.fromisoformat(d["doc_dt"]).toordinal() if d["doc_dt"] else 0 for d in documents],
            dtype=np.int32,
        )
        boxes: dict[str, list[int]] = defaultdict(list)
        for i, doc in enumerate(documents):
            boxes[doc["synthetic_user"]].append(i)
        self.boxes = {key: np.asarray(rows, dtype=np.int32) for key, rows in boxes.items()}
        self.spelling = CorpusSpelling((d["doc"] for d in documents), _stem)
        self.lexical = LexicalIndex(documents)
        self.dense = DenseIndex(documents, Path(cache_dir))

    def search(
        self, synthetic_user: str, query: str, query_date: date, limit: int = 10
    ) -> list[tuple[str, float]]:
        box = self.boxes.get(synthetic_user)
        if box is None or limit < 1 or not query.strip():
            return []
        query, box, field_request = self.fields.restrict(query, box)
        if not len(box):
            return []
        phrases = []
        if '"' in query or "«" in query:
            query, phrases = _extract_phrases(query)
        direct = self.by_id.get(query.strip()) if not phrases else None
        if direct is not None:
            if direct in box and self.dates[direct] <= query_date.toordinal():
                return [(self.ids[direct], 1.0)]
            return []
        articles = []
        def extract_code(match):
            token = match[0].lower().replace("ё", "е")
            if not match.lastgroup and _article(token):
                articles.append("code:" + token)
                return " "
            return match[0]

        text = _CODE_CANDIDATE.sub(extract_code, self.spelling.correct(query))
        scope = parse_date_scope(text, query_date)
        dates = self.dates[box]
        available = dates <= query_date.toordinal()
        if scope.bounded:
            available &= dates > 0
            if scope.start:
                available &= dates >= scope.start.toordinal()
            if scope.end:
                available &= dates <= scope.end.toordinal()
        rows = box[available]
        if phrases:
            rows = np.asarray([
                row for row in rows
                if all(phrase.matches(self.literal_texts[row]) for phrase in phrases)
            ], dtype=box.dtype)
        if not len(rows):
            return []
        words = _words(scope.text)
        features = _features(words, query=True) + articles
        if field_request and not articles and only_field_request(scope.text):
            features = []
        if not features:
            if not words and not scope.bounded and not phrases and not field_request:
                return []
            order = np.argsort(-self.dates[rows], kind="stable")[:limit]
            return [(self.ids[rows[i]], 1.0) for i in order]

        # Opaque IDs, addresses and long numbers are strong literal constraints.
        identifiers = articles + [
            word.replace(",", ".") for i, word in enumerate(words)
            if "@" in word or word.startswith(("syn-", "l-")) or (
                word.isdigit() and len(word) >= 5
                and not (i + 1 < len(words) and re.fullmatch(_UNIT, words[i + 1]))
            )
        ]
        for token in dict.fromkeys(identifiers):
            if token.startswith("syn-") and token in self.opaque_rows:
                exact = np.intersect1d(rows, self.opaque_rows[token], assume_unique=True)
                if len(exact):
                    # Some imports glued non-Latin header labels to a reference.
                    # Keep the original fallback when no complete visible token exists.
                    rows = exact
                    continue
            if token.startswith("l-"):
                row = self.by_id.get(token, self.by_id.get("L-" + token[2:], -1))
                rows = rows[rows == row]
            elif token in self.lexical.vocabulary:
                rows = rows[self.lexical.contains(self.lexical.vocabulary[token], rows)]
            else:
                return []
            if not len(rows):
                return []
        if identifiers and not phrases:
            reference = _completed_order_reference(query.lower().replace("ё", "е"), identifiers)
            if reference:
                rows = np.asarray([
                    row for row in rows if not _warning_without_confirmation(
                        (self.documents[row].get("subject") or "") + "\n" + self.documents[row]["doc"], reference
                    )
                ], dtype=rows.dtype)
                if not len(rows):
                    return []
        if identifiers and len(rows) == 1:
            return [(self.ids[rows[0]], 1.0)]
        if articles and set(features) == set(articles):
            order = np.argsort(-self.dates[rows], kind="stable")[:limit]
            return [(self.ids[rows[i]], 1.0) for i in order]

        lexical_scale = None
        # In correspondence, exact names and nearby details distinguish otherwise similar messages.
        if (not scope.bounded and not phrases and not identifiers
                and not self.fields.has_visible_sender(synthetic_user, query_date)):
            lexical_scale = semantic_lexical_scale(text)
        # Natural questions retain casing, punctuation and their original phrasing for E5.
        terms = text if lexical_scale is not None else " ".join(w for w in words if _stem(w) not in _REQUEST_WORDS)
        vector = self.dense.query(terms)
        # Mailbox rows preserve corpus order through all scope filters. A single
        # contiguous interval can use the mapped matrix directly without a copy.
        contiguous = int(rows[-1]) - int(rows[0]) + 1 == len(rows)
        vectors = self.dense.vectors[rows[0]:rows[-1] + 1] if contiguous else self.dense.vectors[rows]
        cosine = vectors @ vector
        lexical, coverage = self.lexical.score(features, rows)
        semantic = np.clip((cosine - 0.74) / 0.17, 0, 1.25)
        lexical_weight = 0.25 + 0.30 * float(coverage.max())
        evidence = lexical_weight * lexical + (1 - lexical_weight) * semantic
        if lexical_scale is not None:
            # Dense scores of two good answers often differ by just a few hundredths.
            # Keep that contrast; close words alone are weak evidence of an answer.
            scores = (cosine - 0.74 + lexical_scale * lexical) / (0.17 + lexical_scale)
        else:
            shortlist = np.argsort(-evidence, kind="stable")[:max(128, limit)]
            evidence[shortlist] += 0.28 * self.lexical.proximity(features, rows[shortlist])
            scores = evidence
        # Keep paraphrases without shared words when their semantic signal is strong.
        plausible = ((coverage > 0) & (evidence >= 0.22)) | (cosine >= 0.835)
        distinct = set(features)
        unknown = distinct.difference(self.lexical.vocabulary)
        if not phrases and len(unknown) >= 2 and len(unknown) * 2 >= len(distinct):
            # A high cosine for the whole query can be entirely due to words
            # such as "order" or "invoice". Verify the unfamiliar description
            # separately before returning messages about the common subject.
            focus = " ".join(w for w in words if _ALIAS.get(_stem(w), _stem(w)) in unknown)
            if focus:
                backed, rare = self.lexical.literal_evidence(distinct, rows)
                support = vectors @ self.dense.query(focus)
                # Two independent specific matches can identify a letter even
                # when peripheral details were paraphrased. A single rare match
                # still needs semantic support for the unfamiliar description.
                plausible = backed | (plausible & ((support >= 0.85) | (rare & (support >= 0.81))))
                scores += 0.14 * backed
        if phrases:
            # Every candidate now has the requested literal evidence. Outside
            # words rank those matches; a weak embedding alone cannot hide one.
            plausible[:] = True
        # Preserve neutral, conditional and mixed-history messages. Exclude a
        # contradiction only if the explicit opposite decision is about the
        # same subject. A yes/no question can be answered by either outcome.
        request = scope.text.rstrip(" ?") if _SELECT_REQUEST.match(scope.text) else scope.text
        decision, context = _decisions([request])
        context -= _GENERIC_CONTEXT
        if decision and context:
            for i in np.flatnonzero(plausible):
                other, nearby = self.lexical.decisions[rows[i]]
                if other == -decision and 2 * len(context & nearby) > len(context):
                    plausible[i] = False
        if identifiers:
            plausible[:] = True
        if not phrases and not identifiers:
            self.lexical.rank_event_evidence(scope.text, distinct, rows, scores)
        order = np.flatnonzero(plausible)
        order = order[np.argsort(-scores[order], kind="stable")[:limit]]
        if not (identifiers or phrases or field_request):
            suspect = self.bilingual.ungrounded(scope.text, rows[order], _stem, _STOP, _REQUEST_WORDS)
            if not len(order) or suspect:
                translated = self.bilingual.search(scope.text, rows, self.dense, limit, _stem, _STOP, _REQUEST_WORDS)
                supported = {row for row, _ in translated}
                # Preserve already grounded or same-language hits and their order.
                kept = [(int(rows[i]), float(scores[i])) for i in order if rows[i] not in suspect or rows[i] in supported]
                used = {row for row, _ in kept}
                # The fallback score is a cosine; its scale differs from the hybrid score.
                ceiling = kept[-1][1] if kept else float("inf")
                ranked = kept + [(row, min(score, ceiling)) for row, score in translated if row not in used]
                return [(self.ids[row], score) for row, score in ranked[:limit]]
        return [(self.ids[rows[i]], float(scores[i])) for i in order]
