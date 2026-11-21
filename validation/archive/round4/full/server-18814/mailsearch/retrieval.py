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
from .spelling import CorpusSpelling


_EMAIL = r"[a-z0-9_.+%-]+@[a-z0-9.-]+\.[a-z]{2,}"
_OPAQUE = r"(?:syn|l)-[a-z0-9]+(?:-[a-z0-9]+)*"
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
_QUOTED = re.compile(r'«([^«»]*)»|"([^"]*)"')
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


def _extract_phrases(query: str) -> tuple[str, list[tuple[str, re.Pattern]]]:
    phrases = []

    def extract(match):
        phrase = _literal_text(match[match.lastindex])
        if phrase:
            # Preserve punctuation literally. Only alphanumeric/underscore
            # endpoints require a boundary to prevent a match inside a word.
            left = r"(?<!\w)" if re.match(r"\w", phrase[0]) else ""
            right = r"(?!\w)" if re.match(r"\w", phrase[-1]) else ""
            phrases.append((phrase, re.compile(left + re.escape(phrase) + right)))
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
        result.append(_ALIAS.get(stem, stem).replace(",", "."))
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
        columns, counts, pointers = array("i"), array("f"), array("i", [0])
        lengths = []
        for doc in documents:
            passages = _CLAUSE.split(doc["doc"])
            self.decisions.append(_decisions(passages))
            clauses = [_features(_words(clause)) for clause in passages]
            weights = Counter(term for clause in clauses for term in clause)
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
        # Words outside the corpus are still part of the request. Renormalizing
        # over known words alone turns "order for [unknown item]" into "order".
        weights *= (len(cols) / len(terms)) / weights.sum()
        hits = self.weights[candidates][:, cols]
        coverage = np.asarray((hits > 0) @ weights).ravel()
        frequency = np.asarray(hits @ weights).ravel()
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
        present = self.weights[candidates][:, cols] > 0
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
                if all(phrase in self.literal_texts[row] and pattern.search(self.literal_texts[row])
                       for phrase, pattern in phrases)
            ], dtype=box.dtype)
        if not len(rows):
            return []
        words = _words(scope.text)
        features = _features(words, query=True) + articles
        if not features:
            if not words and not scope.bounded and not phrases:
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
            if token.startswith("l-"):
                row = self.by_id.get(token, self.by_id.get("L-" + token[2:], -1))
                rows = rows[rows == row]
            elif token in self.lexical.vocabulary:
                hits = self.lexical.weights[rows, self.lexical.vocabulary[token]].toarray().ravel()
                rows = rows[hits > 0]
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

        terms = " ".join(w for w in words if _stem(w) not in _REQUEST_WORDS)
        vector = self.dense.query(terms)
        cosine = self.dense.vectors[rows] @ vector
        lexical, coverage = self.lexical.score(features, rows)
        semantic = np.clip((cosine - 0.74) / 0.17, 0, 1.25)
        lexical_weight = 0.25 + 0.30 * float(coverage.max())
        scores = lexical_weight * lexical + (1 - lexical_weight) * semantic
        shortlist = np.argsort(-scores, kind="stable")[:max(128, limit)]
        scores[shortlist] += 0.28 * self.lexical.proximity(features, rows[shortlist])
        # Keep paraphrases without shared words when their semantic signal is strong.
        plausible = ((coverage > 0) & (scores >= 0.22)) | (cosine >= 0.835)
        distinct = set(features)
        unknown = distinct.difference(self.lexical.vocabulary)
        if not phrases and len(unknown) >= 2 and len(unknown) * 2 >= len(distinct):
            # A high cosine for the whole query can be entirely due to words
            # such as "order" or "invoice". Verify the unfamiliar description
            # separately before returning messages about the common subject.
            focus = " ".join(w for w in words if _ALIAS.get(_stem(w), _stem(w)) in unknown)
            if focus:
                backed, rare = self.lexical.literal_evidence(distinct, rows)
                support = self.dense.vectors[rows] @ self.dense.query(focus)
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
        order = np.flatnonzero(plausible)
        order = order[np.argsort(-scores[order], kind="stable")[:limit]]
        return [(self.ids[rows[i]], float(scores[i])) for i in order]
