"""Query normalization and decision filtering on small synthetic mailboxes."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mailsearch.retrieval import SearchEngine, _stem
from mailsearch.spelling import CorpusSpelling


class SpellingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        texts = [f"Вложены квитанция, воронка и коронка, партия {n}. Получатель Вероника." for n in range(5)]
        texts.append("Редкое написание квитинция. " + "осциллограф " * 10)
        cls.spelling = CorpusSpelling(texts, _stem)

    def test_unique_frequent_word_with_one_edit(self):
        for typo in ("квтианция", "квитаннция", "квианция", "квипанция"):
            with self.subTest(typo=typo):
                self.assertEqual(self.spelling.correct(f"где {typo}?"), "где квитанция?")

    def test_literal_identifiers_quotes_and_names(self):
        query = 'SYN-квитаннция-123 и 091234, квитаннция@example.ru, «квитаннция», "квитаннция", Веротика'
        self.assertEqual(self.spelling.correct(query), query)

    def test_ambiguous_rare_known_and_inflected_words(self):
        # Both «воронка» and «коронка» are possible for «соронка».
        # The frequent personal name is capitalized in every source.
        for word in ("соронка", "осциллогаф", "квитинция", "квитанцию", "веротика", "ксилофон"):
            with self.subTest(word=word):
                self.assertEqual(self.spelling.correct(word), word)


class DecisionSearchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        messages = {
            "declined": "Банковский перевод отклонён.",
            "negated": "Банковский перевод не одобрен.",
            "approved": "Банковский перевод одобрен.",
            "conditional": "Если банковский перевод отклонён, обратитесь в поддержку.",
            "question": "Банковский перевод одобрен?",
            "history": "Сначала перевод отклонён. Повторный перевод одобрен.",
            "other_subject": "Перевод ожидает рассмотрения. Отпуск одобрен.",
        }
        docs = [{"doc_id": key, "synthetic_user": "test", "doc_dt": "2026-01-10", "doc": text}
                for key, text in messages.items()]
        # Equal semantic scores admit all candidates, isolating the real decision filter.
        with patch("mailsearch.retrieval.DenseIndex") as dense:
            dense.return_value.vectors = np.ones((len(docs), 1), dtype=np.float32)
            dense.return_value.query.return_value = np.ones(1, dtype=np.float32)
            cls.engine = SearchEngine(docs, Path("."))

    def search_ids(self, query):
        return {key for key, _ in self.engine.search("test", query, date(2026, 1, 10))}

    def test_explicit_refusal_only_excludes_opposite_decisions_on_same_subject(self):
        self.assertEqual(self.search_ids("Покажи отклонённые переводы"), {
            "declined", "negated", "conditional", "question", "history", "other_subject",
        })

    def test_yes_no_question_keeps_either_answer(self):
        self.assertTrue({"approved", "declined", "negated"} <= self.search_ids("Перевод одобрен?"))


class LiteralIdentifiers(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        docs = [
            {"doc_id": "archive:mail/1", "synthetic_user": "first", "doc_dt": "2026-03-05",
             "doc": "Устройство с артикулом К17324901 прошло проверку."},
            {"doc_id": "archive:mail/2", "synthetic_user": "first", "doc_dt": "2026-03-04",
             "doc": "Устройство с артикулом K17324901 доставлено."},
            {"doc_id": "archive:mail/3", "synthetic_user": "second", "doc_dt": None,
             "doc": "Заявка принята."},
        ]
        with patch("mailsearch.retrieval.DenseIndex"):
            cls.engine = SearchEngine(docs, Path("."))

    def test_native_id_obeys_owner_and_date(self):
        day = date(2026, 3, 5)
        self.assertEqual(self.engine.search("first", "archive:mail/1", day), [("archive:mail/1", 1.0)])
        self.assertFalse(self.engine.search("first", "archive:mail/1", date(2026, 3, 4)))
        self.assertFalse(self.engine.search("second", "archive:mail/1", day))

    def test_product_code_keeps_its_alphabet(self):
        day = date(2026, 3, 5)
        for code, expected in (("К17324901", "archive:mail/1"), ("K17324901", "archive:mail/2")):
            with self.subTest(code=code):
                self.assertEqual(self.engine.search("first", f"Найди артикул {code}", day), [(expected, 1.0)])


if __name__ == "__main__":
    unittest.main()
