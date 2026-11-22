"""Quoted strings constrain retrieval to an exact span of visible message text."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mailsearch.retrieval import SearchEngine


class LiteralSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def letter(key, text, day="2026-05-01", box="inbox", **extra):
            return dict(doc_id=key, doc=text, doc_dt=day, synthetic_user=box, **extra)

        documents = [
            letter("barometer", "Тема: Барометр\nСЕРЕБРЯНЫЙ\n\tЁЖ. Акт №12/7 [готов]. "
                   "Срок 31.12.2045. Номер L-00000000000000000000."),
            letter("notebook", "Несеребряный ежик и серебряный еж. Тетрадь.", "2026-04-30"),
            letter("separate-subject", "Открыт семинар.", None, subject="Серебряный еж"),
            letter("punctuation", "Серебряный, ёж. Акт №12/7 [готов]."),
            letter("left-fragment", "Пришёл несеребряный еж."),
            letter("right-fragment", "Серебряный ежик."),
            letter("reversed", "Ёж серебряный."),
            letter("future", "Серебряный еж. Срок 31.12.2045.", "2026-05-04"),
            letter("foreign", "Серебряный еж. Срок 31.12.2045.", box="second"),
        ]
        # The literal path must work without semantic help. A zero-valued
        # encoder also makes the ranking of unquoted terms deterministic.
        with patch("mailsearch.retrieval.DenseIndex") as encoder:
            encoder.return_value.vectors = np.zeros((len(documents), 2), dtype=np.float32)
            encoder.return_value.query.return_value = np.zeros(2, dtype=np.float32)
            cls.engine = SearchEngine(documents, Path("."))
        cls.as_of = date(2026, 5, 2)

    def ids(self, query, limit=20, box="inbox"):
        return [key for key, _ in self.engine.search(box, query, self.as_of, limit)]

    def test_exact_words_in_order_across_whitespace_in_text_or_subject(self):
        expected = ["barometer", "notebook", "separate-subject"]
        self.assertEqual(self.ids('«СЕРЕБРЯНЫЙ    ЁЖ»'), expected)
        self.assertEqual(self.ids('"серебряный еж"'), expected)
        self.assertEqual(self.ids('«серебряный еж»', limit=1), expected[:1])
        self.assertEqual(self.ids('«серебряный еж»', box="second"), ["foreign"])
        self.assertEqual(self.ids('«ёж серебряный»'), ["reversed"])

    def test_every_phrase_is_required_and_punctuation_is_literal(self):
        self.assertEqual(self.ids('«серебряный еж» "Акт №12/7 [готов]"'), ["barometer"])
        self.assertEqual(self.ids('«серебряный еж» "Акт №12-7 [готов]"'), [])
        self.assertEqual(self.ids('«серебряный, еж»'), ["punctuation"])
        self.assertEqual(self.ids('«серебряный» «кефирный термос»'), [])

    def test_quoted_dates_and_references_stay_in_content(self):
        self.assertEqual(self.ids('«31.12.2045»'), ["barometer"])
        self.assertEqual(self.ids('"L-00000000000000000000"'), ["barometer"])
        self.assertEqual(self.ids('«серебряный еж» вчера'), ["barometer"])
        self.assertEqual(self.ids('«серебряный еж» 30 апреля 2026 года'), ["notebook"])
        self.assertEqual(self.ids('«31.12.2045» 30 апреля 2026 года'), [])

    def test_outside_search_terms_rank_only_literal_matches(self):
        result = self.ids('тетрадь «серебряный еж»')
        self.assertEqual(result[0], "notebook")
        self.assertEqual(set(result), {"notebook", "barometer", "separate-subject"})
        self.assertEqual(set(self.ids('Найди письма с точной фразой «серебряный еж»')), set(result))

    def test_unpaired_quotes_follow_the_ordinary_path(self):
        expected = self.ids("барометр")
        self.assertEqual(expected, ["barometer"])
        self.assertEqual(self.ids('«барометр'), expected)
        self.assertEqual(self.ids('"барометр'), expected)


if __name__ == "__main__":
    unittest.main()
