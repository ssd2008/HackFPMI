"""An English message needs local evidence for each requested topic in a visible mailbox."""

from datetime import date
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from mailsearch.retrieval import SearchEngine, _REQUEST_WORDS, _STOP, _stem


class BilingualMailTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        def message(index, subject, body, box="mixed", received="2028-03-01"):
            return dict(doc_id=index, doc=body, synthetic_user=box, doc_dt=received,
                        subject=subject, sender="Music shop <shop@example.test>")

        rows = [
            message("guitar", "Fender guitar maintenance", "We can repair your Fender guitar this Wednesday."),
            message("remote", "Weekly report", "Guitar strings are available. " + "The band starts soon. " * 25
                    + "The roof repair budget was approved."),
            message("bill", "Invoice", "This invoice is for an electric piano."),
            message("ru", "Расписание", "У нас появились билеты на ближайшую репетицию."),
            message("later", "Fender guitar repair", "Fender guitar repair is scheduled for tomorrow.", received="2028-04-01"),
            message("other", "Fender guitar repair", "I can repair your Fender guitar tomorrow.", box="other"),
            message("short", "Invoice", "Attached"),
            message("mixed-language", "Rehearsal", "The rehearsal starts at seven. Bring a балалайка."),
            message("earned", "Studio reward", "We credited 25 extra tokens to your wallet.", box="perks"),
            message("spent", "Activity report", "You spent 12 tokens on a rehearsal.", box="perks"),
            message("conditional", "Offer", "If you join today, the gift includes 40 tokens.", box="perks"),
            message("latest", "Studio reward", "We credited 10 tokens to your wallet.", box="perks", received="2028-04-01"),
        ]
        with patch("mailsearch.retrieval.DenseIndex") as dense:
            # General retrieval abstains on these Russian phrases. The English
            # documents barely exceed the stricter translated-evidence threshold.
            dense.return_value.vectors = np.tile(np.array([.80, .60], dtype=np.float32), (len(rows), 1))
            dense.return_value.query.return_value = np.array([1., 0.], dtype=np.float32)
            cls.engine = SearchEngine(rows, Path("."))

    def fallback(self, text, rows=(0, 1, 2, 3)):
        return [i for i, _ in self.engine.bilingual.search(text, rows, self.engine.dense, 10, _stem, _STOP, _REQUEST_WORDS)]

    def test_real_translations_and_local_evidence(self):
        self.assertEqual(self.fallback("Покажи письма о ремонте гитары"), [0])
        self.assertEqual(self.fallback("Найди письма о ремонте гитары Fender"), [0])
        self.assertEqual(self.fallback("Покажи письма о ремонте гитары invoice"), [])
        self.assertEqual(self.fallback("Покажи письма о ремонте гитары AI"), [])
        self.assertEqual(self.fallback("Покажи письма о ремонте ксилофона invoice"), [])

    def test_unsupported_constraints_and_topics_do_not_turn_into_matches(self):
        for q in ("Покажи письма о ремонте 30 гитар", "Покажи письма о ремонте без гитары",
                  "Найди письма о ремонте фырфырфыра", "Сколько стоит ремонт гитары?"):
            with self.subTest(q=q):
                self.assertEqual(self.fallback(q), [])

    def test_single_english_keyword_including_short_mail_is_insufficient(self):
        query = "Покажи письма о ксилофоне invoice"
        self.assertIn(6, self.engine.bilingual.content)
        self.assertEqual(self.engine.search("mixed", query, date(2028, 3, 1)), [])

    def test_descriptive_search_and_several_english_keywords_keep_their_ranking(self):
        review = self.engine.bilingual.ungrounded
        for q in ("Найди переписку мастерской Fender с расписанием ремонта",
                  "Покажи письма о концерте Fender guitar"):
            with self.subTest(q=q):
                self.assertFalse(review(q, [0, 2], _stem, _STOP, _REQUEST_WORDS))

    def test_actual_russian_content_in_an_english_letter_remains_grounded(self):
        query = "Покажи письма о балалайке"
        check = self.engine.bilingual.ungrounded(query, [0, 7], _stem, _STOP, _REQUEST_WORDS)
        self.assertEqual(check, {0})

    def test_one_scoped_number_must_have_the_requested_unit_and_transaction(self):
        query = "Сколько токенов мне вчера начислили?"
        ids = lambda q, day: [did for did, _ in self.engine.search("perks", q, day)]
        self.assertEqual(ids(query, date(2028, 3, 2)), ["earned"])
        self.assertEqual(ids(query, date(2028, 2, 28)), [])
        self.assertEqual(ids("Сколько кредитов мне вчера начислили?", date(2028, 3, 2)), [])
        self.assertIsNone(self.engine.bilingual.counted_event(query.lower(), [8, 11], self.engine.documents))

    def test_language_is_per_message_and_visibility_comes_first(self):
        self.assertNotIn(3, self.engine.bilingual.content)
        self.assertEqual(self.fallback("Найди письма о гитарах Fender", [2, 3]), [])
        hits = self.engine.search("mixed", "Найди письма о ремонте гитары", date(2028, 3, 1))
        self.assertEqual([i for i, _ in hits], ["guitar"])
        self.assertFalse(self.engine.search("mixed", "Найди письма о ремонте гитары", date(2028, 2, 1)))
        self.assertEqual(self.engine.search("mixed", "«ремонте гитары»", date(2028, 3, 1)), [])


if __name__ == "__main__":
    unittest.main()
