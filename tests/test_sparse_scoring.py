import unittest

import numpy as np

from mailsearch.retrieval import LexicalIndex, _features, _words


class SparseScoring(unittest.TestCase):
    def test_request_for_an_event_preserves_plans_as_secondary_evidence(self):
        index = LexicalIndex([
            {"doc": "Заявка поставщика на отгрузку партии."},
            {"doc": "Партия отгружена. Ведём журнал."},
            {"doc": "Разрешение на отгрузку выдано. Отгрузка завершена."},
        ])
        rows = np.arange(3, dtype=np.int32)
        for query, discount in (("отгрузка партии", True), ("разрешение на отгрузку", False)):
            with self.subTest(query=query):
                values = np.ones(3, dtype=np.float32)
                index.rank_event_evidence(query, set(_features(_words(query), query=True)), rows, values)
                self.assertEqual(values[0] < 1, discount)
                np.testing.assert_array_equal(values[1:], [1, 1])

    def test_posting_scores_match_matrix_scoring_for_both_scope_sizes(self):
        words = ("red", "blue", "green", "purple", "orange")
        index = LexicalIndex([
            {"doc": " ".join(w for j, w in enumerate(words) if i % (j + 2) == 0) or "gray"}
            for i in range(80)
        ])
        terms = ("blue", "missing", "red", "blue")
        columns = [index.vocabulary[t] for t in sorted(set(terms)) if t in index.vocabulary]
        weights = index.idf[columns].astype(np.float32)
        weights *= (len(columns) / len(set(terms))) / weights.sum()
        for requested in ([], [11], [9, 0, 9], list(range(7, 56, 3)), list(range(80))):
            with self.subTest(requested=requested):
                rows = np.asarray(requested, dtype=np.int32)
                matched = index.postings.tocsr()[rows][:, columns].toarray()
                coverage = (matched > 0) @ weights
                scores, actual_coverage = index.score(list(terms), rows)
                np.testing.assert_allclose(actual_coverage, coverage, atol=1e-7)
                np.testing.assert_allclose(scores, .48 * coverage + .52 * (matched @ weights), atol=1e-7)
                np.testing.assert_array_equal(index.contains(columns[0], rows), matched[:, 0] > 0)
