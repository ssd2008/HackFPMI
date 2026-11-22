"""Regression cases for event evidence and explicit search intent."""
from datetime import date
from unittest import TestCase

from experiments.ordinary_cycle.ordinary import EventRanker, referenced_events


class FixedRanking:
    def __init__(self, hits):
        self.hits = hits

    def search(self, box, query, at, limit):
        return self.hits[:limit]


class EventRankingTests(TestCase):
    day = date(2026, 1, 12)
    direct = "Покажи уведомления о ремонте крыши."

    def build(self, rows, *, permission=1.5):
        docs = [{"doc_id": did, "doc": text} for did, _, text in rows]
        base = FixedRanking([(did, score) for did, score, _ in rows])
        return base, EventRanker(base, docs, {"permission": permission, "plan": 0.0})

    def test_permission_drops_only_as_far_as_an_independent_report(self):
        base, ranker = self.build([
            ("permit", .9, "Получено разрешение комиссии на ремонт крыши."),
            ("done", .8, "Ремонт крыши завершен по графику."),
            ("other", .65, "Остатки строительных материалов."),
        ])
        ordered = ranker.search("box", self.direct, self.day, 3)
        self.assertEqual([x for x, _ in ordered], ["done", "permit", "other"])
        self.assertTrue(.65 < ordered[1][1] < .8)
        for query in ("Покажи всю переписку по теме ремонта крыши.",
                      "Покажи уведомления о разрешении на ремонт крыши.",
                      "Найди письма с темой «ремонт крыши»."):
            self.assertEqual(ranker.search("box", query, self.day, 3), base.hits)

    def test_preserves_unrelated_authorization_and_separate_confirmation(self):
        base, ranker = self.build([
            ("mixed", .9, "Дано разрешение на ремонт. Ремонт завершен."),
            ("elsewhere", .8, "Дано разрешение на демонтаж лестницы. Ремонт крыши выполнен."),
            ("done", .7, "Ремонт крыши завершен."),
        ])
        self.assertEqual(ranker.search("box", self.direct, self.day, 3), base.hits)

    def test_permission_only_and_zero_weight_leave_scores_unchanged(self):
        rows = [
            ("permit", .9, "Получено разрешение на ремонт крыши."),
            ("unrelated", .8, "Прием заявок на следующий год."),
        ]
        base, ranker = self.build(rows)
        self.assertEqual(ranker.search("box", self.direct, self.day, 2), base.hits)
        base, disabled = self.build(rows + [("done", .7, "Ремонт крыши завершен.")], permission=0.)
        self.assertEqual(disabled.search("box", self.direct, self.day, 2), base.hits[:2])

    def test_consent_noun_inflections_and_verbs_are_distinct(self):
        for noun in ("согласие", "согласия", "согласию", "согласием", "согласии",
                     "согласий", "согласиям", "согласиями", "согласиях"):
            with self.subTest(noun=noun):
                permission, plan = referenced_events(f"Документ: {noun} на ремонт.")
                self.assertEqual(permission, {"ремонт"})
                self.assertEqual(plan, set())
        for word in ("согласился", "согласилась", "согласились", "согласилось", "согласится"):
            with self.subTest(verb=word):
                self.assertIsNone(referenced_events(f"В обсуждении: {word} на ремонт."))
        base, ranker = self.build([
            ("report", .9, "Подрядчик согласился на ремонт."),
            ("done", .8, "Ремонт выполнен."),
            ("permit", .7, "Ранее получено согласие на проведение экспертизы."),
        ])
        self.assertEqual(ranker.search("box", self.direct, self.day, 3), base.hits)
