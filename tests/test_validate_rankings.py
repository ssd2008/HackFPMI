"""Paired ranking evaluation on invented queries, decisions, and report snapshots."""

import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tools.validate_rankings import Pair, Task, compare, paired_ndcg, prepare_tasks, sha


class ValidationTest(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        qids = ["single", "list", "empty", "unseen", "typo"]
        self.queries = [dict(query_id=qid, synthetic_user="box", query=qid, query_date="2026-05-10") for qid in qids]
        self.labels = {
            "single": {"a": 2, "copy": 2, "context": 1, "bad": 0},
            "list": {"b": 1}, "empty": {"bad": 0}, "unseen": {},
            "typo": {"a": 2, "copy": 2},
        }
        self.tasks = [
            dict(query_id="single", family_id="f1", split="dev", task_type="known_item",
                 state="positive", target_ids=["a", "copy"]),
            dict(query_id="list", family_id="f2", split="dev", task_type="collection", state="positive"),
            dict(query_id="empty", family_id="f3", split="blind", task_type="empty", state="verified_empty"),
            dict(query_id="unseen", family_id="f4", split="blind", task_type="collection", state="unjudged"),
            dict(query_id="typo", family_id="f1", split="dev", task_type="known_item",
                 state="positive", target_ids=["a", "copy"], variant="typo", original_query_id="single"),
        ]
        self.write_lines("queries", self.queries)
        self.write_lines("tasks", self.tasks)
        rels = [dict(query_id=q, doc_id=d, relevance=g) for q, docs in self.labels.items() for d, g in docs.items()]
        self.write_lines("qrels", rels)
        before = {"single": ["bad", "unknown", "context", "copy"], "list": [], "empty": ["bad"], "unseen": [], "typo": []}
        after = {"single": ["a", "bad", "unknown"], "list": ["unknown", "b"], "empty": [], "unseen": ["unknown"], "typo": ["copy"]}
        for version, ranks in [("before", before), ("after", after)]:
            dataset = {"documents": 100, "queries": 5, "qrels": len(rels),
                       "sha256": {"documents.jsonl": "2" * 64,
                                  **{k: sha((self.root/k).read_bytes()) for k in ("queries", "qrels")}}}
            rows = {q: {"ranked_ids": ids, "relevant": sum(g > 0 for g in self.labels[q].values()) if self.labels[q] else None}
                    for q, ids in ranks.items()}
            self.write_json(version, {"dataset": dataset, "queries": rows, "timing": {"top_k": 10}})

    def write_lines(self, name, rows):
        (self.root / name).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def write_json(self, name, value):
        (self.root / name).write_text(json.dumps(value), encoding="utf-8")

    def evaluate(self):
        return compare(*(self.root / name for name in ("before", "after", "queries", "qrels", "tasks")), resamples=100)

    def test_metrics_separate_states_and_exclude_variants(self):
        report = self.evaluate()
        primary = report["primary"]["all"]
        self.assertEqual(primary["queries"], 4)
        self.assertEqual(primary["states"], {"positive": 2, "verified_empty": 1, "unjudged": 1})
        self.assertEqual(report["variants"]["typo"]["all"]["queries"], 1)
        m = primary["metrics"]
        ideal = 3 + 3 / math.log2(3) + 1 / 2
        self.assertAlmostEqual(m["ndcg_at_10"]["after"], (3 / ideal + 1 / math.log2(3)) / 2)
        self.assertAlmostEqual(m["known_recall_at_10"]["before"], 1 / 3)
        self.assertAlmostEqual(m["mrr_at_10"]["before"], .25)
        self.assertEqual(m["success_at_5"]["before"], 1)
        self.assertEqual(m["success_at_1"]["after"], 1)
        self.assertEqual(m["judged_at_10"]["before"], .75)
        self.assertAlmostEqual(m["judged_at_10"]["after"], (2/3 + 1/2)/2)
        self.assertEqual((m["judged_at_10"]["n_before"], m["judged_at_10"]["n_after"]), (1, 2))
        self.assertEqual((m["false_empty_rate"]["events_before"], m["false_empty_rate"]["n_before"]), (1, 2))
        self.assertEqual((m["false_nonempty_rate"]["events_before"], m["false_nonempty_rate"]["n_before"]), (1, 1))
        blind = report["primary"]["by_split"]["blind"]
        self.assertIsNone(blind["metrics"]["ndcg_at_10"]["after"])
        self.assertIsNone(blind["paired_ndcg_at_10"]["ci95"])

    def test_rejects_shared_family_across_split_and_inconsistent_judgments(self):
        self.tasks[1].update(family_id="f1", split="blind")
        self.write_lines("tasks", self.tasks)
        with self.assertRaisesRegex(ValueError, "Семейство"):
            self.evaluate()
        self.tasks[1].update(family_id="f2", state="unjudged")
        self.write_lines("tasks", self.tasks)
        with self.assertRaisesRegex(ValueError, "положительными qrels"):
            self.evaluate()

    def test_optional_task_mailbox_and_date_match_query(self):
        queries = {q["query_id"]: q for q in self.queries}
        self.tasks[1].update(synthetic_user="box", query_date="2026-05-10")
        self.assertEqual(len(prepare_tasks(self.tasks, queries, self.labels)), len(queries))
        for field, other in (("synthetic_user", "other-box"), ("query_date", "2026-05-11")):
            with self.subTest(field=field):
                self.tasks[1][field] = other
                with self.assertRaisesRegex(ValueError, field):
                    prepare_tasks(self.tasks, queries, self.labels)
                self.tasks[1][field] = queries["list"][field]

    def test_explicit_owner_and_chain_are_each_confined_to_one_split(self):
        queries = {q["query_id"]: q for q in self.queries}
        queries["empty"]["synthetic_user"] = "second-box"
        for field in ("owner_id", "chain_id"):
            with self.subTest(field=field):
                rows = [dict(row) for row in self.tasks]
                rows[0][field] = rows[1][field] = "same"
                rows[2][field] = "different"
                self.assertEqual(len(prepare_tasks(rows, queries, self.labels)), len(queries))
                rows[2][field] = "same"
                with self.assertRaisesRegex(ValueError, field):
                    prepare_tasks(rows, queries, self.labels)

    def test_supplementary_and_collection_positives_do_not_cross_split(self):
        queries = {q["query_id"]: q for q in self.queries}
        self.tasks[1]["split"] = "blind"
        for doc, grade in (("context", 1), ("copy", 2)):
            with self.subTest(doc=doc):
                self.labels["list"][doc] = grade
                with self.assertRaisesRegex(ValueError, "doc_id"):
                    prepare_tasks(self.tasks, queries, self.labels)
                del self.labels["list"][doc]
        self.labels["list"]["context"] = 0
        self.assertEqual(len(prepare_tasks(self.tasks, queries, self.labels)), len(queries))

    def test_snapshots_labels_query_identity_depth_and_rank_duplicates(self):
        original = (self.root / "after").read_text()
        for field, value, message in (("documents.jsonl", "3"*64, "documents_sha256"),
                                      ("queries", "3"*64, "запросов"), ("qrels", "3"*64, "qrels")):
            obj = json.loads(original)
            obj["dataset"]["sha256"][field] = value
            self.write_json("after", obj)
            with self.assertRaisesRegex(ValueError, message):
                self.evaluate()
        obj = json.loads(original)
        obj["timing"]["top_k"] = 50
        self.write_json("after", obj)
        with self.assertRaisesRegex(ValueError, "requested_depth"):
            self.evaluate()
        obj = json.loads(original)
        obj["queries"]["list"]["ranked_ids"] = ["b", "b"]
        self.write_json("after", obj)
        with self.assertRaisesRegex(ValueError, "повтор документа"):
            self.evaluate()

    def test_common_posthoc_labels_for_explicitly_unjudged_rankings(self):
        for name in ("before", "after"):
            obj = json.loads((self.root / name).read_text())
            obj["dataset"]["sha256"].pop("qrels")
            obj["dataset"].update(qrels=0, qrels_file=None)
            for row in obj["queries"].values():
                row["relevant"] = None
            self.write_json(name, obj)
        output = self.evaluate()
        self.assertFalse(output["inputs"]["before"]["qrels_in_ranking_report"])
        self.assertEqual(output["primary"]["all"]["metrics"]["mrr_at_10"]["after"], 1)

    def test_cluster_bootstrap_resamples_whole_families_and_keeps_query_weights(self):
        def pair(q, family, before, after, source="source"):
            task = Task(q, family, "dev", "positive", "collection", source, "box", "original", frozenset())
            return Pair(task, {"ndcg_at_10": before}, {"ndcg_at_10": after})

        cases = [pair(str(i), "three", 0, .5) for i in range(3)] + [pair("fourth", "one", 1, 0)]
        result = paired_ndcg(cases, seed=12, resamples=1000)
        self.assertEqual(result["positive_families"], 2)
        self.assertEqual(result["delta"], .125)
        self.assertEqual(result["ci95"], [-1, .5])
        self.assertEqual((result["wins"], result["ties"], result["losses"]), (3, 0, 1))
        self.assertEqual(result, paired_ndcg(cases, seed=12, resamples=1000))
        self.assertIsNone(paired_ndcg(cases[:3], seed=12, resamples=1000)["ci95"])
