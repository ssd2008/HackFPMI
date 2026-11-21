import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).resolve().parents[1] / "tools/prepare_judging.py"


class JudgingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        docs = [{"doc_id": f"d{i}", "synthetic_user": "a", "doc_dt": "2020-01-10",
                 "doc": f"Вымышленное письмо {i}"} for i in range(14)]
        docs[12]["doc_dt"] = None
        docs.extend([{"doc_id": "foreign", "synthetic_user": "b", "doc_dt": "2019-01-01", "doc": "Другой ящик"},
                     {"doc_id": "future", "synthetic_user": "a", "doc_dt": "2020-01-11", "doc": "Более позднее"}])
        raw = "".join(json.dumps(doc) + "\n" for doc in docs).encode()
        (self.data / "documents.jsonl").write_bytes(raw)
        self.sha = hashlib.sha256(raw).hexdigest()
        self.dump("owners.json", [{"owner_id": "owner-a", "split": "dev", "mailboxes": ["a"], "documents_sha256": self.sha},
                                  {"owner_id": "owner-b", "split": "blind", "mailboxes": ["b"], "documents_sha256": self.sha}])
        self.task = {"query_id": "q", "owner_id": "owner-a", "split": "dev", "synthetic_user": "a",
                     "query_date": "2020-01-10", "query": "Поиск для теста", "goal": "Открыть письмо",
                     "success_criterion": "Письмо содержит подтверждение.", "known_doc_ids": ["d12"],
                     "manual_candidate_ids": ["d13"]}
        self.dump("tasks.jsonl", self.task)
        self.invoke("queries", "queries.jsonl")
        self.pred("a.json", [f"d{i}" for i in range(11)])
        self.pred("b.json", ["d1", "d2", "d2", "d11"])

    def dump(self, name, value):
        (self.root / name).write_text(json.dumps(value, ensure_ascii=False) + "\n", encoding="utf-8")

    def invoke(self, action, output, *extra, success=True):
        result = subprocess.run(
            [sys.executable, str(CLI), action, "--owners", "owners.json", "--owner", "owner-a",
             "--tasks", "tasks.jsonl", "--data", "data", "--output", output, *extra],
            cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return self.root / output

    def pred(self, filename, ranking, doc_sha=None, query_sha=None):
        self.dump(filename, {"dataset": {"sha256": {"documents.jsonl": doc_sha or self.sha,
                  "queries": query_sha or hashlib.sha256((self.root / "queries.jsonl").read_bytes()).hexdigest()}},
                  "timing": {"top_k": 11}, "queries": {"q": {"ranked_ids": ranking}}})

    def test_initialize_empty_owner_forms_and_preserve_existing_answers(self):
        second = self.root / "second"
        second.mkdir()
        row = {"doc_id": "x", "synthetic_user": "c", "doc_dt": None, "doc": "Тестовое содержимое"}
        (second / "documents.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        output = self.root / "private"
        command = [sys.executable, str(CLI), "init", "--dev-data", str(self.data),
                   "--blind-data", str(second), "--output", str(output)]
        subprocess.run(command, capture_output=True, check=True)
        registry = json.loads((output / "owners.json").read_bytes())
        self.assertEqual([r["mailboxes"] for r in registry], [["a", "b"], ["c"]])
        self.assertEqual(registry[0]["documents_sha256"], self.sha)
        task_ids = set()
        for profile in registry:
            form = output / profile["split"] / "tasks.jsonl"
            tasks = [json.loads(line) for line in form.read_bytes().splitlines()]
            self.assertEqual(len(tasks), 6)
            for task in tasks:
                task_ids.add(task["query_id"])
                self.assertEqual(task["owner_id"], profile["owner_id"])
                self.assertEqual(task["split"], profile["split"])
                self.assertTrue(all(not task[k] for k in ("query", "query_date", "synthetic_user", "goal",
                                                         "success_criterion", "task_type", "state", "empty_verification",
                                                         "known_doc_ids", "manual_candidate_ids")))
        self.assertEqual(len(task_ids), 12)
        edited = output / "blind" / "tasks.jsonl"
        edited.write_text("Здесь владелец уже записал свой запрос.\n", encoding="utf-8")
        previous = edited.read_bytes()
        self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
        self.assertEqual(edited.read_bytes(), previous)

    def test_shuffled_pool_and_partial_consensus(self):
        first = self.invoke("pool", "pool.json", "--predictions", "a.json", "b.json", "--seed", "11")
        reverse = self.invoke("pool", "reverse.json", "--predictions", "b.json", "a.json", "--seed", "11")
        self.assertEqual(first.read_bytes(), reverse.read_bytes())
        pool = json.loads(first.read_bytes())
        group = pool["tasks"][0]
        self.assertNotIn("known_doc_ids", group)
        picks = group["candidates"]
        self.assertEqual({c["doc_id"] for c in picks}, {f"d{i}" for i in range(14)} - {"d10"})
        self.assertEqual(len(picks), 13)
        self.assertTrue(all(set(c) == {"doc_id","doc","doc_dt","relevance"} and c["relevance"] is None for c in picks))
        judged = copy.deepcopy(pool)
        for grade, c in enumerate(judged["tasks"][0]["candidates"][:3]):
            c["relevance"] = grade
        self.dump("review.json", judged)
        qrels = self.invoke("qrels", "qrels.jsonl", "--pool", "pool.json", "--consensus", "review.json")
        entries = [json.loads(line) for line in qrels.read_text().splitlines()]
        self.assertEqual([r["relevance"] for r in entries], [0, 1, 2])
        self.assertTrue(all(set(r)=={"query_id","doc_id","relevance"} for r in entries))
        judged["tasks"][0]["candidates"][0]["relevance"] = False
        self.dump("review.json", judged)
        self.invoke("qrels", "bad-out", "--pool", "pool.json", "--consensus", "review.json", success=False)
        self.assertFalse((self.root / "bad-out").exists())

    def test_guard_scope_snapshot_task_and_destination(self):
        for did in ("foreign", "future", "missing"):
            self.pred("b.json", [did])
            self.invoke("pool", "bad", "--predictions", "a.json", "b.json", success=False)
        self.pred("b.json", ["d0"], doc_sha="other")
        self.invoke("pool", "bad", "--predictions", "a.json", "b.json", success=False)
        self.pred("b.json", ["d0"], query_sha="other")
        self.invoke("pool", "bad", "--predictions", "a.json", "b.json", success=False)
        self.invoke("pool", "bad", "--predictions", "a.json", "a.json", success=False)
        self.dump("tasks.jsonl", {**self.task, "synthetic_user": "b"})
        self.invoke("queries", "bad", success=False)
        self.dump("tasks.jsonl", {**self.task, "known_doc_ids": ["future"]})
        self.invoke("queries", "bad", success=False)
        self.dump("tasks.jsonl", {**self.task, "query": ""})
        self.invoke("queries", "bad", success=False)
        self.dump("tasks.jsonl", self.task)
        self.invoke("queries", "queries.jsonl", success=False)
        self.assertFalse((self.root / "bad").exists())

    def test_owner_judgments_to_paired_comparison(self):
        known = {**self.task, "task_type": "known_item", "state": "positive"}
        empty = {**self.task, "query_id": "e", "query": "Поиск отсутствующего письма", "task_type": "empty",
                 "known_doc_ids": [], "manual_candidate_ids": [], "state": "verified_empty",
                 "empty_verification": "Владелец отдельно просмотрел все доступные письма по своему критерию."}
        undecided = {**empty, "query_id": "u", "query": "Другой вопрос", "task_type": "collection",
                     "state": "unjudged", "empty_verification": ""}
        tasks = [known, empty, undecided]
        task_path = self.root / "tasks.jsonl"

        def write_tasks():
            task_path.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")

        write_tasks()
        qfile = self.invoke("queries", "owner-queries.jsonl")
        dataset = {"documents": 16, "queries": 3, "qrels": 0, "qrels_file": None, "server_corpus_check": "sha256",
                   "sha256": {"documents.jsonl": self.sha, "queries": hashlib.sha256(qfile.read_bytes()).hexdigest()}}
        ranks = [
            {"q": ["d1", "d12"], "e": ["d2"], "u": []},
            {"q": ["d12"], "e": [], "u": []},
        ]
        for filename, per_query in zip(("before.json", "after.json"), ranks, strict=True):
            predictions = {key: {"ranked_ids": ds, "relevant": None} for key, ds in per_query.items()}
            self.dump(filename, {"dataset": dataset, "timing": {"top_k": 10}, "queries": predictions})
        pool = self.invoke("pool", "owner-pool.json", "--queries-file", "owner-queries.jsonl",
                           "--predictions", "before.json", "after.json")
        assessed = json.loads(pool.read_bytes())
        for t in assessed["tasks"]:
            for c in t["candidates"]:
                c["relevance"] = 2 if c["doc_id"] == "d12" else 0
        self.dump("consensus.json", assessed)
        self.invoke("qrels", "judged.jsonl", "--pool", pool.name, "--consensus", "consensus.json")
        self.invoke("metadata", "meta.jsonl", "--qrels", "judged.jsonl")
        meta = {r["query_id"]: r for r in map(json.loads, (self.root / "meta.jsonl").read_bytes().splitlines())}
        self.assertEqual((meta["q"]["source"], meta["q"]["target_ids"]), ("owner-a", ["d12"]))
        self.assertEqual(meta["q"]["owner_id"], "owner-a")
        self.assertEqual(meta["q"]["synthetic_user"], known["synthetic_user"])
        self.assertEqual(meta["q"]["query_date"], known["query_date"])
        self.assertEqual(meta["u"]["state"], "unjudged")
        paired = subprocess.run(
            [sys.executable, str(CLI.with_name("validate_rankings.py")), "--before", "before.json",
             "--after", "after.json", "--queries", qfile.name, "--qrels", "judged.jsonl",
             "--tasks", "meta.jsonl", "--output", "paired.json"], cwd=self.root, capture_output=True, text=True,
        )
        self.assertEqual(paired.returncode, 0, paired.stderr)
        summary = json.loads((self.root / "paired.json").read_bytes())["primary"]["all"]
        self.assertEqual(summary["states"], {"positive": 1, "verified_empty": 1, "unjudged": 1})
        self.assertGreater(summary["paired_ndcg_at_10"]["delta"], 0)
        self.assertEqual(summary["metrics"]["false_nonempty_rate"]["events_before"], 1)
        self.assertEqual(summary["metrics"]["false_nonempty_rate"]["events_after"], 0)

        empty["empty_verification"] = ""
        write_tasks()
        self.invoke("metadata", "unconfirmed", "--qrels", "judged.jsonl", success=False)
        empty["state"] = "unjudged"
        known["known_doc_ids"] = ["d13"]  # Explicit zero judgment.
        write_tasks()
        self.invoke("metadata", "wrong-target", "--qrels", "judged.jsonl", success=False)


if __name__ == "__main__":
    unittest.main()
