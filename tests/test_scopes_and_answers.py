"""Integration checks that the development relevance labels alone cannot cover.

Start the service first. MAILSEARCH_URL can point to another local port.
"""

from collections import Counter
from datetime import date, timedelta
import json
import os
import re
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mailsearch.store import DATA, load_jsonl


def api(route, *, params=None, post=None):
    base = os.environ.get("MAILSEARCH_URL", "http://127.0.0.1:8765")
    url = base + route + ("?" + urlencode(params) if params else "")
    data = json.dumps(post).encode() if post is not None else None
    request = Request(url, data=data, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=90) as response:
        return json.load(response)


class ScopeAndAnswerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docs = load_jsonl(DATA / "documents.jsonl")
        cls.boxes = Counter(d["synthetic_user"] for d in cls.docs)

    def test_future_and_foreign_letters_are_unavailable_even_by_direct_id(self):
        doc = next(d for d in self.docs if d["doc_dt"])
        other = next(box for box in self.boxes if box != doc["synthetic_user"])
        prior = (date.fromisoformat(doc["doc_dt"]) - timedelta(days=1)).isoformat()
        own = {"mailbox": doc["synthetic_user"], "date": doc["doc_dt"], "query": doc["doc_id"]}
        self.assertEqual([d["id"] for d in api("/api/search", post=own)["items"]], [doc["doc_id"]])
        for box, day in ((other, doc["doc_dt"]), (doc["synthetic_user"], prior)):
            context = {"mailbox": box, "date": day}
            with self.subTest(context=context):
                with self.assertRaises(HTTPError) as error:
                    api("/api/mail/" + doc["doc_id"], params=context)
                self.assertEqual(error.exception.code, 404)
                request = {**context, "query": doc["doc_id"]}
                self.assertEqual(api("/api/search", post=request)["items"], [])
                answer = api("/api/answer", post=request)
                self.assertEqual(answer["sources"], [])
                self.assertEqual(answer["mode"], "no_evidence")

    def test_unknown_receipt_date_visible_in_inbox_and_excluded_from_day_query(self):
        doc = next(d for d in self.docs if d["doc_dt"] is None)
        context = {"mailbox": doc["synthetic_user"], "date": "1900-01-02"}
        counts = sum(d["synthetic_user"] == context["mailbox"] and d["doc_dt"] is None for d in self.docs)
        inbox = api("/api/inbox", params=context)
        self.assertEqual(inbox["total"], counts)
        self.assertTrue(all(r["date"] is None for r in inbox["items"]))
        self.assertEqual(api("/api/mail/" + doc["doc_id"], params=context)["body"], doc["doc"])
        self.assertEqual(api("/api/search", post={**context, "query": "Какие письма пришли вчера?"})["items"], [])

    def test_rag_answers_amount_and_cites_the_original(self):
        context = {"mailbox": "box-f9410417c091", "date": "2027-12-19"}
        query = "Какова сумма заказа SYN-077f40913b?"
        result = api("/api/answer", post={**context, "query": query})
        self.assertIn(result["mode"], {"generative", "extractive"})
        self.assertRegex(result["answer"], r"431[,.]03")
        markers = [int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])]
        self.assertTrue(markers)
        self.assertTrue(all(1 <= n <= len(result["sources"]) for n in markers))
        original = api("/api/mail/" + result["sources"][markers[0] - 1]["id"], params=context)
        self.assertIn("SYN-077f40913b", original["body"])
        self.assertIn("431.03 RUB", original["body"])

    def test_absent_identifier_does_not_produce_an_answer(self):
        request = {"mailbox": next(iter(self.boxes)), "date": "2030-01-01",
                   "query": "Когда пришёл заказ SYN-00000000000000?"}
        self.assertEqual(api("/api/search", post=request)["items"], [])
        answer = api("/api/answer", post=request)
        self.assertEqual(answer["mode"], "no_evidence")
        self.assertEqual(answer["sources"], [])


if __name__ == "__main__":
    unittest.main()
