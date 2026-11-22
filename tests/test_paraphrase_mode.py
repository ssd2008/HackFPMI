from datetime import date
from unittest import IsolatedAsyncioTestCase
from unittest.mock import Mock, patch

import httpx

from mailsearch.app import app
from mailsearch.store import MailStore


class ParaphraseRoutingTests(IsolatedAsyncioTestCase):
    async def test_search_answer_and_request_modes(self):
        documents = [
            dict(doc_id=did, synthetic_user="box", doc_dt="2026-01-01", doc=f"Тема: {did}\nТекст письма: пример")
            for did in ("default", "ordinary", "paraphrase")
        ]
        store = MailStore(documents)
        default, ordinary, paraphrase, attachments = Mock(), Mock(), Mock(), Mock()
        for engine, did in zip((default, ordinary, paraphrase), ("default", "ordinary", "paraphrase")):
            engine.search.return_value = [(did, 1.0)]
        attachments.search.side_effect = lambda box, query, at, limit, hits: (hits, {}, False)
        text = "поездку перенесли на май"
        payload = dict(mailbox="box", date="2026-04-01", query=text)
        state = dict(store=store, search_engine=default, ordinary_engine=ordinary,
                     paraphrase_engine=paraphrase, attachments=attachments)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://local") as client:
            with patch.dict(app.state._state, state):
                for fields, did in (({}, "default"), ({"paraphrase": False}, "ordinary"),
                                    ({"paraphrase": True, "include_attachments": True}, "paraphrase")):
                    response = await client.post("/api/search", json=dict(payload, **fields))
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual([row["id"] for row in response.json()["items"]], [did])
                at = date.fromisoformat(payload["date"])
                default.search.assert_called_once_with("box", text, at, 10)
                ordinary.search.assert_called_once_with("box", text, at, 10)
                paraphrase.search.assert_called_once_with("box", f"Найди {text}", at, 10)
                attachments.search.assert_called_once_with("box", text, at, 10, [("paraphrase", 1.0)])
                for query in ("Найди письмо про поездку домой", "письмо от travel@example.org", "все письма за вчера", "Казань"):
                    response = await client.post("/api/search", json=dict(payload, query=query, paraphrase=True))
                    self.assertEqual(response.status_code, 200)
                    paraphrase.search.assert_called_with("box", query, at, 10)
                with patch("mailsearch.generation.answer_question", return_value={
                    "answer": "пример", "sources": ["paraphrase"], "mode": "extractive",
                }) as answer:
                    response = await client.post("/api/answer", json=dict(payload, paraphrase=True))
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual([x["id"] for x in response.json()["sources"]], ["paraphrase"])
                    paraphrase.search.assert_called_with("box", f"Найди {text}", at, 5)
                    self.assertEqual(answer.call_args.args[0], text)
