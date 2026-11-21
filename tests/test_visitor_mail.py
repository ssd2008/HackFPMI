"""Visitor isolation through the ordinary HTTP routes, using local sample messages and a fake IMAP server."""

import asyncio
import imaplib
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import Mock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from mailsearch import app as api
from mailsearch import mail_transport as transport
from mailsearch import visitor_mail
from mailsearch.private_cache import private_request, text_cache
from mailsearch.store import MailStore
from mailsearch.visitor_mail import Visitors

try:
    from tests.test_mail_transport import FakeIMAP, raw_letter
except ModuleNotFoundError:
    from test_mail_transport import FakeIMAP, raw_letter


ORIGIN = "https://demo.example"
EDIT = {"X-Mailsearch-Visitor": "1"}
DATE = "2026-07-01"


class VisitorTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(TemporaryDirectory()))
        self.now = 1000
        self.visitors = Visitors(None, clock=lambda: self.now, ttl=3600)
        self.regular_rows = [dict(doc_id="vk-doc", synthetic_user="vk-box", doc_dt="2026-05-02", doc="Тема: VK")]
        self.app = FastAPI(exception_handlers=api.app.exception_handlers.copy())
        self.app.router.routes.extend(api.app.routes)
        api.configure_access(self.app, ORIGIN)
        self.enterContext(patch.object(api, "app", self.app))
        state = self.app.state
        state.store = MailStore(self.regular_rows)
        state.corpus_sha256 = "synthetic-fixture"
        state.search_engine = Mock()
        state.public_profile = False
        state.visitors = self.visitors
        self.local = TestClient(self.app, base_url="http://localhost:8766", client=("127.0.0.1", 42000))
        self.addCleanup(self.local.close)
        self.remote = TestClient(self.app, base_url="https://api.example", client=("203.0.113.20", 42001))
        self.addCleanup(self.remote.close)

        def letter(subject, body):
            msg = raw_letter(body)
            msg.replace_header("Subject", subject)
            return msg.as_bytes()

        self.records = {
            "first-app": {2: letter("Библиотека", "Получите читательский билет."),
                          5: letter("Обсерватория", "Оплата заказа SYN-11abcdef22. Сумма заказа 129.00 RUB.")},
            "second-app": {5: letter("Керамика", "Обжиг в гончарной мастерской.")},
        }
        records = self.records
        self.connections = []

        class Server(FakeIMAP):
            def login(self, address, password):
                if password not in records:
                    raise imaplib.IMAP4.error("provider reply " + password)
                self.records = records[password]
                return super().login(address, password)

        def connection(*args, **kwargs):
            server = Server()
            self.connections.append(server)
            return server

        self.enterContext(patch.object(transport.imaplib, "IMAP4_SSL", side_effect=connection))
        self.addCleanup(self.visitors.close)

    def session(self, client=None, headers=EDIT):
        response = (client or self.local).post("/api/visitor/session", json={}, headers=headers)
        self.assertEqual(response.status_code, 200, response.text)
        value = response.json()
        self.assertEqual(value["expires_in"], 3600)
        self.assertNotIn("set-cookie", response.headers)
        return {"Authorization": "Bearer " + value["token"]}

    def connect(self, headers, password):
        address = {"provider": "gmail", "address": "Reader@example.test", "password": password}
        return self.local.post("/api/visitor/accounts", headers={**headers, **EDIT}, json=address)

    def synced(self, headers):
        end = monotonic() + 10
        while monotonic() < end:
            values = self.local.get("/api/visitor/accounts", headers=headers).json()["accounts"]
            if values and all(v["status"] != "syncing" for v in values):
                return values[0]
            sleep(.02)
        self.fail("The fake inbox did not finish syncing")

    def test_ownership_for_every_mail_route_including_identical_address_and_uid(self):
        first, second = self.session(), self.session()
        self.assertNotEqual(first, second)
        for owner, password in [(first, "first-app"), (second, "second-app")]:
            self.assertEqual(self.connect(owner, password).status_code, 200)
        a, b = self.synced(first), self.synced(second)
        self.assertEqual((a["count"], b["count"], a["address"], a["total_available"]),
                         (2, 1, b["address"], 2))
        self.assertNotEqual(a["id"], b["id"])
        params = {"mailbox": a["id"], "date": DATE}
        doc = self.local.get("/api/inbox", params=params, headers=first).json()["items"][0]["id"]
        bdocs = self.local.get("/api/inbox", params={**params, "mailbox": b["id"]}, headers=second).json()["items"]
        self.assertNotEqual(doc, bdocs[0]["id"])
        self.assertEqual(self.local.get("/api/mail/" + doc, params=params, headers=first).status_code, 200)
        other_box = {**params, "mailbox": b["id"]}
        self.assertEqual(self.local.get("/api/mail/" + doc, params=other_box, headers=second).status_code, 404)
        body = {**params, "query": "Обсерватория"}
        for outsider in ({}, second):
            for path in ["/api/inbox", "/api/actions", "/api/reminder-candidates", "/api/mail/" + doc]:
                self.assertEqual(self.local.get(path, params=params, headers=outsider).status_code, 404, path)
            for path, payload in [("/api/search", body), ("/api/answer", body), ("/api/assistant", body),
                                  ("/api/assistant/watch", {**body, "after": DATE}),
                                  ("/api/actions", {**params, "marked": [doc]})]:
                self.assertEqual(self.local.post(path, json=payload, headers=outsider).status_code, 404, path)
        self.assertEqual(self.local.post("/api/visitor/accounts/" + a["id"] + "/sync", json={},
                                         headers={**second, **EDIT}).status_code, 404)
        for owner, allowed in [({}, set()), (first, {a["id"]}), (second, {b["id"]})]:
            listing = self.local.get("/api/mailboxes", headers=owner).json()
            self.assertEqual({box["id"] for box in listing["mailboxes"]}, {"vk-box"} | allowed)
        self.assertEqual(self.local.post("/api/local/send", json={}, headers={**first, **EDIT}).status_code, 404)

    def test_genuine_search_and_citation_open_on_the_private_memory_index(self):
        from mailsearch.dense import DenseIndex
        try:
            self.visitors.encoder = DenseIndex(self.regular_rows, self.directory / "synthetic-index")
        except RuntimeError as exc:
            if "Отсутствуют локальные веса" in str(exc):
                self.skipTest(str(exc))
            raise
        files_before = {p.relative_to(self.directory) for p in self.directory.rglob("*")}
        headers = self.session()
        self.assertEqual(self.connect(headers, "first-app").status_code, 200)
        account = self.synced(headers)["id"]
        params = {"mailbox": account, "date": DATE}
        query = {**params, "query": "Какова сумма заказа SYN-11abcdef22?"}
        hits = self.local.post("/api/search", json=query, headers=headers).json()["items"]
        self.assertEqual(hits[0]["subject"], "Обсерватория")
        self.assertEqual(self.local.post("/api/search", json={**query, "date": "2026-01-01"},
                                         headers=headers).json()["items"], [])
        answer = self.local.post("/api/answer", json=query, headers=headers).json()
        self.assertIn("129", answer["answer"])
        self.assertEqual([s["id"] for s in answer["sources"]], [hits[0]["id"]])
        opened = self.local.get("/api/mail/" + hits[0]["id"], params=params, headers=headers).json()
        self.assertIn("129.00 RUB", opened["body"])
        self.assertEqual(files_before, {p.relative_to(self.directory) for p in self.directory.rglob("*")})
        self.assertTrue(all(("select", "INBOX", True) in c.calls for c in self.connections))

    def test_revocation_timeout_and_logout_while_a_fetch_is_in_flight(self):
        headers = self.session()
        session = self.visitors.resolve(headers["Authorization"].split()[1])
        entered, release = Event(), Event()
        fetch = transport.fetch_inbox

        def delayed(*args, **kwargs):
            entered.set()
            release.wait(5)
            return fetch(*args, **kwargs)

        with patch.object(transport, "fetch_inbox", side_effect=delayed):
            self.assertEqual(self.connect(headers, "first-app").status_code, 200)
            self.assertTrue(entered.wait(2))
            item = next(iter(session.accounts.values()))
            result = self.local.post("/api/visitor/logout", headers={**headers, **EDIT}, json={})
            self.assertEqual(result.json(), {"ok": True})
            self.assertEqual((session.accounts, item.password), ({}, ""))
            release.set()
        self.visitors.workers.shutdown(wait=True)
        self.assertFalse(item.rows)
        self.assertEqual(self.local.get("/api/mailboxes", headers=headers).status_code, 401)
        other = self.session()
        self.now += 3600
        expired = self.local.get("/api/visitor/accounts", headers=other)
        self.assertEqual(expired.status_code, 401)
        self.assertFalse(self.visitors.sessions)
        self.assertEqual(self.visitors.row_count, 0)

    def test_cross_origin_tls_request_headers_limits_and_password_redaction(self):
        self.app.state.public_profile = True
        origin = {"Origin": ORIGIN}
        path = "/api/visitor/session"
        preflight = self.remote.options(path, headers={**origin, "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Authorization, X-Mailsearch-Visitor, Content-Type"})
        self.assertEqual(preflight.status_code, 200)
        self.assertIn("x-mailsearch-visitor", preflight.headers["access-control-allow-headers"].lower())
        self.assertNotIn("access-control-allow-credentials", preflight.headers)
        self.assertIn("no-store", preflight.headers["cache-control"])
        self.assertEqual(self.remote.post(path, json={}, headers=origin).status_code, 403)
        self.assertEqual(self.remote.post(path, content="{}", headers={**origin, **EDIT}).status_code, 403)
        self.assertEqual(self.remote.post(path, json={}, headers={**EDIT, "Origin": "https://other.example"}).status_code, 403)
        self.assertEqual(self.remote.get("/api/visitor/accounts", headers=origin).headers["access-control-allow-origin"], ORIGIN)
        headers = self.session(self.remote, {**origin, **EDIT})
        self.assertTrue(self.remote.get("/api/mailboxes").json()["visitor_mail"])
        self.assertEqual([box["id"] for box in self.remote.get("/api/mailboxes", headers=headers).json()["mailboxes"]], ["vk-box"])
        self.assertEqual(self.remote.get("/api/inbox", params={"mailbox": "archive-box", "date": DATE}).status_code, 404)
        self.assertEqual(self.remote.get("/api/visitor/accounts?token=" + headers["Authorization"].split()[1]).status_code, 401)
        self.assertEqual(self.remote.get("/api/visitor/accounts", headers={"Cookie": "session=fake"}).status_code, 401)
        bad = {"address": "bad", "password": "do-not-echo", "provider": "gmail"}
        result = self.remote.post("/api/visitor/accounts", headers={**origin, **headers, **EDIT}, json=bad)
        self.assertEqual(result.status_code, 422)
        self.assertNotIn("do-not-echo", result.text)
        for _ in range(6):
            wrong = self.connect(headers, "do-not-echo")
            self.assertEqual(wrong.status_code, 403)
            self.assertNotIn("do-not-echo", wrong.text)
        self.assertEqual(self.connect(headers, "first-app").status_code, 429)
        independent = self.session()
        self.assertEqual(self.connect(independent, "first-app").status_code, 200)
        self.assertEqual(self.synced(independent)["count"], 2)
        self.assertEqual(self.remote.get("/api/mailboxes", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        self.assertEqual(self.remote.get("/api/mailboxes").headers["cache-control"], "private, no-store")
        self.app.state.public_profile = False
        forwarded = self.local.get("/api/mailboxes", headers={"X-Forwarded-For": "203.0.113.20"}).json()
        self.assertFalse(forwarded["visitor_mail"])
        self.assertEqual([box["id"] for box in forwarded["mailboxes"]], ["vk-box"])
        self.assertEqual(self.remote.post(path, json={}, headers={**EDIT, **origin}).status_code, 404)

    def test_capacity_released_on_logout_and_server_restart(self):
        self.visitors.capacity = 1
        first = self.session()
        self.assertEqual(self.local.post("/api/visitor/session", json={}, headers=EDIT).status_code, 429)
        self.assertEqual(self.connect(first, "first-app").status_code, 200)
        self.synced(first)
        self.assertGreater(self.visitors.text_bytes, 0)
        self.local.post("/api/visitor/logout", json={}, headers={**first, **EDIT})
        self.assertEqual((self.visitors.text_bytes, self.visitors.row_count), (0, 0))
        second = self.session()
        self.visitors.close()
        fresh = Visitors(None)
        self.addCleanup(fresh.close)
        self.app.state.visitors = fresh
        self.assertEqual(self.local.get("/api/visitor/accounts", headers=second).status_code, 401)
        self.assertFalse(fresh.sessions)

    def test_worker_honours_process_budget_and_reconciles_expunged_uids(self):
        headers = self.session()
        with patch.object(visitor_mail, "PROCESS_ROWS", 1):
            response = self.connect(headers, "first-app").json()["account"]
            limited = self.synced(headers)
            self.assertEqual((limited["status"], limited["pending"], limited["count"], limited["total_available"]),
                             ("error", 1, 1, 2))
        self.records["first-app"].pop(5)
        self.local.post("/api/visitor/accounts/" + response["id"] + "/sync", json={}, headers={**headers, **EDIT})
        updated = self.synced(headers)
        self.assertEqual((updated["status"], updated["count"], updated["total_available"]), ("idle", 1, 1))
        result = self.local.get("/api/inbox", params={"mailbox": response["id"], "date": DATE}, headers=headers)
        self.assertEqual(result.json()["items"][0]["subject"], "Библиотека")

    def test_public_profile_checks_corpus_bytes_before_parsing(self):
        (self.directory / "documents.jsonl").write_text("not JSON", encoding="utf-8")

        async def start():
            async with api.lifespan(self.app):
                self.fail("Unapproved public snapshot was opened")

        with patch.object(api, "frontend_origin", ORIGIN), patch.object(api, "DATA", self.directory):
            with self.assertRaisesRegex(ValueError, "синтетический корпус"):
                asyncio.run(start())

    def test_private_text_memoization_is_released_after_each_request(self):
        calls = []
        @text_cache(10)
        def process(word):
            calls.append(word)
            return word
        for epoch in range(2):
            with private_request():
                process("личное")
                process("личное")
        process("общее")
        process("общее")
        self.assertEqual(calls, ["личное", "личное", "общее"])

    def test_explicit_session_disable_and_enable_on_loopback(self):
        from mailsearch import __main__ as entry
        with patch.dict("os.environ", {}, clear=True), patch.object(api, "frontend_origin", ""), \
             patch.object(entry.uvicorn, "run") as runner:
            for flag, enabled in [("--visitor-mail", True), ("--no-visitor-mail", False)]:
                with patch("sys.argv", ["mailsearch", "--port", "18792", flag]):
                    entry.main()
                self.assertEqual(self.app.state.visitor_enabled, enabled)
                self.assertEqual(runner.call_args.kwargs["host"], "127.0.0.1")


if __name__ == "__main__":
    unittest.main()
