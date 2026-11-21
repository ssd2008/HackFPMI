from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
import unittest
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient

from mailsearch import app as api
from mailsearch import mail_transport
from mailsearch.local_mail import LocalMail, LocalMailError
from mailsearch.store import MailStore


class Vault(dict):
    def save(self, key, value):
        self[key] = value


class Provider:
    def __init__(self):
        self.validity = "1"
        self.uids = {1, 2, 3}
        self.sends = []
        self.result = None
        self.entered = Event()
        self.release = Event()
        self.release.set()

    def verify_account(self, provider, address, password):
        if password == "invalid":
            raise mail_transport.AuthenticationError("Проверьте пароль приложения.")

    def fetch_inbox(self, provider, address, password, *, mailbox, known_validity, known_uids):
        known = known_uids if known_validity == self.validity else set()
        missing = sorted(self.uids - known, reverse=True)
        messages = {n: dict(doc_id=f"{mailbox}-{self.validity}-{n}", synthetic_user=mailbox,
                            doc_dt=f"2026-11-{n:02}", doc=f"Тема: Пример {n}\n\nТекст {n}")
                    for n in missing[:2]}
        return mail_transport.SyncBatch(self.validity, self.uids.copy(), messages, len(missing) - len(messages))

    def send_message(self, provider, address, password, **letter):
        self.sends.append(letter)
        self.entered.set()
        assert self.release.wait(4)
        if self.result:
            raise self.result


class LocalMailTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.vault = Vault()
        self.provider = Provider()
        self.local = LocalMail(self.root, credentials=self.vault, remote=self.provider)
        self.addCleanup(lambda: self.local.close())

    def connected(self, provider="gmail"):
        row = self.local.connect(provider, "reader@" + provider + ".example", "fake-app-secret")
        self.wait_sync()
        return row["id"]

    def wait_sync(self):
        deadline = monotonic() + 5
        while any(a["status"] == "syncing" for a in self.local.accounts()):
            self.assertLess(monotonic(), deadline)
            sleep(.01)

    def test_reconcile_pages_removed_messages_and_replaced_inbox(self):
        account = self.connected()
        first = self.local.inbox(account)
        self.assertEqual(len(first.by_id), 3)
        self.assertEqual(first.inbox(account, "2026-11-02", 0, 9)["total"], 2)
        self.provider.uids = {2, 4}
        self.local.sync(account)
        self.wait_sync()
        self.assertEqual(set(self.local.inbox(account).by_id), {account + "-1-2", account + "-1-4"})
        self.provider.validity = "2"
        self.local.sync(account)
        self.wait_sync()
        self.assertEqual(set(self.local.inbox(account).by_id), {account + "-2-2", account + "-2-4"})
        self.provider.uids = set()
        self.local.sync(account)
        self.wait_sync()
        self.assertEqual(self.local.inbox(account).rank("привет", None, 10), [])
        self.assertEqual(self.local.boxes()[0]["count"], 0)
        self.assertIsNone(self.local.inbox("missing"))

    def test_reconnect_and_restart(self):
        account = self.connected("yandex")
        with self.assertRaises(mail_transport.AuthenticationError):
            self.local.connect("yandex", "reader@yandex.example", "invalid")
        self.assertEqual(self.vault[account], "fake-app-secret")
        self.local.close()
        self.local = LocalMail(self.root, credentials=self.vault, remote=self.provider)
        self.assertEqual(len(self.local.inbox(account).by_id), 3)
        self.assertEqual(self.local.accounts()[0]["provider"], "yandex")
        self.assertNotIn(b"fake-app-secret", self.local.database.read_bytes())
        self.assertEqual(self.local.database.stat().st_mode & 0o777, 0o600)

    def test_duplicate_submissions_and_uncertain_delivery(self):
        account = self.connected()
        args = [account, ["reader@example.org"], "Тема", "Письмо", str(uuid4())]
        self.local.send(*args)
        self.local.send(*args)
        self.assertEqual(len(self.provider.sends), 1)
        with self.assertRaises(LocalMailError):
            self.local.send(*[account, args[1], "Иная тема", *args[3:]])
        args[-1] = str(uuid4())
        self.provider.result = mail_transport.DeliveryUncertain("Нет подтверждения.")
        with self.assertRaises(mail_transport.DeliveryUncertain):
            self.local.send(*args)
        with self.assertRaises(LocalMailError):
            self.local.send(*args)
        self.assertEqual(len(self.provider.sends), 2)
        self.provider.result = mail_transport.TransportError("Сервер отклонил письмо.")
        args[-1] = str(uuid4())
        with self.assertRaises(mail_transport.TransportError):
            self.local.send(*args)
        self.provider.result = None
        self.local.send(*args)
        self.assertEqual(len(self.provider.sends), 4)

    def test_double_click_during_delivery_calls_smtp_once(self):
        account = self.connected()
        args = (account, ["reader@example.org"], "Тема", "Проверка", str(uuid4()))
        self.provider.release.clear()
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.local.send, *args)
            try:
                self.assertTrue(self.provider.entered.wait(3))
                with self.assertRaises(LocalMailError) as error:
                    self.local.send(*args)
                self.assertEqual(error.exception.status, 409)
            finally:
                self.provider.release.set()
            pending.result(timeout=3)
        self.local.send(*args)
        self.assertEqual(len(self.provider.sends), 1)

    def test_web_routes_and_public_isolation(self):
        public = MailStore([dict(doc_id="vk-doc", synthetic_user="vk-box", doc_dt="2026-11-01", doc="Тема: VK")])
        with patch.object(api.app.state, "local_mail", self.local, create=True), \
             patch.object(api.app.state, "store", public, create=True), \
             patch.object(api.app.state, "corpus_sha256", "vk-digest", create=True):
            client = TestClient(api.app, base_url="http://localhost:8766", client=("127.0.0.1", 49001))
            self.addCleanup(client.close)
            data = dict(provider="gmail", address="reader@gmail.example", password="fake-app-secret")
            path = "/api/local/accounts"
            headers = {"X-Mailsearch-Local": "1", "Origin": "http://localhost:8766"}
            self.assertEqual(client.post(path, json=data).status_code, 403)
            self.assertEqual(client.get("/api/mailboxes", headers={"Host": "outside.example"}).status_code, 403)
            self.assertEqual(client.get("/api/mailboxes", headers={"X-Forwarded-For": "127.0.0.2"}).status_code, 403)
            self.assertEqual(client.post(path, headers={**headers, "Origin": "https://elsewhere.example"},
                                         json=data).status_code, 403)
            invalid = client.post(path, headers=headers, json={**data, "address": "not an email"})
            self.assertEqual(invalid.status_code, 422)
            self.assertNotIn(data["password"], invalid.text)
            response = client.post(path, json=data, headers=headers)
            self.assertEqual(response.status_code, 200, response.text)
            account = response.json()["account"]["id"]
            self.wait_sync()
            boxes = client.get("/api/mailboxes").json()
            self.assertTrue(boxes["local_mail"])
            self.assertEqual([b["id"] for b in boxes["mailboxes"]], ["vk-box", account])
            inbox = client.get("/api/inbox", params={"mailbox": account, "date": "2026-11-03"}).json()
            self.assertEqual(inbox["total"], 3)
            doc = inbox["items"][0]["id"]
            self.assertEqual(client.get("/api/mail/" + doc, params={"mailbox": "vk-box", "date": "2026-11-03"}).status_code, 404)
            letter = dict(account_id=account, to=["reader@example.org"], subject="", body="Проверка", request_id=str(uuid4()))
            self.assertEqual(client.post("/api/local/send", json=letter, headers=headers).json(), {"status": "sent"})
            self.assertEqual(client.post("/api/local/send", json=letter, headers=headers).status_code, 200)
            self.assertEqual(len(self.provider.sends), 1)
            self.assertEqual(client.post("/api/local/send", json={**letter, "account_id": "vk-box"}, headers=headers).status_code, 404)
            with patch.object(api.app.state, "local_mail", None):
                self.assertEqual(client.get(path).status_code, 404)
                self.assertFalse(client.get("/api/mailboxes").json()["local_mail"])
                self.assertEqual(client.get("/api/inbox", params={"mailbox": account, "date": "2026-11-03"}).status_code, 404)
