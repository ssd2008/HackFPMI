"""Protocol-boundary tests. The IMAP connection is replaced with a fake."""
from email import policy
from email.message import EmailMessage
import imaplib
import os
import ssl
import unittest
from unittest.mock import patch

from mailsearch import mail_transport as transport
from mailsearch.store import describe


class FakeIMAP:
    def __init__(self, records=None):
        self.records = records or {}
        self.epoch = b"7"
        self.calls = []
        self.date = b'"01-Jul-2026 01:10:00 +0300"'
        self.missing_once = set()
        self.vanished = set()

    def login(self, address, password):
        self.calls.append(("login", address, password))
        return "OK", []

    def select(self, folder, *, readonly):
        self.calls.append(("select", folder, readonly))
        return "OK", [str(len(self.records)).encode()]

    def response(self, item):
        return item, [self.epoch]

    def fetch(self, sequence, query):
        assert query == "(UID)"
        self.calls.append(("sequence", sequence, query))
        first, last = map(int, sequence.split(":"))
        return "OK", [f"{i} (UID {uid})".encode() for i, uid in enumerate(sorted(self.records), 1)
                      if first <= i <= last]

    def uid(self, command, ids, query):
        self.calls.append((command, ids, query))
        if command == 'SEARCH':
            return 'OK', [b' '.join(str(i).encode() for i in sorted(self.records))]
        subset = [int(i) for i in ids.split(',')]
        rows = []
        for uid in subset:
            if uid not in self.records:
                continue
            meta = f'{uid + 4} (UID {uid} '.encode()
            body = self.records[uid]
            if query == '(UID RFC822.SIZE)':
                rows.append(meta + f'RFC822.SIZE {len(body)})'.encode())
                continue
            partial = query == f'(UID INTERNALDATE BODY.PEEK[]<0.{transport.MAX_MIME_BYTES}>)'
            assert partial or query == '(UID INTERNALDATE BODY.PEEK[])'
            if partial:
                body = body[:transport.MAX_MIME_BYTES]
            if uid in self.vanished:
                del self.records[uid]
                continue
            if uid in self.missing_once:
                self.missing_once.remove(uid)
                continue
            # Exercise both permitted orders of the literal and UID/INTERNALDATE.
            literal = (b'BODY[]<0> {' if partial else b'BODY[] {') + str(len(body)).encode() + b'}'
            if uid % 2:
                rows.extend([(meta + b'INTERNALDATE ' + self.date + b' ' + literal, body), b')'])
            else:
                rows.extend([(f'{uid + 4} ('.encode() + literal, body),
                             f' UID {uid} INTERNALDATE '.encode() + self.date + b')'])
        return 'OK', rows

    def logout(self):
        self.calls.append(('logout',))


def raw_letter(body='Проверка почты'):
    msg = EmailMessage(policy=policy.SMTP)
    msg['From'] = 'Ольга <sender@example.test>'
    msg['Subject'] = 'Встреча команды'
    msg['Date'] = 'Mon, 01 Jan 2024 15:00:00 +0000'
    msg.set_content(body)
    return msg


class MailTransportTests(unittest.TestCase):
    def setUp(self):
        self.imap = FakeIMAP({n: raw_letter(f'Номер {n}').as_bytes() for n in (2, 5, 9, 11)})
        self.imap_factory = self.enterContext(patch.object(transport.imaplib, 'IMAP4_SSL', return_value=self.imap))

    def fetch(self, *, known_validity='7', known_uids=None, batch_size=2, mailbox='local-a'):
        return transport.fetch_inbox('yandex', 'me@yandex.ru', 'test-app-pass', mailbox=mailbox,
                                     known_validity=known_validity, known_uids=known_uids or set(), batch_size=batch_size)

    def assert_tls(self, factory, host, port, key):
        args, kwargs = factory.call_args
        self.assertEqual(args, (host, port))
        self.assertGreater(kwargs['timeout'], 0)
        self.assertTrue(kwargs[key].check_hostname)
        self.assertEqual(kwargs[key].verify_mode, ssl.CERT_REQUIRED)

    def test_verify_authenticates_and_examines_without_loading_messages(self):
        transport.verify_account('yandex', 'me@yandex.ru', 'test-app-pass')
        self.assert_tls(self.imap_factory, 'imap.yandex.ru', 993, 'ssl_context')
        self.assertEqual(self.imap.calls, [('login', 'me', 'test-app-pass'), ('select', 'INBOX', True), ('logout',)])
        transport.verify_account('yandex', 'office@hosted.example', 'pass')
        self.assert_tls(self.imap_factory, 'imap.yandex.ru', 993, 'ssl_context')
        self.assertIn(('login', 'office@hosted.example', 'pass'), self.imap.calls)

    @patch.dict(os.environ, {"SSL_CERT_FILE": os.devnull, "SSL_CERT_DIR": os.devnull})
    def test_explicit_ca_bundle_when_python_has_no_default_roots(self):
        transport.verify_account("yandex", "me@yandex.ru", "test-app-pass")
        self.assert_tls(self.imap_factory, "imap.yandex.ru", 993, "ssl_context")
        context = self.imap_factory.call_args.kwargs["ssl_context"]
        self.assertGreater(context.cert_store_stats()["x509_ca"], 0)

    def test_newest_first_then_background_batches_and_epoch_change(self):
        first = self.fetch(known_uids={5})
        self.assertEqual((first.validity, first.current_uids, set(first.messages), first.remaining),
                         ('7', {2,5,9,11}, {11,9}, 1))
        self.assertIn(('FETCH', '11,9', '(UID INTERNALDATE BODY.PEEK[])'), self.imap.calls)
        second = self.fetch(known_uids={5,9,11})
        self.assertEqual((set(second.messages), second.remaining), ({2}, 0))
        done = self.fetch(known_uids=first.current_uids)
        self.assertEqual(done.messages, {})
        original = first.messages[11]
        self.assertEqual((original['synthetic_user'], original['doc_dt']), ('local-a','2026-06-30'))
        self.assertEqual(describe(original)['subject'], 'Встреча команды')
        self.assertEqual(original, self.fetch().messages[11])
        self.assertNotEqual(original['doc_id'], self.fetch(mailbox='local-b').messages[11]['doc_id'])
        self.imap.epoch = b'8'
        replacement = self.fetch(known_uids=first.current_uids)
        self.assertEqual(set(replacement.messages), {9,11})
        self.assertNotEqual(original['doc_id'], replacement.messages[11]['doc_id'])

    def test_expunged_and_nonreturned_are_distinguished(self):
        self.imap.vanished = {11}
        batch = self.fetch()
        self.assertEqual((batch.current_uids, set(batch.messages), batch.remaining), ({2,5,9},{9},2))
        self.imap.missing_once = {9}
        with self.assertRaises(transport.TransportError):
            self.fetch()
        self.assertIn(9,self.fetch().messages)
        self.imap.epoch = None
        with self.assertRaises(transport.TransportError):
            self.fetch()

    def test_plain_and_html_headers_charset_and_internal_date_fallback(self):
        msg = raw_letter('Прямой текст')
        msg.add_alternative('<p>Лишняя версия</p>', subtype='html')
        msg.add_attachment(b'attachment-data', maintype='application', subtype='octet-stream', filename='doc.bin')
        only_html = raw_letter(' ')
        only_html.add_alternative('<html><head><title>Заголовок сайта</title></head><p>Отчёт &amp; план</p><script>hidden()</script><p>готов</p></html>', subtype='html', charset='windows-1251')
        self.imap.records = {1:msg.as_bytes(),2:only_html.as_bytes()}
        self.imap.date = b'"bad"'
        batch = self.fetch()
        plain, html = batch.messages[1], batch.messages[2]
        self.assertIn('Ольга <sender@example.test>', plain['doc'])
        self.assertIn('Прямой текст', plain['doc'])
        for extra in ('attachment-data','Лишняя версия'):
            self.assertNotIn(extra, plain['doc'])
        self.assertIn('Отчёт & план\nготов', html['doc'])
        self.assertNotIn('hidden', html['doc'])
        self.assertNotIn('Заголовок сайта', html['doc'])
        self.assertEqual(html['doc_dt'], '2024-01-01')
        self.assertIsNone(transport._day('bad','bad'))

    def test_named_inline_attachment_before_the_plain_body_is_skipped(self):
        msg = EmailMessage()
        msg.make_mixed()
        attachment = raw_letter('Текст из вложения')
        attachment.add_header('Content-Disposition', 'inline', filename='attachment.txt')
        msg.attach(attachment)
        msg.attach(raw_letter('Основной текст'))
        self.imap.records = {21: msg.as_bytes()}
        document = self.fetch().messages[21]['doc']
        self.assertIn('Основной текст', document)
        self.assertNotIn('Текст из вложения', document)

    def test_large_uid_enumeration_avoids_a_single_oversized_search_line(self):
        with patch.object(self.imap, 'uid', return_value=('OK', [b'1 (UID 20)', b'2 (UID 90)'])) as command:
            self.assertEqual(transport._uid_set(self.imap, 70000), {20,90})
            command.assert_called_once_with('FETCH', '1:*', '(UID)')
        with patch.object(self.imap, 'uid', return_value=('OK', [None])):
            with self.assertRaises(transport.TransportError):
                transport._uid_set(self.imap, 70000)

    def test_authentication_and_network_errors_are_sanitized(self):
        with patch.object(self.imap,'login',side_effect=imaplib.IMAP4.error('private-user private-password')):
            with self.assertRaises(transport.AuthenticationError) as error:
                self.fetch()
            self.assertNotIn('private-', str(error.exception))

    def test_gmail_is_rejected_before_opening_a_connection(self):
        with self.assertRaises(transport.TransportError):
            transport.verify_account("gmail", "reader@example.test", "test-app-pass")
        self.imap_factory.assert_not_called()

    def test_capped_window_uses_only_recent_uids_and_never_fetches_a_whole_oversize_message(self):
        msg = raw_letter("Начало письма. " + "x" * 1000)
        msg.add_attachment(b"z" * 1024, maintype="application", subtype="octet-stream", filename="unused.bin")
        self.imap.records[15] = msg.as_bytes()
        with patch.object(transport, "MAX_MIME_BYTES", 1500), patch.object(transport, "MAX_BODY_CHARS", 60):
            batch = transport.fetch_inbox("yandex", "reader@yandex.ru", "fake", mailbox="account-fixture",
                                         known_validity="7", known_uids={9}, max_messages=3)
        self.assertEqual((batch.current_uids, set(batch.messages), batch.total_available, batch.remaining),
                         ({9, 11, 15}, {11, 15}, 5, 0))
        self.assertIn(("sequence", "3:5", "(UID)"), self.imap.calls)
        self.assertIn(("FETCH", "15", "(UID INTERNALDATE BODY.PEEK[]<0.1500>)"), self.imap.calls)
        self.assertTrue(batch.messages[15]["truncated"])
        self.assertIn("Начало письма", batch.messages[15]["doc"])
        self.assertNotIn("x" * 61, batch.messages[15]["doc"])


if __name__ == "__main__":
    unittest.main()
