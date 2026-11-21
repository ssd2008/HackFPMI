"""Protocol-boundary tests. IMAP and SMTP constructors are always replaced with fakes."""
from email import policy
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import imaplib
import smtplib
import ssl
import unittest
from unittest.mock import Mock, patch

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
            assert query == '(UID INTERNALDATE BODY.PEEK[])'
            if uid in self.vanished:
                del self.records[uid]
                continue
            if uid in self.missing_once:
                self.missing_once.remove(uid)
                continue
            # Exercise both permitted orders of the literal and UID/INTERNALDATE.
            literal = b'BODY[] {' + str(len(body)).encode() + b'}'
            if uid % 2:
                rows.extend([(meta + b'INTERNALDATE ' + self.date + b' ' + literal, body), b')'])
            else:
                rows.extend([(f'{uid + 4} ('.encode() + literal, body),
                             f' UID {uid} INTERNALDATE '.encode() + self.date + b')'])
        return 'OK', rows

    def logout(self):
        self.calls.append(('logout',))


class FakeSMTP:
    def __init__(self, *, implicit_tls=True):
        self.tls = implicit_tls
        self.offers_starttls = True
        self.sock = Mock()
        self.events = []
        self.login_args = None
        self.mail_args = None
        self.destinations = []
        self.codes = []
        self.raw = None
        self.begin_data = 354
        self.reply = 250
        self.closed = False
        self.international = True

    def set_debuglevel(self, n):
        assert n == 0

    def ehlo_or_helo_if_needed(self):
        self.ehlo()

    def ehlo(self):
        self.events.append('ehlo_tls' if self.tls else 'ehlo_plain')
        return 250, b''

    def starttls(self, *, context):
        assert context.check_hostname and context.verify_mode == ssl.CERT_REQUIRED
        self.events.append('starttls')
        self.tls = True
        return 220, b''

    def login(self, login, password):
        assert self.tls
        self.events.append('login')
        self.login_args = login, password

    def has_extn(self, opt):
        if opt == 'starttls':
            return self.offers_starttls
        return self.international and opt == 'smtputf8'

    def mail(self, address, options):
        self.mail_args = address, options
        return 250, b''

    def rcpt(self, address):
        self.destinations.append(address)
        return self.codes.pop(0) if self.codes else 250, b''

    def docmd(self, command):
        assert command == 'DATA'
        return self.begin_data, b''

    def send(self, raw):
        self.raw = raw

    def getreply(self):
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply, b'private content from a provider'

    def close(self):
        self.closed = True


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
        self.smtp = FakeSMTP()
        self.alternate = FakeSMTP(implicit_tls=False)
        self.alternate_factory = self.enterContext(patch.object(transport.smtplib, 'SMTP', return_value=self.alternate))
        self.imap_factory = self.enterContext(patch.object(transport.imaplib, 'IMAP4_SSL', return_value=self.imap))
        self.smtp_factory = self.enterContext(patch.object(transport.smtplib, 'SMTP_SSL', return_value=self.smtp))

    def fetch(self, *, known_validity='7', known_uids=None, batch_size=2, mailbox='local-a'):
        return transport.fetch_inbox('gmail', 'me@example.test', 'test-app-pass', mailbox=mailbox,
                                     known_validity=known_validity, known_uids=known_uids or set(), batch_size=batch_size)

    def send(self, **changes):
        values = dict(to=['Редактор <news+list@example.test>', 'editor@sub.example.test'],
                      subject='Письмо о прототипе', body='Строка\n.\n..после точки', message_id='<local-7@example.test>')
        transport.send_message('yandex', 'test@yandex.ru', 'test-app-pass', **{**values, **changes})

    def assert_tls(self, factory, host, port, key):
        args, kwargs = factory.call_args
        self.assertEqual(args, (host, port))
        self.assertGreater(kwargs['timeout'], 0)
        self.assertTrue(kwargs[key].check_hostname)
        self.assertEqual(kwargs[key].verify_mode, ssl.CERT_REQUIRED)

    def test_verify_authenticates_and_examines_without_loading_messages(self):
        transport.verify_account('gmail', 'me@example.test', 'abcd efgh ijkl mnop')
        self.assert_tls(self.imap_factory, 'imap.gmail.com', 993, 'ssl_context')
        self.assertEqual(self.imap.calls, [('login', 'me@example.test', 'abcdefghijklmnop'), ('select', 'INBOX', True), ('logout',)])
        transport.verify_account('yandex', 'office@hosted.example', 'pass')
        self.assert_tls(self.imap_factory, 'imap.yandex.ru', 993, 'ssl_context')
        self.assertIn(('login', 'office@hosted.example', 'pass'), self.imap.calls)

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

    def test_smtp_envelope_encoded_headers_dot_transparency_and_yandex_login(self):
        self.send()
        self.assert_tls(self.smtp_factory,'smtp.yandex.ru',465,'context')
        self.assertEqual(self.smtp.login_args,('test','test-app-pass'))
        self.assertEqual(self.smtp.mail_args,('test@yandex.ru',()))
        self.assertEqual(self.smtp.destinations,['news+list@example.test','editor@sub.example.test'])
        self.assertTrue(self.smtp.raw.endswith(b'\r\n.\r\n'))
        self.assertIn(b'\r\n..\r\n...', self.smtp.raw)
        unstuffed = self.smtp.raw[:-3].replace(b'\r\n..', b'\r\n.')
        msg = BytesParser(policy=policy.default).parsebytes(unstuffed)
        self.assertEqual(str(msg['Subject']),'Письмо о прототипе')
        self.assertEqual(str(msg['Message-ID']),'<local-7@example.test>')
        self.assertIsNotNone(parsedate_to_datetime(str(msg['Date'])).tzinfo)
        self.assertEqual(msg['To'].addresses[0].display_name,'Редактор')
        self.assertIn('.\r\n..после точки',msg.get_content())
        self.assertTrue(self.smtp.closed)
        self.alternate_factory.assert_not_called()

    def test_short_initial_wait_then_starttls_before_authentication_on_both_providers(self):
        self.smtp_factory.side_effect = TimeoutError()
        for provider in ('gmail', 'yandex'):
            with self.subTest(provider=provider):
                alternate = FakeSMTP(implicit_tls=False)
                self.alternate_factory.return_value = alternate
                transport.send_message(provider, 'sender@example.test', 'test-password', to=['reader@example.test'],
                                       subject='Обновления', body='Тест', message_id='round@example.test')
                host = 'smtp.gmail.com' if provider == 'gmail' else 'smtp.yandex.ru'
                self.assert_tls(self.smtp_factory, host, 465, 'context')
                self.assertLess(self.smtp_factory.call_args.kwargs['timeout'], self.alternate_factory.call_args.kwargs['timeout'])
                self.assertEqual(self.alternate_factory.call_args.args, (host, 587))
                self.assertEqual(alternate.events, ['ehlo_plain', 'starttls', 'ehlo_tls', 'login'])
                alternate.sock.settimeout.assert_called_once_with(transport._TIMEOUT)
                self.assertTrue(alternate.closed)

    def test_unreachable_ports_report_connection_failure_without_authentication(self):
        self.smtp_factory.side_effect = TimeoutError('hidden detail')
        self.alternate_factory.side_effect = ConnectionRefusedError('hidden detail')
        with self.assertRaises(transport.TransportError) as failure:
            self.send()
        self.assertIs(type(failure.exception), transport.TransportError)
        self.assertIn('465 и 587', str(failure.exception))
        self.assertNotIn('hidden', str(failure.exception))
        self.assertIsNone(self.smtp.login_args)
        self.assertIsNone(self.alternate.login_args)

    def test_pre_authentication_disconnect_releases_primary_before_fallback(self):
        with patch.object(self.smtp, 'ehlo_or_helo_if_needed', side_effect=smtplib.SMTPServerDisconnected()):
            self.send()
        self.assertIsNone(self.smtp.login_args)
        self.assertTrue(self.smtp.closed)
        self.assertEqual(self.alternate.events[-1], 'login')

    def test_optional_or_invalid_tls_never_receives_credentials(self):
        self.smtp_factory.side_effect = ConnectionRefusedError()
        self.alternate.offers_starttls = False
        with self.assertRaises(transport.TransportError):
            self.send()
        self.assertEqual(self.alternate.events, ['ehlo_plain'])
        self.assertIsNone(self.alternate.login_args)
        self.assertTrue(self.alternate.closed)
        self.alternate.offers_starttls = True
        with patch.object(self.alternate, 'starttls', side_effect=ssl.SSLCertVerificationError('private details')):
            with self.assertRaises(transport.TransportError) as error:
                self.send()
        self.assertNotIn('private', str(error.exception))
        self.assertIsNone(self.alternate.login_args)
        self.assertIsNone(self.alternate.raw)

    def test_primary_certificate_or_auth_failure_never_retries_on_another_port(self):
        self.smtp_factory.side_effect = ssl.SSLCertVerificationError()
        with self.assertRaises(transport.TransportError):
            self.send()
        self.smtp_factory.side_effect = None
        with patch.object(self.smtp, 'login', side_effect=smtplib.SMTPServerDisconnected('private details')):
            with self.assertRaises(transport.TransportError):
                self.send()
        self.alternate_factory.assert_not_called()

    def test_recipients_are_accepted_as_one_transaction(self):
        self.smtp.codes = [250,550]
        with self.assertRaises(transport.TransportError) as failure:
            self.send()
        self.assertNotIsInstance(failure.exception,transport.DeliveryUncertain)
        self.assertIsNone(self.smtp.raw)
        self.assertTrue(self.smtp.closed)

    def test_failure_before_body_definite_after_body_uncertain_unless_rejected(self):
        for stage, code, expected in [('begin_data', 421, transport.TransportError),('reply', 553, transport.TransportError),
                                     ('reply', smtplib.SMTPServerDisconnected('secret'), transport.DeliveryUncertain),
                                     ('reply', -1, transport.DeliveryUncertain)]:
            with self.subTest(stage=stage,code=repr(code)):
                self.smtp.begin_data = 354
                self.smtp.reply = 250
                self.smtp.raw = None
                setattr(self.smtp,stage,code)
                with self.assertRaises(expected) as error:
                    self.send()
                self.assertIs(type(error.exception),expected)
                self.assertNotIn('secret',str(error.exception))
                self.assertNotIn('private',str(error.exception))
                self.assertEqual(self.smtp.raw is None,stage=='begin_data')
        self.alternate_factory.assert_not_called()

    def test_reject_header_injection_before_any_connection(self):
        for kwargs in (dict(subject='subject\r\nBcc: x@example.test'),dict(to=['a@example.test\nCc: x@example.test']),
                       dict(message_id='<id@example.test>\nBcc: x'),dict(to=['a@example.test, b@example.test']),
                       dict(to=['not-an-address'])):
            with self.subTest(kwargs=kwargs),self.assertRaises(transport.TransportError):
                self.send(**kwargs)
        self.smtp_factory.assert_not_called()
        with self.assertRaises(transport.TransportError):
            transport.verify_account('unlisted', 'user@example.test', 'pass')
        self.imap_factory.assert_not_called()

    def test_international_domain_encoded_and_unicode_mailbox_negotiated(self):
        self.send(to=['Олег <name@пример.рф>', 'm+case@example.test'])
        self.assertEqual(self.smtp.destinations[0], 'name@xn--e1afmkfd.xn--p1ai')
        self.send(to=['пользователь@example.test'])
        self.assertEqual(self.smtp.mail_args[1], ('SMTPUTF8',))
        self.smtp.international = False
        with self.assertRaises(transport.TransportError):
            self.send(to=['пользователь@example.test'])

    def test_authentication_and_network_errors_are_sanitized(self):
        with patch.object(self.imap,'login',side_effect=imaplib.IMAP4.error('private-user private-password')):
            with self.assertRaises(transport.AuthenticationError) as error:
                self.fetch()
            self.assertNotIn('private-', str(error.exception))
        with patch.object(self.smtp,'login',side_effect=smtplib.SMTPAuthenticationError(535,b'private-password')):
            with self.assertRaises(transport.AuthenticationError) as error:
                self.send()
            self.assertNotIn('private-', str(error.exception))


if __name__ == '__main__':
    unittest.main()
