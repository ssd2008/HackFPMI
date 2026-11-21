"""TLS-only IMAP and SMTP for locally connected Gmail and Yandex accounts."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from email import errors, policy
from email.headerregistry import Address
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import format_datetime, parsedate_to_datetime
import hashlib
import heapq
from html.parser import HTMLParser
import imaplib
import re
import smtplib
import ssl


_HOSTS = {"gmail": ("imap.gmail.com", "smtp.gmail.com"),
          "yandex": ("imap.yandex.ru", "smtp.yandex.ru")}
_TIMEOUT = 30
_SMTP_PRIMARY_TIMEOUT = 3
_SMTP_FALLBACK_TIMEOUT = 10
_UID = re.compile(rb"\bUID (\d+)\b", re.I)
_SIZE = re.compile(rb"\bRFC822\.SIZE (\d+)\b", re.I)
_DATE = re.compile(rb'\bINTERNALDATE "([^"]+)"', re.I)
_HEADER_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_DOMAIN_LABEL = re.compile(r"[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?", re.I | re.ASCII)
_MSG_ID = re.compile(r"[a-z\d!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z\d!#$%&'*+/=?^_`{|}~-]+)*@[a-z\d-]+(?:\.[a-z\d-]+)*", re.I | re.ASCII)


class TransportError(Exception):
    """Safe for display. An SMTP transaction in this state has not been accepted."""


class AuthenticationError(TransportError):
    """The provider rejected the supplied credentials."""


class DeliveryUncertain(TransportError):
    """The provider may have accepted the body. Do not automatically resubmit."""


@dataclass
class SyncBatch:
    validity: str
    current_uids: set[int]
    messages: dict[int, dict]
    remaining: int


def parse_address(value: str) -> Address:
    if not isinstance(value, str) or _HEADER_CONTROL.search(value):
        raise TransportError("Некорректный почтовый адрес.")
    try:
        header = policy.default.header_factory("To", value)
        if (len(header.addresses) != 1 or any(group.display_name is not None for group in header.groups)
                or any(not isinstance(defect, errors.NonASCIILocalPartDefect) for defect in header.defects)):
            raise ValueError
        addr = header.addresses[0]
        domain = addr.domain.encode("idna").decode("ascii").lower()
        if (not addr.username or _HEADER_CONTROL.search(addr.username + addr.display_name)
                or len(domain) > 253 or not all(_DOMAIN_LABEL.fullmatch(s) for s in domain.split("."))):
            raise ValueError
        return Address(display_name=addr.display_name, username=addr.username, domain=domain)
    except (ValueError, IndexError, errors.MessageError):
        raise TransportError("Некорректный почтовый адрес.") from None


def _account(provider: str, address: str, password: str):
    if provider not in _HOSTS:
        raise TransportError("Почтовый сервис не поддерживается.")
    sender = parse_address(address)
    if not isinstance(password, str) or not password or _HEADER_CONTROL.search(password):
        raise AuthenticationError("Проверьте пароль приложения для почты.")
    if provider == "gmail" and re.fullmatch(r"[A-Za-z0-9]{4}(?: [A-Za-z0-9]{4}){3}", password):
        password = password.replace(" ", "")
    # For a personal @yandex.ru account Yandex documents the local login; hosted domains use the full address.
    login = sender.username if provider == "yandex" and sender.domain == "yandex.ru" else sender.addr_spec
    return sender, login, password


def _ok(response):
    status, data = response
    if status != "OK":
        raise TransportError("Не удалось прочитать данные из входящих писем.")
    return data


@contextmanager
def _inbox(provider: str, address: str, password: str):
    _, login, password = _account(provider, address, password)
    client = None
    try:
        client = imaplib.IMAP4_SSL(_HOSTS[provider][0], 993, ssl_context=ssl.create_default_context(), timeout=_TIMEOUT)
        client.debug = 0
        # imaplib quotes the password. Escape the username if it is not an IMAP atom.
        if re.search(r'[ (){%*"\\\]]', login):
            login = '"' + login.replace("\\", "\\\\").replace('"', '\\"') + '"'
        try:
            authenticated, _ = client.login(login, password)
            if authenticated != "OK":
                raise AuthenticationError("Проверьте пароль приложения и доступ по IMAP.")
        except imaplib.IMAP4.abort:
            raise
        except imaplib.IMAP4.error:
            raise AuthenticationError("Проверьте пароль приложения и доступ по IMAP.") from None
        selected = _ok(client.select("INBOX", readonly=True))
        if not selected or not isinstance(selected[0], bytes) or not selected[0].isdigit():
            raise TransportError("Почтовый сервер не подтвердил открытие входящих писем.")
        yield client, int(selected[0])
    except (imaplib.IMAP4.error, OSError, EOFError, UnicodeError):
        raise TransportError("Не удалось соединиться с сервером входящей почты.") from None
    finally:
        if client is not None:
            with suppress(imaplib.IMAP4.error, OSError, EOFError):
                client.logout()


def verify_account(provider: str, address: str, password: str) -> None:
    with _inbox(provider, address, password):
        pass


def _uid_set(client, selected_size: int) -> set[int]:
    # Very large SEARCH replies exceed imaplib's single-line limit. FETCH UID emits a short line per message.
    if selected_size > 60_000:
        rows = _ok(client.uid("FETCH", "1:*", "(UID)"))
        if not rows:
            raise TransportError("Почтовый сервер не вернул полный список входящих писем.")
        current = set()
        for row in rows:
            uid = _UID.search(row) if isinstance(row, bytes) else None
            if not uid or len(uid[1]) > 10 or not 0 < int(uid[1]) < 2**32:
                raise TransportError("Почтовый сервер не вернул полный список входящих писем.")
            current.add(int(uid[1]))
        return current
    rows = _ok(client.uid("SEARCH", None, "ALL"))
    if not rows or any(not isinstance(row, bytes) for row in rows):
        raise TransportError("Почтовый сервер не вернул список входящих писем.")
    items = b" ".join(rows).split()
    if any(len(s) > 10 or not s.isdigit() or not 0 < int(s) < 2**32 for s in items):
        raise TransportError("Почтовый сервер вернул некорректный список писем.")
    return {int(s) for s in items}


class _HTMLText(HTMLParser):
    hidden_tags = {"head", "script", "style", "template"}
    block_tags = {"p", "br", "div", "li", "tr", "td", "h1", "h2", "h3", "blockquote"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.hidden = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in self.hidden_tags:
            self.hidden += 1
        elif not self.hidden and tag in self.block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in self.hidden_tags:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in self.block_tags:
            self.parts.append("\n")

    def handle_data(self, text):
        if not self.hidden:
            self.parts.append(re.sub(r"\s+", " ", text))


def _body(message: EmailMessage) -> str:
    # Skip named inline attachments and encapsulated forwarded messages as well as ordinary attachments.
    pending = [message]
    html = ""
    while pending:
        part = pending.pop()
        if part.get_content_disposition() == "attachment" or part.get_filename():
            continue
        mime = part.get_content_type()
        if mime.startswith("multipart/"):
            children = list(part.iter_parts())
            if mime == "multipart/related" and children:
                start = part.get_param("start")
                children = [next((child for child in children if start and child.get("Content-ID") == start), children[0])]
            pending.extend(reversed(children))
            continue
        if mime not in {"text/plain", "text/html"} or (mime == "text/html" and html):
            continue
        raw = part.get_payload(decode=True) or b""
        charset = part.get_content_charset() or "utf-8"
        try:
            value = raw.decode(charset)
        except (LookupError, UnicodeError):
            value = raw.decode("utf-8", errors="replace")
        if mime == "text/plain" and value.strip():
            return value.strip()
        if mime == "text/html":
            parser = _HTMLText()
            # Unknown legacy markup should not block every later message in this mailbox.
            with suppress(AssertionError, ValueError):
                parser.feed(value)
                parser.close()
            html = "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip())
    return html


def _day(internal_date: str, message_date: str) -> str | None:
    for value in (internal_date, message_date):
        try:
            parsed = parsedate_to_datetime(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).date().isoformat()
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _document(raw: bytes, metadata: bytes, mailbox: str, validity: str, uid: int) -> dict:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    def head(name):
        return " ".join(str(message.get(name, "")).split())
    internal_date = _DATE.search(metadata)
    day = _day(internal_date[1].decode("ascii", "replace") if internal_date else "", head("Date"))
    key = hashlib.sha256(f"{mailbox}\0{validity}\0{uid}".encode()).hexdigest()
    return {"doc_id": "mail-" + key, "synthetic_user": mailbox, "doc_dt": day,
            "doc": f"От: {head('From')}\nТема: {head('Subject')}\nТекст письма:\n{_body(message)}"}


def _groups(client, wanted: list[int]) -> Iterator[list[int]]:
    # Fetch large attachments one message at a time. Only decoded body text is returned to storage.
    sequence = ",".join(map(str, wanted))
    sizes = {}
    for row in _ok(client.uid("FETCH", sequence, "(UID RFC822.SIZE)")):
        if isinstance(row, bytes) and (uid := _UID.search(row)) and (n := _SIZE.search(row)):
            sizes[int(uid[1])] = int(n[1])
    limit = 4 * 1024 * 1024
    group, size = [], 0
    for uid in wanted:
        length = sizes.get(uid, limit)
        if group and (size + length > limit or len(group) >= 20):
            yield group
            group, size = [], 0
        group.append(uid)
        size += length
    if group:
        yield group


def _literals(rows):
    # A server can return UID and INTERNALDATE either before or after the body literal.
    metadata, raw = b"", None
    for row in rows:
        if isinstance(row, tuple):
            if raw is not None:
                yield metadata, raw
            metadata, raw = row
        elif isinstance(row, bytes) and raw is not None:
            metadata += b" " + row
            if row.rstrip().endswith(b")"):
                yield metadata, raw
                metadata, raw = b"", None
    if raw is not None:
        yield metadata, raw


def fetch_inbox(provider: str, address: str, password: str, *, mailbox: str, known_validity: str | None,
                known_uids: set[int], batch_size: int = 100) -> SyncBatch:
    """Fetch one newest-first batch. Repeat with the committed UID set while remaining is positive."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Размер порции должен быть положительным целым числом.")
    with _inbox(provider, address, password) as (client, selected_size):
        _, epoch = client.response("UIDVALIDITY")
        if (not epoch or not isinstance(epoch[0], bytes) or not re.fullmatch(rb"[1-9]\d{0,9}", epoch[0])
                or int(epoch[0]) >= 2**32):
            raise TransportError("Почтовый сервер не подтвердил версию папки входящих.")
        validity = epoch[0].decode("ascii")
        current = _uid_set(client, selected_size)
        known = known_uids if validity == known_validity else set()
        wanted = heapq.nlargest(batch_size, current - known)
        messages = {}
        if wanted:
            for group in _groups(client, wanted):
                data = _ok(client.uid("FETCH", ",".join(map(str, group)), "(UID INTERNALDATE BODY.PEEK[])"))
                for metadata, raw in _literals(data):
                    uid = _UID.search(metadata)
                    if uid and int(uid[1]) in group:
                        full = re.search(rb"\bBODY\[\]\s+\{(\d+)\}", metadata, re.I)
                        if not full or not isinstance(raw, bytes) or int(full[1]) != len(raw):
                            continue
                        ident = int(uid[1])
                        messages[ident] = _document(raw, metadata, mailbox, validity, ident)
            if set(wanted) != messages.keys():
                # EXPUNGE may race with SEARCH. Surviving missing literals must be retried as a whole batch.
                current = _uid_set(client, selected_size)
                if (set(wanted) & current) - messages.keys():
                    raise TransportError("Не все письма удалось получить. Повторите синхронизацию.")
                messages = {uid: row for uid, row in messages.items() if uid in current}
        return SyncBatch(validity, current, messages, len(current - known - messages.keys()))


def _close_smtp(client):
    with suppress(OSError, smtplib.SMTPException):
        client.close()


@contextmanager
def _submission(host: str):
    """Yield a TLS-verified connection. Port selection ends before credentials or mail are sent."""
    context = ssl.create_default_context()
    with ExitStack() as connections:
        try:
            smtp = smtplib.SMTP_SSL(host, 465, context=context, timeout=_SMTP_PRIMARY_TIMEOUT)
            connections.callback(_close_smtp, smtp)
            smtp.set_debuglevel(0)
            smtp.ehlo_or_helo_if_needed()
        except ssl.SSLCertVerificationError:
            raise
        except (OSError, smtplib.SMTPException, EOFError):
            connections.close()
            try:
                smtp = smtplib.SMTP(host, 587, timeout=_SMTP_FALLBACK_TIMEOUT)
            except (OSError, smtplib.SMTPException, EOFError):
                raise TransportError("Не удалось подключиться к серверу отправки по портам 465 и 587.") from None
            connections.callback(_close_smtp, smtp)
            smtp.set_debuglevel(0)
            code, _ = smtp.ehlo()
            if code != 250 or not smtp.has_extn("starttls"):
                raise TransportError("Сервер отправки не предложил защищённое соединение STARTTLS.")
            code, _ = smtp.starttls(context=context)
            if code != 220:
                raise TransportError("Сервер отправки отказался установить TLS.")
            code, _ = smtp.ehlo()
            if code != 250:
                raise TransportError("Сервер не подтвердил почтовый протокол после установки TLS.")
        smtp.sock.settimeout(_TIMEOUT)
        yield smtp


def send_message(provider: str, address: str, password: str, *, to: list[str], subject: str,
                 body: str, message_id: str) -> None:
    sender, login, password = _account(provider, address, password)
    if not isinstance(to, list) or not to:
        raise TransportError("Укажите получателя письма.")
    recipients = [parse_address(value) for value in to]
    if not isinstance(subject, str) or _HEADER_CONTROL.search(subject) or not isinstance(body, str):
        raise TransportError("Проверьте тему и текст письма.")
    ident = message_id if isinstance(message_id, str) else ""
    if ident.startswith("<") and ident.endswith(">"):
        ident = ident[1:-1]
    if not _MSG_ID.fullmatch(ident):
        raise TransportError("Некорректный идентификатор исходящего письма.")
    envelope = list(dict.fromkeys(recipient.addr_spec for recipient in recipients))
    international = any(not spec.isascii() for spec in [sender.addr_spec, *envelope])
    try:
        msg = EmailMessage(policy=policy.SMTPUTF8 if international else policy.SMTP)
        msg["From"], msg["To"], msg["Subject"] = sender, recipients, subject
        msg["Date"], msg["Message-ID"] = format_datetime(datetime.now(timezone.utc)), f"<{ident}>"
        msg.set_content(body, cte="quoted-printable")
        # SMTP DATA transparency and exactly one end-of-data line, independent of the body supplied by the user.
        wire = re.sub(rb"(?m)^\.", b"..", msg.as_bytes())
    except (ValueError, errors.MessageError):
        raise TransportError("Не удалось подготовить заголовки или текст письма.") from None
    wire += (b"" if wire.endswith(b"\r\n") else b"\r\n") + b".\r\n"
    try:
        with _submission(_HOSTS[provider][1]) as smtp:
            smtp.login(login, password)
            if international and not smtp.has_extn("smtputf8"):
                raise TransportError("Сервер не поддерживает этот международный почтовый адрес.")
            code, _ = smtp.mail(sender.addr_spec, options=("SMTPUTF8",) if international else ())
            if code != 250:
                raise TransportError("Почтовый сервер отклонил адрес отправителя.")
            for recipient in envelope:
                code, _ = smtp.rcpt(recipient)
                if code not in (250, 251, 252):
                    raise TransportError("Сервер отклонил адрес получателя. Письмо не отправлено.")
            code, _ = smtp.docmd("DATA")
            if code != 354:
                raise TransportError("Почтовый сервер отказался принять письмо.")
            try:
                smtp.send(wire)
                code, _ = smtp.getreply()
            except (OSError, smtplib.SMTPException, EOFError):
                raise DeliveryUncertain("Сервер не подтвердил отправку. Проверьте отправленные письма перед повтором.") from None
            if 400 <= code < 600:
                raise TransportError("Почтовый сервер не принял отправленное письмо.")
            if not 200 <= code < 300:
                raise DeliveryUncertain("Получен неоднозначный ответ сервера. Проверьте отправленные письма.")
    except ssl.SSLCertVerificationError:
        raise TransportError("Не удалось проверить сертификат почтового сервера.") from None
    except smtplib.SMTPAuthenticationError:
        raise AuthenticationError("Проверьте пароль приложения и разрешение на отправку.") from None
    except (OSError, smtplib.SMTPException, EOFError, UnicodeError):
        raise TransportError("Не удалось соединиться с сервером исходящей почты.") from None
