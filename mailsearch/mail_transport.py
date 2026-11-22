"""Read-only TLS access to a provider-selected IMAP inbox."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from email import errors, policy
from email.headerregistry import Address
from email.message import EmailMessage
from email.parser import BytesParser
from email.utils import parsedate_to_datetime
import hashlib
import heapq
from html.parser import HTMLParser
import imaplib
import logging
import re
import ssl

import certifi


_HOSTS = {"gmail": "imap.gmail.com", "yandex": "imap.yandex.ru"}
_LOGGER = logging.getLogger(__name__)
_TIMEOUT = 30
MAX_MIME_BYTES = 256 * 1024
MAX_BODY_CHARS = 16_384
_UID = re.compile(rb"\bUID (\d+)\b", re.I)
_SIZE = re.compile(rb"\bRFC822\.SIZE (\d+)\b", re.I)
_DATE = re.compile(rb'\bINTERNALDATE "([^"]+)"', re.I)
_HEADER_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f\u2028\u2029]")
_DOMAIN_LABEL = re.compile(r"[a-z\d](?:[a-z\d-]{0,61}[a-z\d])?", re.I | re.ASCII)
_GMAIL_APP_PASSWORD = re.compile(r"[A-Za-z0-9]{16}")


class TransportError(Exception):
    """The message is safe to display and contains no credentials or provider reply."""


class AuthenticationError(TransportError):
    """A safe message and a fixed diagnostic reason for a failed login."""

    def __init__(self, detail: str, *, reason: str):
        super().__init__(detail)
        self.reason = reason


@dataclass
class SyncBatch:
    validity: str
    current_uids: set[int]
    messages: dict[int, dict]
    remaining: int
    total_available: int | None = None


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
    if provider == "gmail" and isinstance(password, str):
        compact = "".join(password.split())
        if _GMAIL_APP_PASSWORD.fullmatch(compact):
            password = compact
    if not isinstance(password, str) or not password or _HEADER_CONTROL.search(password):
        raise AuthenticationError("Проверьте формат пароля приложения.", reason="invalid_format")
    # For a personal @yandex.ru account Yandex documents the local login; hosted domains use the full address.
    login = sender.username if provider == "yandex" and sender.domain == "yandex.ru" else sender.addr_spec
    return sender, login, password


def _login_error(provider: str, reply: str) -> AuthenticationError:
    if provider == "gmail":
        reply = reply.casefold()
        if "application-specific password required" in reply:
            return AuthenticationError(
                "Gmail требует пароль приложения. Создайте его в настройках того же аккаунта Google.",
                reason="app_password_required",
            )
        if "please log in via your web browser" in reply:
            return AuthenticationError(
                "Gmail просит подтвердить вход. Откройте Gmail в браузере и проверьте оповещения о безопасности.",
                reason="browser_confirmation_required",
            )
        return AuthenticationError(
            "Gmail отклонил вход. Проверьте полный адрес и пароль приложения от этого аккаунта Google.",
            reason="login_rejected",
        )
    return AuthenticationError(
        "Проверьте пароль приложения и разрешение на чтение почты по IMAP.", reason="login_rejected",
    )


def _ok(response):
    status, data = response
    if status != "OK":
        raise TransportError("Не удалось прочитать данные из входящих писем.")
    return data


@contextmanager
def _inbox(provider: str, address: str, password: str):
    _, login, password = _account(provider, address, password)
    client = None
    stage = "connect"
    try:
        # The public IMAP providers use public roots, even if the Python install has no CA store.
        context = ssl.create_default_context(cafile=certifi.where())
        client = imaplib.IMAP4_SSL(_HOSTS[provider], 993, ssl_context=context, timeout=_TIMEOUT)
        client.debug = 0
        # imaplib quotes the password. Escape the username if it is not an IMAP atom.
        if re.search(r'[ (){%*"\\\]]', login):
            login = '"' + login.replace("\\", "\\\\").replace('"', '\\"') + '"'
        stage = "login"
        try:
            authenticated, reply = client.login(login, password)
        except imaplib.IMAP4.abort:
            raise
        except imaplib.IMAP4.error as error:
            raise _login_error(provider, str(error)) from None
        if authenticated != "OK":
            raise _login_error(provider, str(reply))
        stage = "select"
        selected = _ok(client.select("INBOX", readonly=True))
        if not selected or not isinstance(selected[0], bytes) or not selected[0].isdigit():
            raise TransportError("Почтовый сервер не подтвердил открытие входящих писем.")
        stage = "read"
        yield client, int(selected[0])
    except (imaplib.IMAP4.error, OSError, EOFError, UnicodeError) as error:
        verify_code = getattr(error, "verify_code", None)
        diagnostics = f" verify_code={verify_code}" if verify_code is not None else ""
        _LOGGER.warning("IMAP provider=%s stage=%s error=%s%s", provider, stage, type(error).__name__, diagnostics)
        raise TransportError("Не удалось соединиться с сервером входящей почты.") from None
    finally:
        if client is not None:
            with suppress(imaplib.IMAP4.error, OSError, EOFError):
                client.logout()


def verify_account(provider: str, address: str, password: str) -> None:
    with _inbox(provider, address, password):
        pass


def _uid_set(client, selected_size: int, max_messages: int | None = None) -> set[int]:
    # Very large SEARCH replies exceed imaplib's single-line limit. FETCH UID emits a short line per message.
    window = max_messages is not None and selected_size > max_messages
    if window or selected_size > 60_000:
        if window:
            # IMAP message sequence numbers ascend with UIDs. Request only metadata for the latest window.
            rows = _ok(client.fetch(f"{selected_size - max_messages + 1}:{selected_size}", "(UID)"))
        else:
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


def _document(raw: bytes, metadata: bytes, mailbox: str, validity: str, uid: int, clipped: bool = False) -> dict:
    message = BytesParser(policy=policy.default).parsebytes(raw)
    def head(name):
        return " ".join(str(message.get(name, "")).split())[:512]
    internal_date = _DATE.search(metadata)
    day = _day(internal_date[1].decode("ascii", "replace") if internal_date else "", head("Date"))
    key = hashlib.sha256(f"{mailbox}\0{validity}\0{uid}".encode()).hexdigest()
    body = _body(message)
    result = {"doc_id": "mail-" + key, "synthetic_user": mailbox, "doc_dt": day,
              "doc": f"От: {head('From')}\nТема: {head('Subject')}\nТекст письма:\n{body[:MAX_BODY_CHARS]}"}
    if clipped or len(body) > MAX_BODY_CHARS:
        result["truncated"] = True
    return result


def _groups(client, wanted: list[int]) -> Iterator[tuple[list[int], bool]]:
    sequence = ",".join(map(str, wanted))
    sizes = {}
    for row in _ok(client.uid("FETCH", sequence, "(UID RFC822.SIZE)")):
        if isinstance(row, bytes) and (uid := _UID.search(row)) and (n := _SIZE.search(row)):
            sizes[int(uid[1])] = int(n[1])
    limit = 4 * 1024 * 1024
    group, size = [], 0
    for uid in wanted:
        length = sizes.get(uid)
        if length is None or length > MAX_MIME_BYTES:
            yield [uid], True
            continue
        if group and (size + length > limit or len(group) >= 20):
            yield group, False
            group, size = [], 0
        group.append(uid)
        size += length
    if group:
        yield group, False


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
                known_uids: set[int], batch_size: int = 100, max_messages: int | None = None) -> SyncBatch:
    """Fetch one newest-first batch. Repeat with the committed UID set while remaining is positive."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Размер порции должен быть положительным целым числом.")
    if max_messages is not None and (type(max_messages) is not int or max_messages < 1):
        raise ValueError("Размер окна должен быть положительным целым числом.")
    with _inbox(provider, address, password) as (client, selected_size):
        _, epoch = client.response("UIDVALIDITY")
        if (not epoch or not isinstance(epoch[0], bytes) or not re.fullmatch(rb"[1-9]\d{0,9}", epoch[0])
                or int(epoch[0]) >= 2**32):
            raise TransportError("Почтовый сервер не подтвердил версию папки входящих.")
        validity = epoch[0].decode("ascii")
        current = _uid_set(client, selected_size, max_messages)
        known = known_uids if validity == known_validity else set()
        wanted = heapq.nlargest(batch_size, current - known)
        messages = {}
        if wanted:
            for group, clipped in _groups(client, wanted):
                span = f"<0.{MAX_MIME_BYTES}>" if clipped else ""
                data = _ok(client.uid("FETCH", ",".join(map(str, group)), f"(UID INTERNALDATE BODY.PEEK[]{span})"))
                for metadata, raw in _literals(data):
                    uid = _UID.search(metadata)
                    if uid and int(uid[1]) in group:
                        full = re.search(rb"\bBODY\[\](?:<0>)?\s+\{(\d+)\}", metadata, re.I)
                        if not full or not isinstance(raw, bytes) or int(full[1]) != len(raw):
                            continue
                        ident = int(uid[1])
                        messages[ident] = _document(raw, metadata, mailbox, validity, ident, clipped)
            if set(wanted) != messages.keys():
                # EXPUNGE may race with SEARCH. Surviving missing literals must be retried as a whole batch.
                current = _uid_set(client, selected_size, max_messages)
                if (set(wanted) & current) - messages.keys():
                    raise TransportError("Не все письма удалось получить. Повторите синхронизацию.")
                messages = {uid: row for uid, row in messages.items() if uid in current}
        return SyncBatch(validity, current, messages, len(current - known - messages.keys()), selected_size)
