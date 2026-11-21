from ipaddress import ip_address
from typing import Annotated, Literal
from urllib.parse import urlsplit
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from starlette.datastructures import Headers, MutableHeaders

from .local_mail import LocalMailError, SYNC_SECONDS
from .mail_transport import AuthenticationError, DeliveryUncertain, TransportError, parse_address


class LocalAccess:
    def __init__(self, app, state):
        self.app = app
        self.state = state

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if getattr(self.state, "local_mail", None) is None:
            if path.startswith("/api/local/"):
                return await JSONResponse({"detail": "Локальные аккаунты отключены."}, status_code=404)(scope, receive, send)
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        try:
            host = urlsplit("//" + headers.get("host", ""))
            peer = ip_address(scope["client"][0])
            local = peer.is_loopback and host.hostname in {"localhost", "127.0.0.1", "::1"}
        except (ValueError, TypeError, IndexError):
            local = False
        forwarded = any(h in headers for h in ("forwarded", "x-forwarded-for", "x-forwarded-host",
                                                 "cf-connecting-ip", "x-real-ip"))
        origin = headers.get("origin")
        foreign_origin = origin is not None and origin.lower() != f"{scope['scheme']}://{headers.get('host', '')}".lower()
        foreign_fetch = path.startswith("/api/") and headers.get("sec-fetch-site") == "cross-site"
        modify = path.startswith("/api/local/") and scope["method"] not in {"GET", "HEAD"}
        if not local or forwarded or foreign_origin or foreign_fetch or (modify and headers.get("x-mailsearch-local") != "1"):
            denied = JSONResponse({"detail": "Откройте почту напрямую на localhost."}, status_code=403)
            return await denied(scope, receive, send)

        async def private_send(message):
            if message["type"] == "http.response.start" and path.startswith("/api/"):
                MutableHeaders(scope=message)["Cache-Control"] = "no-store"
            await send(message)

        return await self.app(scope, receive, private_send)


def normalized_address(value: str):
    try:
        parsed = parse_address(value)
    except TransportError as exc:
        raise ValueError("Некорректный адрес") from exc
    if len(parsed.addr_spec) > 254 or "." not in parsed.domain:
        raise ValueError("Некорректный адрес")
    return parsed


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["gmail", "yandex"]
    address: str
    password: SecretStr = Field(min_length=1, max_length=1024)

    @field_validator("address")
    @classmethod
    def check_address(cls, value):
        # Both providers use case-insensitive account identifiers.
        return normalized_address(value).addr_spec.casefold()


class Letter(BaseModel):
    model_config = ConfigDict(extra="forbid")
    account_id: str = Field(min_length=1, max_length=128)
    to: list[Annotated[str, Field(min_length=1, max_length=254)]] = Field(min_length=1, max_length=100)
    subject: str = Field(max_length=998)
    body: str = Field(max_length=1_000_000)
    request_id: UUID

    @field_validator("to")
    @classmethod
    def check_recipients(cls, values):
        return list(dict.fromkeys(str(normalized_address(item)) for item in values))

    @field_validator("subject")
    @classmethod
    def check_subject(cls, value):
        if any(ord(c) < 32 or ord(c) == 127 for c in value):
            raise ValueError("Недопустимый перенос в теме")
        return value

    @model_validator(mode="after")
    def has_content(self):
        if not self.subject.strip() and not self.body.strip():
            raise ValueError("Укажите тему или текст письма")
        return self


router = APIRouter(prefix="/api/local")


def service(request):
    local = getattr(request.app.state, "local_mail", None)
    if local is None:
        raise HTTPException(404, "Локальные аккаунты отключены.")
    return local


def handle(operation, *args):
    try:
        return operation(*args)
    except LocalMailError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    except DeliveryUncertain as exc:
        raise HTTPException(409, str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(401, str(exc)) from exc
    except TransportError as exc:
        raise HTTPException(502, str(exc)) from exc


@router.get("/accounts")
def accounts(request: Request):
    return {"accounts": service(request).accounts(), "auto_sync_seconds": SYNC_SECONDS}


@router.post("/accounts")
def connect(request: Request, entry: Connection):
    local = service(request)
    return {"account": handle(local.connect, entry.provider, entry.address, entry.password.get_secret_value())}


@router.post("/accounts/{account_id}/sync")
def sync(request: Request, account_id: str):
    return {"account": handle(service(request).sync, account_id)}


@router.post("/send")
def send_mail(request: Request, letter: Letter):
    handle(service(request).send, letter.account_id, letter.to, letter.subject, letter.body, str(letter.request_id))
    return {"status": "sent"}
