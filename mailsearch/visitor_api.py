"""Access rules at the HTTP boundary and the read-only visitor account endpoints."""

from contextlib import nullcontext
from ipaddress import ip_address
from typing import Literal
from urllib.parse import urlsplit

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from starlette.datastructures import Headers, MutableHeaders

from .mail_transport import TransportError, parse_address
from .private_cache import private_request
from .visitor_mail import RELOGIN, SYNC_SECONDS, VisitorError

_FORWARDED = {"forwarded", "x-forwarded-for", "x-forwarded-host", "x-forwarded-proto", "cf-connecting-ip", "x-real-ip"}


def frontend_origin(value):
    if not value.strip():
        return ""
    try:
        parsed = urlsplit(value.strip())
        host, port = parsed.hostname, parsed.port
        if (not host or "*" in value or any(ord(c) < 32 for c in value) or parsed.username is not None
                or parsed.password is not None or parsed.path not in {"", "/"} or parsed.query or parsed.fragment
                or not (parsed.scheme == "https" or parsed.scheme == "http" and host in {"localhost", "127.0.0.1", "::1"})):
            raise ValueError
    except ValueError:
        raise ValueError("MAILSEARCH_FRONTEND_ORIGIN должен содержать точный HTTPS-адрес сайта.") from None
    host = "[" + host + "]" if ":" in host else host
    default_port = 443 if parsed.scheme == "https" else 80
    return f"{parsed.scheme}://{host}" + (f":{port}" if port is not None and port != default_port else "")


def _loopback(scope, headers):
    try:
        peer = ip_address(scope["client"][0])
        host = urlsplit("//" + headers.get("host", "")).hostname
    except (ValueError, TypeError, IndexError, KeyError):
        return False
    return peer.is_loopback and host in {"localhost", "127.0.0.1", "::1"} and not any(h in headers for h in _FORWARDED)


class VisitorAccess:
    def __init__(self, app, state, origin):
        self.app = app
        self.state = state
        self.origin = origin

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        if scope["type"] != "http" or not path.startswith("/api/"):
            return await self.app(scope, receive, send)
        headers = Headers(scope=scope)
        origin = headers.get("origin")
        same_origin = origin == f"{scope['scheme']}://{headers.get('host', '')}"
        direct = _loopback(scope, headers)
        public = getattr(self.state, "public_profile", False)
        secure = direct or (public and scope["scheme"] == "https")
        registry = getattr(self.state, "visitors", None) if secure else None
        state = scope.setdefault("state", {})
        state.update(visitors=registry, visitor=None)

        async def private_send(message):
            if message["type"] == "http.response.start":
                out = MutableHeaders(scope=message)
                out["Cache-Control"] = "private, no-store"
                out.add_vary_header("Authorization")
                out.add_vary_header("Origin")
            await send(message)

        async def deny(status, detail):
            return await JSONResponse({"detail": detail}, status_code=status)(scope, receive, private_send)

        visitor_path = path.startswith("/api/visitor/")
        create = path == "/api/visitor/session"
        if visitor_path and registry is None:
            return await deny(404, "Подключение личной почты недоступно.")
        if visitor_path:
            allowed_origin = same_origin or bool(public and self.origin and origin == self.origin)
            if (origin is not None and not allowed_origin) or (not origin and headers.get("sec-fetch-site") == "cross-site"):
                return await deny(403, "Откройте подключение с сайта почты.")
            if scope["method"] == "POST" and (headers.get("x-mailsearch-visitor") != "1" or
                    headers.get("content-type", "").partition(";")[0].strip().lower() != "application/json"):
                return await deny(403, "Для изменения подключения требуется подтверждённый запрос приложения.")
        if "authorization" in headers:
            auth = headers.getlist("authorization")
            kind, _, token = auth[0].partition(" ")
            if len(auth) != 1 or kind.casefold() != "bearer" or not registry:
                return await deny(401, RELOGIN)
            try:
                state["visitor"] = registry.resolve(token)
            except VisitorError as exc:
                return await deny(exc.status, str(exc))
        elif visitor_path and not create:
            return await deny(401, RELOGIN)

        downstream_receive = receive
        if visitor_path and scope["method"] == "POST":
            # Connecting requires just three small fields. Bound chunked requests as well as Content-Length.
            data = bytearray()
            while True:
                part = await receive()
                if part["type"] == "http.disconnect":
                    return
                chunk = part.get("body", b"")
                if len(data) + len(chunk) > 16_384:
                    return await deny(413, "Слишком большой запрос подключения.")
                data.extend(chunk)
                if not part.get("more_body"):
                    break
            body = bytes(data)

            async def downstream_receive():
                nonlocal body
                if body is None:
                    return await receive()
                result, body = body, None
                return {"type": "http.request", "body": result, "more_body": False}

        # A logout or expiry while search/generation is running must also suppress its pending output.
        visitor = state["visitor"]
        discarded = False

        async def owned_send(message):
            nonlocal discarded
            if message["type"] == "http.response.start" and not registry.active(visitor):
                discarded = True
                await deny(401, RELOGIN)
            if not discarded:
                await private_send(message)

        protect_response = visitor is not None and path != "/api/visitor/logout"
        with private_request() if visitor else nullcontext():
            return await self.app(scope, downstream_receive, owned_send if protect_response else private_send)


class Empty(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Connection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: Literal["yandex"]
    address: str = Field(min_length=3, max_length=254)
    password: SecretStr = Field(min_length=1, max_length=1024)

    @field_validator("address")
    @classmethod
    def validate_address(cls, value):
        try:
            parsed = parse_address(value)
        except TransportError:
            raise ValueError("Некорректный адрес") from None
        if parsed.display_name or "." not in parsed.domain or len(parsed.addr_spec) > 254:
            raise ValueError("Некорректный адрес")
        return parsed.addr_spec.casefold()


router = APIRouter(prefix="/api/visitor")


@router.post("/session")
def create_session(request: Request, entry: Empty):
    peer = request.client.host if request.client else "local"
    return request.state.visitors.create(peer)


@router.get("/accounts")
def list_accounts(request: Request):
    return {"accounts": request.state.visitors.accounts(request.state.visitor), "auto_sync_seconds": SYNC_SECONDS}


@router.post("/accounts")
def connect(request: Request, entry: Connection):
    result = request.state.visitors.connect(request.state.visitor, entry.provider, entry.address,
                                            entry.password.get_secret_value())
    return {"account": result}


@router.post("/accounts/{account_id}/sync")
def sync(request: Request, account_id: str, entry: Empty):
    return {"account": request.state.visitors.sync(request.state.visitor, account_id)}


@router.post("/logout")
def logout(request: Request, entry: Empty):
    request.state.visitors.logout(request.state.visitor)
    return {"ok": True}
