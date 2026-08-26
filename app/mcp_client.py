import time
from dataclasses import dataclass

import anyio
import config
import errors
import httpx
import logging_setup
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.exceptions import ToolError
from mcp.types import TextContent

log = logging_setup.get_logger(__name__)


@dataclass
class CallOutcome:
    text: str
    error_kind: str | None = None


# fastmcp hides the real cause under RuntimeError inside a TaskGroup group
def classify(exc: BaseException, depth: int = 0) -> str:
    if isinstance(exc, ToolError):
        return "tool"
    if isinstance(exc, BaseExceptionGroup):
        kinds = [classify(inner, depth + 1) for inner in exc.exceptions]
        return next((kind for kind in kinds if kind != "unknown"), "unknown")
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is not None:
        if status in (401, 403):
            return "auth"
        return "server" if status >= 500 else "client"
    if isinstance(exc, httpx.TimeoutException | TimeoutError):
        return "timeout"
    if isinstance(exc, httpx.ConnectError | OSError):
        return "connect"
    # the stream dies in the writer task, so the real status lands in a traceback we never see
    if isinstance(exc, anyio.BrokenResourceError | anyio.ClosedResourceError):
        return "connect"
    cause = exc.__cause__ or exc.__context__
    if cause is not None and depth < 5:
        return classify(cause, depth + 1)
    if depth == 0:
        log.warning("mcp.unclassified_error", type=type(exc).__name__, error=repr(exc)[:200])
    return "unknown"


def auth_headers(integration) -> dict:
    auth = integration.auth or {}
    if auth.get("type") == "bearer":
        token = config.settings.mcp_integrations.secret(auth["token_env"])
        return {"Authorization": f"Bearer {token}"} if token else {}
    if auth.get("type") == "header":
        value = config.settings.mcp_integrations.secret(auth["value_env"])
        return {auth["header"]: value} if value else {}
    return {}


def build_client(integration) -> Client:
    transport = StreamableHttpTransport(integration.url, headers=auth_headers(integration))
    return Client(transport, timeout=integration.timeout_s)


async def ping(integration) -> float:
    start = time.perf_counter()
    async with build_client(integration) as client:
        await client.ping()
    return round(time.perf_counter() - start, 3)


async def list_tools(integration) -> list:
    async with build_client(integration) as client:
        return await client.list_tools()


async def call_tool(integration, tool, args) -> CallOutcome:
    try:
        async with build_client(integration) as client:
            result = await client.call_tool(tool, arguments=args)
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
        return CallOutcome(text=text[: integration.max_result_chars])
    except Exception as e:
        kind = classify(e)
        log.warning(
            "mcp.call_failed",
            integration=integration.name,
            tool=tool,
            kind=kind,
            error=type(e).__name__,
        )
        return CallOutcome(
            text=f"{errors.ERROR_PREFIX}tool '{tool}' failed ({kind}): {type(e).__name__}",
            error_kind=kind,
        )
