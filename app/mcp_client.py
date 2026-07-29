import time

import config
import errors
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent


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


async def call_tool(integration, tool, args) -> str:
    try:
        async with build_client(integration) as client:
            result = await client.call_tool(tool, arguments=args)
        text = "\n".join(block.text for block in result.content if isinstance(block, TextContent))
        return text[: integration.max_result_chars]
    except Exception as e:
        return f"{errors.ERROR_PREFIX}tool '{tool}' failed: {type(e).__name__}"
