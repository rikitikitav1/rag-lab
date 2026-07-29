import time

import config
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


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
