"""
MCP Client — Manages a persistent connection to the StockResearch MCP server.

The main app spawns the MCP server as a subprocess (stdio transport) and
communicates with it over stdin/stdout using the MCP protocol. This gives
full tool-use semantics: list tools, call tools by name, read resources.

Usage in the app:
    from src.mcp_server.client import mcp_client

    # On app startup
    await mcp_client.start()

    # Call any tool
    result = await mcp_client.call_tool("fetch_stock_news", symbol="RELIANCE", limit=5)

    # On app shutdown
    await mcp_client.stop()
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────
# Determine the Python executable & server module path
# ──────────────────────────────────────────────────────────────

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PYTHON = sys.executable  # Same interpreter that's running the app


class MCPClient:
    """
    Persistent MCP client that manages a subprocess StockResearch server.

    Lifecycle:
        start()  → spawns the MCP server, initialises the session
        call_tool(name, **kwargs)  → calls a tool and returns parsed JSON
        list_tools()  → lists all available tools
        read_resource(uri)  → reads an MCP resource
        stop()  → gracefully shuts down the server subprocess
    """

    def __init__(self) -> None:
        self._session: Optional[ClientSession] = None
        self._exit_stack: Optional[AsyncExitStack] = None
        self._started = False
        self._tools_cache: list[dict[str, Any]] = []

    @property
    def is_connected(self) -> bool:
        return self._started and self._session is not None

    # ──────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Spawn the MCP server subprocess and initialise the session."""
        if self._started:
            logger.info("mcp_client_already_started")
            return

        try:
            server_params = StdioServerParameters(
                command=_PYTHON,
                args=["-m", "src.mcp_server"],
                cwd=_BASE_DIR,
                env={**os.environ},
            )

            self._exit_stack = AsyncExitStack()
            stdio_transport = await self._exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            read_stream, write_stream = stdio_transport

            self._session = await self._exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialise the MCP session (capability handshake)
            await self._session.initialize()

            self._started = True
            logger.info("mcp_client_started", server="StockResearch")

            # Cache the tool list
            await self._refresh_tools()

        except Exception as e:
            logger.error("mcp_client_start_failed", error=str(e))
            await self.stop()
            raise

    async def stop(self) -> None:
        """Shut down the MCP server subprocess."""
        if self._exit_stack:
            try:
                await self._exit_stack.aclose()
            except Exception as e:
                logger.warning("mcp_client_stop_error", error=str(e))
        self._session = None
        self._exit_stack = None
        self._started = False
        self._tools_cache = []
        logger.info("mcp_client_stopped")

    # ──────────────────────────────────────────────────────────
    # Tool Discovery
    # ──────────────────────────────────────────────────────────

    async def _refresh_tools(self) -> None:
        """Fetch and cache the list of available tools from the server."""
        if not self._session:
            return
        try:
            result = await self._session.list_tools()
            self._tools_cache = [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema if hasattr(t, "inputSchema") else {},
                }
                for t in result.tools
            ]
            logger.info("mcp_tools_refreshed", count=len(self._tools_cache))
        except Exception as e:
            logger.warning("mcp_tools_refresh_failed", error=str(e))

    async def list_tools(self) -> list[dict[str, Any]]:
        """Return the list of available MCP tools."""
        if not self._tools_cache:
            await self._refresh_tools()
        return self._tools_cache

    # ──────────────────────────────────────────────────────────
    # Tool Invocation
    # ──────────────────────────────────────────────────────────

    async def call_tool(self, name: str, **kwargs: Any) -> dict[str, Any]:
        """
        Call an MCP tool by name and return parsed JSON result.

        Args:
            name: Tool name (e.g. "fetch_stock_news", "get_best_stocks")
            **kwargs: Tool arguments (e.g. symbol="RELIANCE", limit=5)

        Returns:
            Parsed dict from the tool's JSON response.

        Raises:
            RuntimeError: If the client is not started.
            Exception: If the tool call fails.
        """
        if not self._session:
            raise RuntimeError("MCP client not started. Call await mcp_client.start() first.")

        try:
            logger.info("mcp_tool_call", tool=name, args=kwargs)
            result = await self._session.call_tool(name, arguments=kwargs)

            # Extract text content from the result
            if result.content:
                for block in result.content:
                    if hasattr(block, "text"):
                        try:
                            return json.loads(block.text)
                        except json.JSONDecodeError:
                            return {"raw_text": block.text}

            # Check for errors
            if result.isError:
                return {"error": "Tool returned an error", "content": str(result.content)}

            return {"message": "No content returned"}

        except Exception as e:
            logger.error("mcp_tool_call_failed", tool=name, error=str(e))
            return {"error": f"MCP tool call failed: {e}"}

    async def call_tool_raw(self, name: str, **kwargs: Any) -> str:
        """Call an MCP tool and return the raw string response."""
        if not self._session:
            raise RuntimeError("MCP client not started.")

        result = await self._session.call_tool(name, arguments=kwargs)
        if result.content:
            for block in result.content:
                if hasattr(block, "text"):
                    return block.text
        return ""

    # ──────────────────────────────────────────────────────────
    # Resource Access
    # ──────────────────────────────────────────────────────────

    async def read_resource(self, uri: str) -> dict[str, Any]:
        """
        Read an MCP resource by URI.

        Args:
            uri: Resource URI (e.g. "stock://scanner/top-picks")

        Returns:
            Parsed JSON content of the resource.
        """
        if not self._session:
            raise RuntimeError("MCP client not started.")

        try:
            result = await self._session.read_resource(uri)
            if result.contents:
                for block in result.contents:
                    if hasattr(block, "text"):
                        try:
                            return json.loads(block.text)
                        except json.JSONDecodeError:
                            return {"raw_text": block.text}
            return {"message": "No content"}
        except Exception as e:
            logger.error("mcp_resource_read_failed", uri=uri, error=str(e))
            return {"error": f"Resource read failed: {e}"}

    # ──────────────────────────────────────────────────────────
    # Prompt Templates
    # ──────────────────────────────────────────────────────────

    async def get_prompt(self, name: str, **kwargs: Any) -> str:
        """
        Get a rendered prompt template from the MCP server.

        Args:
            name: Prompt name (e.g. "stock_analysis_prompt")
            **kwargs: Prompt arguments (e.g. symbol="RELIANCE")

        Returns:
            Rendered prompt string.
        """
        if not self._session:
            raise RuntimeError("MCP client not started.")

        try:
            result = await self._session.get_prompt(name, arguments=kwargs)
            if result.messages:
                texts = []
                for msg in result.messages:
                    if hasattr(msg.content, "text"):
                        texts.append(msg.content.text)
                return "\n".join(texts)
            return ""
        except Exception as e:
            logger.error("mcp_prompt_failed", name=name, error=str(e))
            return f"Error getting prompt: {e}"


# ══════════════════════════════════════════════════════════════
# Singleton instance — import and use this throughout the app
# ══════════════════════════════════════════════════════════════

mcp_client = MCPClient()
