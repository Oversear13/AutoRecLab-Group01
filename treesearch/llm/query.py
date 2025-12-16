import sys
from dataclasses import dataclass
from typing import Optional, Self, TypeAlias, TypeVar, overload

from langchain.agents import create_agent
from langchain.agents.middleware.types import ResponseT
from langchain.agents.structured_output import ResponseFormat
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection

from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("llm")

ResponseFormatType: TypeAlias = ResponseFormat[ResponseT] | type[ResponseT]
RT = TypeVar("RT", bound=ResponseFormatType)


@dataclass
class MCPConnection:
    name: str
    connection: Connection


""" # TODO:
- [ ] limit number of tool calls somehow?
- [ ] the error handling in lines 75 - 80 is prob not ideal 
"""


class Query:
    def __init__(self) -> None:
        self._mcp_connections: list[MCPConnection] = []
        self._tools: list[BaseTool] = []
        self._system_prompt: Optional[str] = None

    def with_tool(self, *tool: BaseTool) -> Self:
        self._tools.extend(tool)
        return self

    def with_mcp(self, *mcp_connection: MCPConnection) -> Self:
        self._mcp_connections.extend(mcp_connection)
        return self

    def with_system(self, system_prompt: str) -> Self:
        self._system_prompt = system_prompt
        return self

    @overload
    async def run(self, input: str) -> str: ...

    @overload
    async def run(self, input: str, response_format: RT) -> RT: ...

    async def run(self, input: str, response_format: Optional[RT] = None) -> RT | str:
        tools = await self._get_all_tools()

        agent = create_agent(
            model="gpt-5-mini",
            tools=tools,
            response_format=response_format,
            system_prompt=self._system_prompt,
        )

        resp = await agent.ainvoke({"messages": [HumanMessage(input)]})

        if response_format:
            structured_resp: RT = resp["structured_response"]
            return structured_resp

        messages = resp.get("messages")
        if messages is None or len(messages) == 0:
            logger.critical("LLM did not return any message!")
            sys.exit(1)

        last_msg = messages[-1]
        if not isinstance(last_msg, AIMessage):
            logger.critical("Last message was not an AIMessage!")
            sys.exit(1)

        return str(last_msg.content)

    async def _get_all_tools(self) -> list[BaseTool]:
        tools = self._tools

        connection_dict: dict[str, Connection] = {
            mcp.name: mcp.connection for mcp in self._mcp_connections
        }
        client = MultiServerMCPClient(connection_dict)
        tools.extend(await client.get_tools())

        return tools
