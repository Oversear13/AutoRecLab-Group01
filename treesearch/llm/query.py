import sys
from dataclasses import dataclass
from typing import Optional, Self, TypeAlias, TypeVar, overload, Type, Any

from langchain.agents import create_agent
from langchain.agents.structured_output import ProviderStrategy, ResponseFormat, SchemaT, StructuredOutputValidationError
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import Connection
from langchain_openai import ChatOpenAI

import json
import re
from dataclasses import is_dataclass, fields as dc_fields
from typing import Any, Type

from config import get_config
from utils.log import _ROOT_LOGGER

logger = _ROOT_LOGGER.getChild("llm")

ResponseFormatType: TypeAlias = type[SchemaT]
RT = TypeVar("RT", bound=ResponseFormatType)

Prompt: TypeAlias = str | list["Prompt"] | dict[str, "Prompt"]


@dataclass
class MCPConnection:
    name: str
    connection: Connection


""" # TODO:
- [x] limit number of tool calls (solved: implemented via max_iterations parameter)
- [ ] the error handling in lines 75 - 80 is prob not ideal 
"""


class Query:
    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        max_iterations: int = 25,
    ) -> None:
        self._mcp_connections: list[MCPConnection] = []
        self._tools: list[BaseTool] = []
        self._system_prompt: Optional[str] = None
        self._strict = True

        config = get_config()
        if model is None:
            self._model = config.agent.code.model
        else:
            self._model = model

        if temperature is None:
            self._temperature = config.agent.code.model_temp
        else:
            self._temperature = temperature

        self._max_iterations = max_iterations
        
        self._mode = config.local_llm.llm_mode
        self._local_model=config.local_llm.local_model
        self._local_base_url=config.local_llm.base_url
    def with_tool(self, *tool: BaseTool) -> Self:
        self._tools.extend(tool)
        return self

    def with_mcp(self, *mcp_connection: MCPConnection) -> Self:
        self._mcp_connections.extend(mcp_connection)
        return self

    def with_system(self, system_prompt: str) -> Self:
        self._system_prompt = system_prompt
        return self

    def non_strict(self) -> Self:
        self._strict = False
        return self

    @overload
    async def run(self, input: Prompt) -> str: ...

    @overload
    async def run(self, input: Prompt, response_schema: RT) -> RT: ...

    async def run(
        self, input: Prompt, response_schema: Optional[RT] = None
    ) -> RT | str:
        input = prompt_to_md(input)
        tools = await self._get_all_tools()

        if response_schema is None:
            response_format = None
        else:
            response_format = ProviderStrategy(response_schema, strict=self._strict)

        # Set up the model based on mode
        if self._mode != "local":
            model = ChatOpenAI(model=self._model, temperature=self._temperature)
        else:
            model = ChatOpenAI(
                model=self._local_model,
                temperature=self._temperature,
                base_url=self._local_base_url,
                api_key="not needed"
            )
        
        agent = create_agent(
            model=model,
            tools=tools,
            response_format=response_format,
            system_prompt=self._system_prompt,
        )
        # TODO make this more elegant instead of using an large if / else

        # Different execution path for local vs non-local LLMs
        if self._mode == "local":
            # Enhanced error handling for local LLMs
            try:
                resp = await agent.ainvoke(
                    {"messages": [HumanMessage(input)]},
                    config={"recursion_limit": self._max_iterations},
                )

                if response_schema:
                    # Try to get structured response
                    try:
                        structured_resp: RT = resp["structured_response"]
                        return structured_resp
                    except (KeyError, StructuredOutputValidationError) as e:
                        # If structured response fails, try to repair
                        logger.warning(f"Structured response failed: {e}. Attempting repair...")
                        
                        # Use structured_output to repair
                        repaired = ensure_structured_agent_response(
                            agent_response=resp, 
                            schema=response_schema, 
                            llm=model, 
                            original_prompt=input, 
                            max_attempts=3
                        )
                        return repaired

                messages = resp.get("messages")
                if messages is None or len(messages) == 0:
                    raise RuntimeError("LLM did not return any message!")

                # Find the last AIMessage in the conversation
                ai_messages = [msg for msg in reversed(messages) if isinstance(msg, AIMessage)]
                if not ai_messages:
                    raise RuntimeError("No AIMessage found in response!")
                    
                return str(ai_messages[0].content)
                
            except Exception as e:
                # If agent fails completely, try direct structured output as fallback
                if response_schema:
                    logger.warning(f"Agent failed: {e}. Trying direct structured output...")
                    try:
                        return structured_output(
                            llm=model, 
                            schema=response_schema, 
                            prompt=input, 
                            max_retries=3
                        )
                    except Exception as repair_error:
                        logger.error(f"Repair also failed: {repair_error}")
                        raise
                else:
                    raise
        else:
            # Original, simpler logic for non-local (OpenAI) LLMs
            resp = await agent.ainvoke(
                {"messages": [HumanMessage(input)]},
                config={"recursion_limit": self._max_iterations},
            )

            if response_schema:
                structured_resp: RT = resp["structured_response"]
                return structured_resp

            messages = resp.get("messages")
            if messages is None or len(messages) == 0:
                raise RuntimeError("LLM did not return any message!")

            # Find the last AIMessage in the conversation
            ai_messages = [msg for msg in reversed(messages) if isinstance(msg, AIMessage)]
            if not ai_messages:
                raise RuntimeError("No AIMessage found in response!")

            return str(ai_messages[0].content)

    async def _get_all_tools(self) -> list[BaseTool]:
        tools = self._tools

        connection_dict: dict[str, Connection] = {
            mcp.name: mcp.connection for mcp in self._mcp_connections
        }
        client = MultiServerMCPClient(connection_dict)
        tools.extend(await client.get_tools())

        return tools


def prompt_to_md(prompt: Prompt) -> str:
    return _prompt_to_md(prompt)[0]


def _prompt_to_md(prompt: Prompt | None, level=1) -> tuple[str, bool]:
    if prompt is None:
        return "None", True
    elif isinstance(prompt, dict):
        parts = []
        any_text = False

        for k, v in prompt.items():
            body, has_text = _prompt_to_md(v, level + 1)
            parts.append(f"{'#' * level} {k}")
            if body:
                parts.append(body)
            if has_text:
                parts.append("")
                any_text = True

        return "\n".join(parts).rstrip(), any_text

    elif isinstance(prompt, list):
        parts = []
        prev_was_text = False
        any_text = False

        for v in prompt:
            body, has_text = _prompt_to_md(v, level)
            if not body:
                continue

            if prev_was_text and body.lstrip().startswith("#"):
                parts.append("")

            parts.append(body)
            prev_was_text = has_text
            any_text |= has_text

        return "\n".join(parts), any_text

    elif isinstance(prompt, str):
        stripped = prompt.strip()
        return stripped, bool(stripped)

    else:
        print(f"Invalid prompt type: {type(prompt)}")
        sys.exit(1)

# TODO clean up / put in "utils" file or similar
###
# Recovery Helpers for weaker (local) LLM models that dont comply with strcutured responses
###
def _schema_instructions(schema: Type[Any]) -> str:
    """
    Create minimal, explicit field instructions for either a dataclass or a Pydantic model.
    """
    if is_dataclass(schema):
        lines = []
        for f in dc_fields(schema):
            # Keep it simple: strings are most common; show type name if available
            tname = getattr(f.type, "__name__", str(f.type))
            lines.append(f'- "{f.name}": {tname}')
        return "JSON object with exactly these keys:\n" + "\n".join(lines)

      # Pydantic v1 model fallback
    if hasattr(schema, "__fields__"):
        lines = []
        for name, f in schema.__fields__.items():  # type: ignore[attr-defined]
            tname = getattr(getattr(f, "type_", None), "__name__", "any")
            lines.append(f'- "{name}": {tname}')
        return "JSON object with exactly these keys:\n" + "\n".join(lines)

    return f"JSON object matching schema type {getattr(schema, '__name__', str(schema))}."

def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    # Remove ```json ... ``` or ``` ... ```
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()

def _coerce_faulty_output(agent_response: Any) -> str:
    """
    Extract the most useful 'faulty output' text from LangGraph/LangChain agent responses.
    This is intentionally simple: keep content, don't overthink.
    """
    if agent_response is None:
        return ""

    # If it's the standard dict response from create_agent
    if isinstance(agent_response, dict):
        # Prefer structured_response if present
        if "structured_response" in agent_response:
            return str(agent_response["structured_response"])

        msgs = agent_response.get("messages")
       
        if msgs:
            # last message content if available
            last = msgs[-1]
            return str(getattr(last, "content", last))
        return str(agent_response)

    # If already an AIMessage-like
    return str(getattr(agent_response, "content", agent_response))

def _is_empty_or_incoherent(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True

    bad_markers = ["commentary to=", "<|", "functions.", "}]**", "ListToolsRequest"]
    if any(m in t for m in bad_markers) and len(t) < 400:
        return True

    return False

def _parse_json_object(text: str) -> dict:
    """
    Parse a JSON object from text. If the model wraps JSON with extra text,
    try to salvage the first {...} block.
    """
    text = _strip_fences(text)

    try:
        obj = json.loads(text)
    except Exception:
        # salvage: find first JSON object substring
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise
        obj = json.loads(m.group(0))

    if not isinstance(obj, dict):
        raise ValueError(f"Expected JSON object, got {type(obj)}")
    return obj

def structured_output(
    llm: Any,
    schema: Type[Any],         
    prompt: str,
    max_retries: int = 2,
) -> Any:
    """
    Ask for JSON matching `schema`. If output isn't JSON, reprompt with:
    - faulty output
    - strict JSON-only rule
    - schema key instructions
    If faulty output is empty/incoherent -> regenerate from original prompt.
    """
    schema_text = _schema_instructions(schema)

    last_text = ""
    last_err = ""

    for attempt in range(max_retries + 1):
        if attempt == 0:
            cur_prompt = f"""
                        Return ONLY a single JSON object. No markdown, no prose, no backticks.

                        TARGET STRUCTURE:
                        {schema_text}

                        TASK:
                        {prompt}
                        """.strip()
        else:
            # If the last output was garbage, force regeneration rather than "convert"
            regen = _is_empty_or_incoherent(last_text)
            instruction = (
                "The previous output is empty/incoherent. Regenerate from scratch from the TASK."
                if regen
                else "Convert the previous output into the TARGET STRUCTURE JSON."
            )

            cur_prompt = f"""
                Return ONLY a single JSON object. No markdown, no prose, no backticks.

                TARGET STRUCTURE:
                {schema_text}

                TASK (reference):
                {prompt}

                PREVIOUS OUTPUT (convert if usable):
                {last_text}

                PARSING ERROR:
                {last_err}

                INSTRUCTION:
                {instruction}

                Now return ONLY the corrected JSON object:
                """.strip()

        resp = llm.invoke(cur_prompt)
        text = str(getattr(resp, "content", resp))
        last_text = text

        try:
            data = _parse_json_object(text)

            if is_dataclass(schema):
                allowed = {f.name for f in dc_fields(schema)}
            elif hasattr(schema, "__fields__"):  # pydantic v1
                allowed = set(schema.__fields__.keys())  # type: ignore[attr-defined]
            else:
                allowed = set(data.keys())

            filtered = {k: v for k, v in data.items() if k in allowed}

            return schema(**filtered)

        except Exception as e:
            last_err = repr(e)

    raise ValueError(
        f"Failed to produce valid structured output for {getattr(schema, '__name__', str(schema))} "
        f"after {max_retries + 1} attempts. Last error: {last_err}. Last output: {last_text}"
    )

def ensure_structured_agent_response(
    agent_response: Any,
    schema: Type[Any],          # <-- dataclass OR pydantic model
    llm: Any,
    original_prompt: str,
    max_attempts: int = 2
) -> Any:
    """
    If agent output didn't validate, pass the faulty output back to LLM and request
    strict JSON matching the target schema. If faulty output is empty/incoherent,
    regenerate from original prompt.
    """
    faulty_output = _coerce_faulty_output(agent_response)

    repair_task = f"""
                You must produce output for the original task, formatted as the target JSON structure.

                Original task:
                {original_prompt}

                Faulty output (use if helpful, otherwise regenerate):
                {faulty_output}
                """.strip()

    return structured_output(llm, schema, repair_task, max_retries=max_attempts)
