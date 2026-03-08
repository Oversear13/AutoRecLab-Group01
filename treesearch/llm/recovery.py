from __future__ import annotations

import json
import re
from dataclasses import is_dataclass, fields as dc_fields, asdict
from typing import Any, Iterable, Type


class StructuredRecoveryPolicy:
    def __init__(
        self,
        max_attempts: int = 3,
        bad_markers: Iterable[str] | None = None,
        incoherent_len_threshold: int = 400,
    ) -> None:
        self.max_attempts = max_attempts
        self.bad_markers = list(
            bad_markers
            if bad_markers is not None
            else ["commentary to=", "<|", "functions.", "}]**", "ListToolsRequest"]
        )
        self.incoherent_len_threshold = incoherent_len_threshold


def _schema_instructions(schema: Type[Any]) -> str:
    if is_dataclass(schema):
        lines = []
        for f in dc_fields(schema):
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
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


def _serialize_structured_value(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value).strip()

    if is_dataclass(value):
        try:
            return json.dumps(asdict(value), ensure_ascii=False)
        except Exception:
            return str(value).strip()

    if hasattr(value, "dict") and callable(getattr(value, "dict")):
        try:
            return json.dumps(value.dict(), ensure_ascii=False)
        except Exception:
            return str(value).strip()

    if hasattr(value, "model_dump") and callable(getattr(value, "model_dump")):
        try:
            return json.dumps(value.model_dump(), ensure_ascii=False)
        except Exception:
            return str(value).strip()

    return str(value).strip()


def _extract_text(value: Any) -> str:
    """
    Robustly unwrap text/content from:
    - plain strings
    - LangChain AIMessage / message-like objects with .content
    - OpenAI Responses API style blocks:
      [{"type": "text", "text": "..."}]
    - dict/list containers with nested content
    Falls back to string conversion only when necessary.
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    # Message-like objects
    content = getattr(value, "content", None)
    if content is not None and content is not value:
        return _extract_text(content)

    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            extracted = _extract_text(item)
            if extracted:
                parts.append(extracted)
        return "\n".join(parts).strip()

    if isinstance(value, dict):
        # Common Responses API style text block
        if value.get("type") == "text" and "text" in value:
            return str(value["text"]).strip()

        # Other possible text keys
        for key in ("text", "content", "output_text"):
            if key in value:
                return _extract_text(value[key])

        # Sometimes nested under message/output containers
        for key in ("message", "output", "data"):
            if key in value:
                extracted = _extract_text(value[key])
                if extracted:
                    return extracted

        return str(value).strip()

    return str(value).strip()


def _coerce_faulty_output(agent_response: Any) -> str:
    """
    Preserve existing behavior (use structured_response/messages/content when available),
    but avoid destroying valid JSON by converting rich objects into Python repr strings
    too early.
    """
    if agent_response is None:
        return ""

    if isinstance(agent_response, dict):
        if "structured_response" in agent_response and agent_response["structured_response"] is not None:
            return _serialize_structured_value(agent_response["structured_response"])

        msgs = agent_response.get("messages")
        if msgs:
            last = msgs[-1]
            return _extract_text(last)

        return _extract_text(agent_response)

    return _extract_text(agent_response)


def _is_empty_or_incoherent(text: str, policy: StructuredRecoveryPolicy) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    if any(m in t for m in policy.bad_markers) and len(t) < policy.incoherent_len_threshold:
        return True
    return False


def _extract_json_candidate_from_text(text: str) -> str:
    """
    Try to locate the most likely JSON object substring if the model wrapped it in prose.
    """
    text = _strip_fences(text)

    # Fast path: already looks like an object
    if text.startswith("{") and text.endswith("}"):
        return text

    # Try to find the first object-like region
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)

    return text


def _parse_json_object(text: str) -> dict:
    """
    Accepts:
    - plain JSON object text
    - fenced JSON
    - JSON embedded in prose
    - outer list/dict wrappers that contain a text block with JSON
    - JSON string containing an object
    """
    text = _strip_fences(text)

    # First attempt: direct parse
    try:
        obj = json.loads(text)
    except Exception:
        obj = None

    if isinstance(obj, dict):
        return obj

    # Responses-style list of blocks or other wrapped payloads
    if isinstance(obj, list):
        extracted = _extract_text(obj)
        if extracted and extracted != text:
            return _parse_json_object(extracted)

    # Sometimes the parsed JSON itself is a stringified JSON object
    if isinstance(obj, str):
        return _parse_json_object(obj)

    # Try extracting object-like substring from raw text
    candidate = _extract_json_candidate_from_text(text)
    if candidate != text:
        obj = json.loads(candidate)
        if isinstance(obj, dict):
            return obj

        if isinstance(obj, str):
            return _parse_json_object(obj)

    raise ValueError(f"Expected JSON object, got unparsable text: {text!r}")


def _allowed_keys(schema: Type[Any]) -> set[str]:
    if is_dataclass(schema):
        return {f.name for f in dc_fields(schema)}
    if hasattr(schema, "__fields__"):  # pydantic v1
        return set(schema.__fields__.keys())  # type: ignore[attr-defined]
    return set()


def _instantiate_schema(schema: Type[Any], data: dict) -> Any:
    allowed = _allowed_keys(schema)
    filtered = {k: v for k, v in data.items() if (k in allowed) or not allowed}
    return schema(**filtered)


def structured_output(
    *,
    llm: Any,
    schema: Type[Any],
    task_prompt: str,
    policy: StructuredRecoveryPolicy,
) -> Any:
    """
    Existing behavior is preserved:
    - attempt direct schema JSON generation first
    - retry by converting/repairing prior output

    Added robustness:
    - handles Responses-style content blocks
    - handles fenced JSON
    - handles JSON hidden inside list/dict/text wrappers
    """
    schema_text = _schema_instructions(schema)

    last_text = ""
    last_err = ""

    for attempt in range(policy.max_attempts + 1):
        if attempt == 0:
            cur_prompt = f"""
Return ONLY a single JSON object. No markdown, no prose, no backticks.

TARGET STRUCTURE:
{schema_text}

TASK:
{task_prompt}
""".strip()
        else:
            regen = _is_empty_or_incoherent(last_text, policy)
            instruction = (
                "The previous output is empty, malformed, or incoherent. Regenerate from scratch from the TASK."
                if regen
                else "Convert the previous output into the TARGET STRUCTURE JSON."
            )

            cur_prompt = f"""
Return ONLY a single JSON object. No markdown, no prose, no backticks.

TARGET STRUCTURE:
{schema_text}

TASK (reference):
{task_prompt}

PREVIOUS OUTPUT (convert if usable):
{last_text}

PARSING ERROR:
{last_err}

INSTRUCTION:
{instruction}

Now return ONLY the corrected JSON object:
""".strip()

        resp = llm.invoke(cur_prompt)
        text = _extract_text(resp)
        last_text = text

        try:
            data = _parse_json_object(text)
            return _instantiate_schema(schema, data)
        except Exception as e:
            last_err = repr(e)

    raise ValueError(
        f"Failed to produce valid structured output for {getattr(schema, '__name__', str(schema))} "
        f"after {policy.max_attempts + 1} attempts. Last error: {last_err}. Last output: {last_text}"
    )


def ensure_structured_agent_response(
    *,
    agent_response: Any,
    schema: Type[Any],
    llm: Any,
    original_prompt: str,
    policy: StructuredRecoveryPolicy,
) -> Any:
    faulty_output = _coerce_faulty_output(agent_response)

    repair_task = f"""
You must produce output for the original task, formatted as the target JSON structure.

Original task:
{original_prompt}

Faulty output (use if helpful, otherwise regenerate):
{faulty_output}
""".strip()

    return structured_output(
        llm=llm,
        schema=schema,
        task_prompt=repair_task,
        policy=policy,
    )