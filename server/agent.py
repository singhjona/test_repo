from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ValidationError

try:
    # These imports are expected to exist in the real server package.
    from server import config as server_config  # type: ignore
    from server import tools as server_tools  # type: ignore
except Exception:  # pragma: no cover - defensive import for editor context
    server_config = None  # type: ignore
    server_tools = None  # type: ignore


logger = logging.getLogger(__name__)


router = APIRouter(prefix="/agent", tags=["agent"])


ACTIVE_AGENT_TASKS: Dict[str, asyncio.Task[Any]] = {}
_SESSION_STATE: Dict[str, Dict[str, Any]] = {}
_SESSION_LOCKS: Dict[str, asyncio.Lock] = {}


def _get_session_lock(session_id: str) -> asyncio.Lock:
    if session_id not in _SESSION_LOCKS:
        _SESSION_LOCKS[session_id] = asyncio.Lock()
    return _SESSION_LOCKS[session_id]


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Stable key to maintain chat context across requests.")
    message: str = Field(..., description="User message to the agent.")
    metadata: Dict[str, Any] | None = Field(
        default=None,
        description="Optional opaque metadata forwarded to the planner/agent.",
    )


class PlannerTodo(BaseModel):
    id: str
    title: str
    description: str | None = None
    status: str = Field(
        ...,
        description="Todo status: pending, in_progress, or done.",
    )


class PlannerOutput(BaseModel):
    reasoning: str = Field(
        ...,
        description="Natural language reasoning explaining why these TODOS were created and how they address the request.",
    )
    todos: List[PlannerTodo]
    plan_notes: str | None = None
    task_complete: bool | None = Field(
        default=None,
        description="Optional flag that the overall task is already complete.",
    )


class AgentStatusUpdate(BaseModel):
    id: str
    status: str


class AgentStepOutput(BaseModel):
    selected_tool: str | None
    tool_input: Dict[str, Any] | None = None
    step_explanation: str
    new_todos: List[PlannerTodo] | None = None
    status_updates: List[AgentStatusUpdate] | None = None
    removed_todo_ids: List[str] | None = Field(
        default=None,
        description="Optional list of todo ids that should be removed because requirements have changed.",
    )


class SummarizerOutput(BaseModel):
    summary: str
    details: str | None = None
    needs_more_planning: bool = Field(
        ...,
        description=(
            "Set to true if more work/todos are still needed and the workflow should return to the planner; "
            "false if the task is fully complete and this summary can be shown as final."
        ),
    )


def _oss_base_url() -> str:
    url = getattr(server_config, "OSS_CHAT_URL", None) if server_config else None
    return (
        url
        or "https://ai-icp-ccaas.channels.euw1.dev.aws.cloud.hsbc/llm/qwen/v1/chat/completions"
    )


def _oss_model() -> str:
    model = getattr(server_config, "OSS_MODEL_PATH", None) if server_config else None
    return model or "/mnt/llm/llm_hosting/gpt-oss-20b/model_files"


def _oss_timeout() -> int:
    timeout = getattr(server_config, "OSS_TIMEOUT_SECONDS", None) if server_config else None
    return timeout or 120


def _oss_headers() -> Dict[str, str]:
    return {
        "Content-Type": "application/json",
        "User-Agent": "fast-mcp-agent/1.0",
    }


def _build_json_schema_from_pydantic(model: type[BaseModel]) -> Dict[str, Any]:
    schema = model.model_json_schema()
    return {
        "name": model.__name__,
        "description": f"Structured response schema for {model.__name__}.",
        "schema": schema,
        "strict": True,
    }


def _oss_call_json(
    messages: List[Dict[str, Any]],
    response_model: type[BaseModel],
    reasoning_effort: str = "medium",
    extra_payload: Optional[Dict[str, Any]] = None,
) -> BaseModel:
    """
    Call the OSS chat endpoint using streaming mode, but buffer the full
    structured output content before parsing it into `response_model`.

    The OSS API streams OpenAI-style chat completion chunks where each line
    is a JSON object with `choices[0].delta.content` fragments.
    """
    payload: Dict[str, Any] = {
        "messages": messages,
        "model": _oss_model(),
        "reasoning_effort": reasoning_effort,
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": _build_json_schema_from_pydantic(response_model),
        },
    }
    if extra_payload:
        payload.update(extra_payload)

    logger.debug(
        "Calling OSS with structured output (streaming) for %s", response_model.__name__
    )
    max_attempts = 5
    content: Any | None = None
    for attempt in range(1, max_attempts + 1):
        # Stream and accumulate delta.content fragments into a single string.
        content_buffer = ""
        try:
            resp = requests.post(
                _oss_base_url(),
                json=payload,
                headers=_oss_headers(),
                timeout=_oss_timeout(),
                verify=False,
                stream=True,
            )
        except Exception as exc:  # pragma: no cover - network failure
            logger.exception("OSS call failed: %s", exc)
            raise HTTPException(status_code=502, detail="Upstream model call failed.") from exc

        if resp.status_code != 200:
            try:
                from json import dumps as _dumps

                payload_str = _dumps(payload, indent=2, default=str)
            except Exception:  # pragma: no cover - logging best-effort
                payload_str = "<unserializable payload>"
            logger.error(
                "OSS returned non-200: %s %s\nPayload: %s",
                resp.status_code,
                resp.text,
                payload_str,
            )
            raise HTTPException(
                status_code=502,
                detail=f"Upstream model error: {resp.status_code}",
            )

        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    decoded = line.decode("utf-8").strip()
                except Exception:
                    continue
                if not decoded:
                    continue
                # Support both raw JSON lines and SSE-style "data: {...}" lines
                if decoded.startswith("data:"):
                    decoded = decoded[5:].strip()
                if decoded == "[DONE]":
                    break
                try:
                    chunk = json.loads(decoded)
                except Exception:
                    # Not a valid JSON chunk; skip
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                piece = delta.get("content")
                if isinstance(piece, str):
                    content_buffer += piece
        finally:
            resp.close()

        if not content_buffer and attempt < max_attempts:
            logger.warning(
                "OSS streaming returned empty content for %s; retrying (%d/%d)",
                response_model.__name__,
                attempt,
                max_attempts,
            )

        if not content_buffer:
            logger.error(
                "OSS streaming returned no content for %s on attempt %d/%d",
                response_model.__name__,
                attempt,
                max_attempts,
            )
            if attempt == max_attempts:
                raise HTTPException(
                    status_code=502,
                    detail="Model did not return any streamed content.",
                )
            continue

        # For structured output, the model is expected to stream a JSON string.
        try:
            content = json.loads(content_buffer)
        except json.JSONDecodeError as exc:
            logger.exception(
                "Failed to parse streamed content as JSON for %s on attempt %d/%d: %s",
                response_model.__name__,
                attempt,
                max_attempts,
                exc,
            )
            if attempt == max_attempts:
                raise HTTPException(
                    status_code=502,
                    detail="Model did not return valid structured JSON.",
                ) from exc
            # Retry on JSON parse errors.
            continue

        # Validate against the structured output schema.
        try:
            return response_model.model_validate(content)
        except ValidationError as exc:
            logger.exception(
                "Failed to validate structured output for %s on attempt %d/%d: %s",
                response_model.__name__,
                attempt,
                max_attempts,
                exc,
            )
            if attempt == max_attempts:
                raise HTTPException(
                    status_code=502,
                    detail="Model output did not match expected schema.",
                ) from exc
            # Retry on validation errors too, since they typically stem from malformed JSON fields.
            continue

    # This line should be unreachable, but keeps type-checkers happy.
    raise HTTPException(
        status_code=502,
        detail="Model did not return usable structured output.",
    )


def _oss_stream_text(
    messages: List[Dict[str, Any]],
    reasoning_effort: str = "medium",
) -> requests.Response:
    payload: Dict[str, Any] = {
        "messages": messages,
        "model": _oss_model(),
        "reasoning_effort": reasoning_effort,
        "stream": True,
    }
    logger.debug("Calling OSS with streaming response.")
    try:
        resp = requests.post(
            _oss_base_url(),
            json=payload,
            headers=_oss_headers(),
            timeout=_oss_timeout(),
            verify=False,
            stream=True,
        )
    except Exception as exc:  # pragma: no cover
        logger.exception("OSS streaming call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream model streaming failed.") from exc
    if resp.status_code != 200:
        try:
            from json import dumps as _dumps

            payload_str = _dumps(payload, indent=2, default=str)
        except Exception:  # pragma: no cover
            payload_str = "<unserializable payload>"
        logger.error(
            "OSS streaming returned non-200: %s %s\nPayload: %s",
            resp.status_code,
            resp.text,
            payload_str,
        )
        raise HTTPException(
            status_code=502,
            detail=f"Upstream model streaming error: {resp.status_code}",
        )
    return resp


def _build_available_tools() -> List[Dict[str, Any]]:
    tools: List[Dict[str, Any]] = []

    try:
        tools.append(
            {
                "name": "run_mongo_db_read_code_tool",
                "description": (
                    "Execute a SINGLE or BULK MongoDB READ operation(s) described by a JSON 'spec' "
                    "(or list of specs). Strictly read-only."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "spec": {
                            "description": "Mongo read spec object OR list of spec objects.",
                            "type": ["object", "array"],
                        },
                        "execution_mode": {
                            "description": "How to execute bulk items.",
                            "type": "string",
                            "enum": ["parallel", "sequential"],
                            "default": "parallel",
                        },
                    },
                    "required": ["spec"],
                },
                "examples": [
                    {
                        "description": "Find 10 users.",
                        "input": {
                            "spec": {
                                "operation": "find",
                                "collection": "ccaas.users",
                                "filter": {},
                                "limit": 10,
                            }
                        },
                    }
                ],
            }
        )

        tools.append(
            {
                "name": "describe_collection",
                "description": "Return a brief collection summary (count + lightweight schema) for one or more collections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "description": "Collection name OR list of names.",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                        },
                        "sample_limit": {
                            "type": "integer",
                            "minimum": 1,
                            "description": "Sample size per collection.",
                            "default": 100,
                        },
                        "execution_mode": {
                            "type": "string",
                            "enum": ["parallel", "sequential"],
                            "default": "parallel",
                        },
                    },
                    "required": ["collection"],
                },
                "examples": [
                    {
                        "description": "Describe orders.",
                        "input": {"collection": "ccaas.orders"},
                    }
                ],
            }
        )

        tools.append(
            {
                "name": "describe_database",
                "description": "List collections in the default database and return basic stats.",
                "parameters": {"type": "object", "properties": {}, "required": []},
                "examples": [{"description": "Describe default database.", "input": {}}],
            }
        )

        tools.append(
            {
                "name": "get_documents",
                "description": "Fetch a SMALL, bounded set of documents for inspection from one or more collections.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "description": "Collection name OR list of names.",
                            "oneOf": [
                                {"type": "string"},
                                {"type": "array", "items": {"type": "string"}},
                            ],
                        },
                        "selection_mode": {
                            "type": "string",
                            "enum": ["first", "last", "random", "filter", "skip", "sample"],
                            "default": "first",
                        },
                        "selection_filter": {"type": ["object", "null"]},
                        "projection": {"type": ["object", "null"]},
                        "sort": {"type": ["object", "array", "null"]},
                        "number_of_documents": {"type": "integer", "default": 10, "minimum": 1},
                        "skip": {"type": "integer", "default": 0, "minimum": 0},
                        "seed": {"type": ["integer", "null"]},
                        "execution_mode": {
                            "type": "string",
                            "enum": ["parallel", "sequential"],
                            "default": "parallel",
                        },
                    },
                    "required": ["collection"],
                },
                "examples": [
                    {
                        "description": "Sample 5 users.",
                        "input": {
                            "collection": "ccaas.users",
                            "selection_mode": "sample",
                            "number_of_documents": 5,
                        },
                    }
                ],
            }
        )

        tools.append(
            {
                "name": "profile_collection_anomalies",
                "description": (
                    "STEP 1 of 2 (MUST RUN FIRST): Create a bounded anomaly profile for a collection. "
                    "Returns deterministic metrics (presence/nulls/type counts/basic stats/top values). "
                    "Use this BEFORE calling anomalies_tools or fix_anomalies."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": (
                                "Collection name to profile. Accepts 'collection' or 'db.collection'. "
                                "Examples: 'users', 'ccaas.users'."
                            ),
                        }
                    },
                    "required": ["collection"],
                },
                "examples": [
                    {
                        "description": "Profile anomalies for users.",
                        "input": {"collection": "ccaas.users"},
                    }
                ],
            }
        )

        tools.append(
            {
                "name": "anomalies_tools",
                "description": (
                    "STEP 2 of 2 (MUST RUN AFTER profiling): Execute SMALL, targeted anomaly checks derived "
                    "from profile_collection_anomalies. Supported actions: dependency_check, statistical_outlier, "
                    "entropy_check, fuzzy_match_anomaly, velocity_check."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": (
                                "Collection name to check. Accepts 'collection' or 'db.collection'. "
                                "Examples: 'orders', 'ccaas.orders'."
                            ),
                        },
                        "checks": {
                            "type": "array",
                            "description": (
                                "List of SMALL (1-5) check specifications derived from the profile. "
                                "Each item MUST include 'action'."
                            ),
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["collection", "checks"],
                },
                "examples": [
                    {
                        "description": "Run a dependency_check anomaly.",
                        "input": {
                            "collection": "ccaas.orders",
                            "checks": [
                                {
                                    "action": "dependency_check",
                                    "if_field": "status",
                                    "is_value": "active",
                                    "then_field": "end_date",
                                    "must_not_be": None,
                                }
                            ],
                        },
                    },
                    {
                        "description": "Run a statistical_outlier anomaly.",
                        "input": {
                            "collection": "ccaas.orders",
                            "checks": [
                                {
                                    "action": "statistical_outlier",
                                    "field": "amount",
                                    "method": "percentile",
                                    "threshold": 99.5,
                                }
                            ],
                        },
                    },
                ],
            }
        )

        tools.append(
            {
                "name": "fix_anomalies",
                "description": (
                    "NEVER RUN AUTOMATICALLY UNLESS EXPLICITLY APPROVED (HUMAN IN THE LOOP). "
                    "Immediately apply bounded fixes for anomalies. Writes are gated by ENABLE_WRITE_TOOLS."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": (
                                "Collection to fix. Accepts 'collection' or 'db.collection'. "
                                "Example: 'ccaas.users'."
                            ),
                        },
                        "fixes": {
                            "type": "array",
                            "description": (
                                "List of deterministic fix specifications. Each fix must include: "
                                "fix_id, description, precondition, write, postcondition."
                            ),
                            "items": {"type": "object"},
                        },
                        "approved": {
                            "type": "boolean",
                            "description": "Ignored flag kept for backward compatibility.",
                            "default": False,
                        },
                        "approval_id": {
                            "type": ["string", "null"],
                            "description": "Ignored approval id kept for backward compatibility.",
                            "default": None,
                        },
                    },
                    "required": ["collection", "fixes"],
                },
                "examples": [
                    {
                        "description": "Apply a single bounded anomaly fix.",
                        "input": {
                            "collection": "ccaas.users",
                            "fixes": [
                                {
                                    "fix_id": "normalize_status_values",
                                    "description": "Normalize status field to lowercase.",
                                    "precondition": {
                                        "filter": {"status": {"$exists": True}},
                                        "expected_count_max": 1000,
                                    },
                                    "write": {
                                        "operation": "update_many",
                                        "filter": {},
                                        "update": {"$set": {"status": "active"}},
                                    },
                                    "postcondition": {
                                        "must_be_zero_count_filter": {
                                            "status": {"$in": ["ACTIVE", "Active"]}
                                        }
                                    },
                                }
                            ],
                            "approved": True,
                        },
                    }
                ],
            }
        )

        tools.append(
            {
                "name": "create_collection_tool",
                "description": (
                    "Create a collection and optional indexes. Gated by ENABLE_WRITE_TOOLS. "
                    "Validates collection name and limited options (capped, size, max, validator, indexes)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "database": {
                            "type": ["string", "null"],
                            "description": (
                                "Optional database name. If omitted, defaults to 'ccaas'. Accepts 'db' only."
                            ),
                        },
                        "collection": {
                            "type": "string",
                            "description": (
                                "Collection name (no database). Example: 'users'. "
                                "Pass 'db.collection' in database param if needed."
                            ),
                        },
                        "options": {
                            "type": ["object", "null"],
                            "description": (
                                "Optional creation options: allowed keys: "
                                "capped(bool), size(int), max(int), validator(dict limited), indexes(list of index specs)."
                            ),
                        },
                    },
                    "required": ["collection"],
                },
                "examples": [
                    {
                        "description": "Create a capped users collection with an index.",
                        "input": {
                            "database": "ccaas",
                            "collection": "users",
                            "options": {
                                "capped": True,
                                "size": 1048576,
                                "indexes": [
                                    {"keys": [["email", 1]], "unique": True},
                                ],
                            },
                        },
                    }
                ],
            }
        )

        tools.append(
            {
                "name": "insert_documents_tool",
                "description": (
                    "Insert documents into a collection (gated by ENABLE_WRITE_TOOLS). "
                    "Validates and sanitizes documents, enforces batch size limits, and writes via the write runner."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "collection": {
                            "type": "string",
                            "description": (
                                "Target collection. Accepts 'collection' or 'db.collection'. "
                                "Example: 'ccaas.users'."
                            ),
                        },
                        "documents": {
                            "type": "array",
                            "description": (
                                "Non-empty list of documents to insert. Each item must be an object."
                            ),
                            "items": {"type": "object"},
                        },
                        "ordered": {
                            "type": "boolean",
                            "description": (
                                "If true, insert is ordered and stops on first failure. Default true."
                            ),
                            "default": True,
                        },
                        "bypass_document_validation": {
                            "type": "boolean",
                            "description": "Ignored for now; kept for API parity.",
                            "default": False,
                        },
                    },
                    "required": ["collection", "documents"],
                },
                "examples": [
                    {
                        "description": "Insert two users.",
                        "input": {
                            "collection": "ccaas.users",
                            "documents": [
                                {"email": "user1@example.com", "status": "active"},
                                {"email": "user2@example.com", "status": "inactive"},
                            ],
                        },
                    }
                ],
            }
        )
    except Exception:  # pragma: no cover - defensive
        logger.exception("Failed to build AVAILABLE_TOOLS metadata.")

    return tools


AVAILABLE_TOOLS: List[Dict[str, Any]] = _build_available_tools()


def _get_session(session_id: str) -> Dict[str, Any]:
    if session_id not in _SESSION_STATE:
        _SESSION_STATE[session_id] = {
            "messages": [],
            "todos": [],
            "metadata": {},
        }
    return _SESSION_STATE[session_id]


def _append_message(
    session: Dict[str, Any],
    role: str,
    content: str,
) -> None:
    session["messages"].append({"role": role, "content": content})


async def _planner_phase(
    session_id: str,
    user_message: str,
) -> PlannerOutput:
    session = _get_session(session_id)
    # Make the planner explicitly aware of available tools so it can
    # propose tool-aware todos (e.g., which Mongo tools to call).
    tools_description = json.dumps(
        AVAILABLE_TOOLS,
        indent=2,
        default=str,
    )
    system_prompt = (
        "You are a planner agent that coordinates work for an execution agent which has access to tools.\n\n"
        "GOALS:\n"
        "- Given the latest user request and context, produce a structured PlannerOutput JSON object.\n"
        "- First, think step by step and write a clear natural-language 'reasoning' string explaining your plan.\n"
        "- Then, create a list of TODOS, where each todo has id, title, optional description, and "
        "status in {pending,in_progress,done}.\n"
        "- Todos SHOULD explicitly reference which tool(s) they expect the execution agent to use where relevant.\n\n"
        "TOOLS AVAILABLE TO THE EXECUTION AGENT (for your planning only):\n"
        f"{tools_description}\n\n"
        "IMPORTANT:\n"
        "- You are only PLANNING; you MUST NOT invent tool results.\n"
        "- Return ONLY valid JSON matching the PlannerOutput schema. Do not include prose outside JSON.\n"
        "- If you output anything that is not valid JSON for this schema, you have FAILED the task."
    )

    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(session["messages"])
    messages.append({"role": "user", "content": user_message})

    output = _oss_call_json(messages, PlannerOutput)
    session["todos"] = [todo.model_dump() for todo in output.todos]
    _append_message(session, "assistant", f"Planner TODOS: {json.dumps(session['todos'])}")
    return output


async def _invoke_tool(
    name: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if server_tools is None:
        raise HTTPException(status_code=500, detail="Tools module not available.")

    if not hasattr(server_tools, name):
        raise HTTPException(status_code=400, detail=f"Unknown tool: {name}")

    fn = getattr(server_tools, name)
    try:
        result = fn(**params)
    except Exception as exc:  # pragma: no cover - tool failure
        logger.exception("Tool %s failed: %s", name, exc)
        return {"success": False, "error": str(exc)}

    if isinstance(result, dict) and "success" in result:
        return result
    return {"success": True, "result": result}


async def _agent_loop(
    session_id: str,
    send_event,
    max_iterations: int = 32,
) -> None:
    session = _get_session(session_id)

    for iteration in range(max_iterations):
        todos: List[Dict[str, Any]] = session.get("todos", [])
        pending = [t for t in todos if t.get("status") != "done"]
        if not pending:
            break

        current = pending[0]
        system_prompt = (
            "You are an execution agent. You have access to TOOLS and must choose at most one tool per step. "
            "Return JSON matching AgentStepOutput: selected_tool (or null), tool_input (or null), "
            "step_explanation, optional new_todos, status_updates, and optional removed_todo_ids "
            "for todos that are no longer needed because requirements changed. "
            "TOOLS metadata is provided in the system message; only use listed tools.\n\n"
            "CRITICAL:\n"
            "- You MUST return ONLY valid JSON matching the AgentStepOutput schema.\n"
            "- Do not include any explanation outside the JSON.\n"
            "- If you output anything that is not valid JSON for this schema, you have FAILED the task."
        )
        tools_description = json.dumps(AVAILABLE_TOOLS)

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt + "\n\nTOOLS:\n" + tools_description,
            }
        ]
        messages.extend(session["messages"])
        messages.append(
            {
                "role": "user",
                "content": f"Current todo: {json.dumps(current)}\nAll todos: {json.dumps(todos)}",
            }
        )

        step_output = _oss_call_json(messages, AgentStepOutput)

        if step_output.status_updates:
            for upd in step_output.status_updates:
                for t in todos:
                    if t.get("id") == upd.id:
                        t["status"] = upd.status

        if step_output.new_todos:
            todos.extend([nt.model_dump() for nt in step_output.new_todos])

        if step_output.removed_todo_ids:
            remove_ids = set(step_output.removed_todo_ids)
            todos = [t for t in todos if t.get("id") not in remove_ids]

        session["todos"] = todos

        await send_event(
            "agent_step",
            {
                "iteration": iteration,
                "step_explanation": step_output.step_explanation,
                "todos": todos,
            },
        )

        if step_output.selected_tool:
            tool_name = step_output.selected_tool
            tool_params = step_output.tool_input or {}
            await send_event(
                "tool_call",
                {"name": tool_name, "input": tool_params},
            )
            tool_result = await _invoke_tool(tool_name, tool_params)
            await send_event(
                "tool_result",
                {"name": tool_name, "result": tool_result},
            )
            _append_message(
                session,
                "assistant",
                f"Tool {tool_name} invoked with {json.dumps(tool_params)} -> {json.dumps(tool_result)}",
            )
    else:
        await send_event(
            "agent_abort",
            {"reason": "max_iterations_reached"},
        )


async def _summarizer_phase(
    session_id: str,
    user_message: str,
    send_event,
) -> SummarizerOutput:
    session = _get_session(session_id)
    system_prompt = (
        "You are a summarizer/controller. Look at the full history of messages, TODOS, and tool calls. "
        "Decide whether the user's request is fully satisfied or if more planning/work is needed.\n\n"
        "Return ONLY a SummarizerOutput JSON object:\n"
        "- summary: concise natural-language summary of what has been done so far.\n"
        "- details: optional extra detail.\n"
        "- needs_more_planning: true if more work/todos are needed and the workflow should go back to the planner; "
        "false if the task is complete and the summary can be shown as final.\n\n"
        "CRITICAL:\n"
        "- You MUST return ONLY valid JSON matching the SummarizerOutput schema.\n"
        "- Do not include any explanation outside the JSON.\n"
        "- If you output anything that is not valid JSON for this schema, you have FAILED the task."
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(session["messages"])
    messages.append(
        {"role": "user", "content": f"Summarize the work performed for: {user_message}"},
    )

    summary = _oss_call_json(messages, SummarizerOutput)
    _append_message(session, "assistant", f"Summary: {summary.summary}")
    await send_event(
        "summary",
        summary.model_dump(),
    )
    return summary


async def _stream_model_reasoning(
    session_id: str,
    user_message: str,
    send_event,
) -> None:
    session = _get_session(session_id)
    system_prompt = (
        "Think step by step about how you will approach the user's request. "
        "This stream is only for the user UI and does not need strict structure."
    )
    messages: List[Dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    messages.extend(session["messages"])
    messages.append({"role": "user", "content": user_message})

    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(
        None,
        lambda: _oss_stream_text(messages),
    )

    try:
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                decoded = line.decode("utf-8").strip()
            except Exception:
                continue
            if not decoded:
                continue
            # Handle SSE-style prefix
            if decoded.startswith("data:"):
                decoded = decoded[5:].strip()
            if decoded == "[DONE]":
                break
            try:
                chunk = json.loads(decoded)
            except Exception:
                # If the gateway sends plain text instead of JSON, pass it through.
                await send_event("model_stream", {"delta": decoded})
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content")
            if isinstance(piece, str):
                # Push small deltas immediately so the UI updates in near real-time.
                await send_event("model_stream", {"delta": piece})
    finally:
        resp.close()


async def _agent_session_stream(request: ChatRequest) -> AsyncGenerator[bytes, None]:
    session_id = request.session_id
    lock = _get_session_lock(session_id)

    async with lock:
        session = _get_session(session_id)
        _append_message(session, "user", request.message)

    async def send_event(event_type: str, data: Dict[str, Any]) -> None:
        # Debug logging so we can correlate server-side event emission with
        # what the frontend sees (or does not see) over SSE.
        try:
            preview = json.dumps(data, default=str)
        except Exception:
            preview = "<unserializable>"
        logger.debug("SSE enqueue -> event=%s data=%s", event_type, preview[:512])

        payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
        await queue.put(payload.encode("utf-8"))

    queue: asyncio.Queue[bytes] = asyncio.Queue()

    async def runner() -> None:
        try:
            await send_event("start", {"session_id": session_id})
            await _stream_model_reasoning(session_id, request.message, send_event)
            max_cycles = 4
            for cycle in range(max_cycles):
                planner_output = await _planner_phase(session_id, request.message)
                await send_event(
                    "planner_reasoning",
                    {
                        "reasoning": planner_output.reasoning,
                        "cycle": cycle,
                    },
                )
                await send_event(
                    "planner_todos",
                    {
                        "todos": [t.model_dump() for t in planner_output.todos],
                        "plan_notes": planner_output.plan_notes,
                        "cycle": cycle,
                    },
                )
                await _agent_loop(session_id, send_event)
                summary = await _summarizer_phase(session_id, request.message, send_event)
                if not summary.needs_more_planning:
                    break
            await send_event("end", {"session_id": session_id})
        except HTTPException as exc:
            await send_event(
                "error",
                {"status_code": exc.status_code, "detail": exc.detail},
            )
        except Exception as exc:  # pragma: no cover
            logger.exception("Unexpected error in agent session: %s", exc)
            await send_event(
                "error",
                {"status_code": 500, "detail": "Internal agent error."},
            )
        finally:
            await queue.put(b"")

    task = asyncio.create_task(runner())
    ACTIVE_AGENT_TASKS[session_id] = task

    while True:
        chunk = await queue.get()
        if not chunk:
            break
        # Debug logging for what is actually being streamed to the client.
        logger.debug("SSE send -> bytes=%d chunk_preview=%r", len(chunk), chunk[:120])
        yield chunk


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    generator = _agent_session_stream(request)
    return StreamingResponse(generator, media_type="text/event-stream")

