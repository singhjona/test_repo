from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

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
    todo_id: str
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
    payload: Dict[str, Any] = {
        "messages": messages,
        "model": _oss_model(),
        "reasoning_effort": reasoning_effort,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": _build_json_schema_from_pydantic(response_model),
        },
    }
    if extra_payload:
        payload.update(extra_payload)

    logger.debug("Calling OSS with structured output: %s", response_model.__name__)
    try:
        resp = requests.post(
            _oss_base_url(),
            json=payload,
            headers=_oss_headers(),
            timeout=_oss_timeout(),
            verify=False,
        )
    except Exception as exc:  # pragma: no cover - network failure
        logger.exception("OSS call failed: %s", exc)
        raise HTTPException(status_code=502, detail="Upstream model call failed.") from exc

    if resp.status_code != 200:
        logger.error("OSS returned non-200: %s %s", resp.status_code, resp.text)
        raise HTTPException(
            status_code=502,
            detail=f"Upstream model error: {resp.status_code}",
        )

    try:
        data = resp.json()
    except Exception as exc:  # pragma: no cover - malformed JSON
        logger.exception("OSS returned invalid JSON: %s", exc)
        raise HTTPException(status_code=502, detail="Invalid JSON from model.") from exc

    content = data.get("choices", [{}])[0].get("message", {}).get("content")
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError as exc:
            logger.exception("Failed to parse content as JSON: %s", exc)
            raise HTTPException(
                status_code=502,
                detail="Model did not return valid structured JSON.",
            ) from exc

    try:
        return response_model.model_validate(content)
    except Exception as exc:
        logger.exception("Failed to validate structured output: %s", exc)
        raise HTTPException(
            status_code=502,
            detail="Model output did not match expected schema.",
        ) from exc


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
        logger.error("OSS streaming returned non-200: %s %s", resp.status_code, resp.text)
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
    system_prompt = (
        "You are a planner agent. Given the latest user request and context, "
        "produce a structured PlannerOutput JSON object. "
        "You MUST:\n"
        "- First, think step by step and write a clear natural-language 'reasoning' string explaining your plan.\n"
        "- Then, create a list of TODOS, where each todo has id, title, optional description, and "
        "status in {pending,in_progress,done}.\n"
        "Return ONLY valid JSON matching the PlannerOutput schema."
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
            "TOOLS metadata is provided in the system message; only use listed tools."
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
                    if t.get("id") == upd.todo_id:
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
        "false if the task is complete and the summary can be shown as final."
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

    def iterate_chunks():
        try:
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    decoded = line.decode("utf-8")
                except Exception:
                    continue
                yield decoded
        finally:
            resp.close()

    for chunk in iterate_chunks():
        await send_event("model_stream", {"delta": chunk})


async def _agent_session_stream(request: ChatRequest) -> AsyncGenerator[bytes, None]:
    session_id = request.session_id
    lock = _get_session_lock(session_id)

    async with lock:
        session = _get_session(session_id)
        _append_message(session, "user", request.message)

    async def send_event(event_type: str, data: Dict[str, Any]) -> None:
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
        yield chunk


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    generator = _agent_session_stream(request)
    return StreamingResponse(generator, media_type="text/event-stream")








<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <title>Agentic Mongo Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <link
      href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap"
      rel="stylesheet"
    />
    <style>
      :root {
        --bg: #050712;
        --bg-elevated: #0d1024;
        --bg-elevated-softer: #11152b;
        --accent: #46c2ff;
        --accent-soft: rgba(70, 194, 255, 0.16);
        --accent-strong: #38a6da;
        --danger: #ff4f6d;
        --success: #54e9b5;
        --text: #f5f7ff;
        --text-muted: #9aa0c6;
        --border-subtle: #272c4a;
        --shadow-soft: 0 18px 40px rgba(3, 7, 28, 0.85);
        --radius-lg: 18px;
        --radius-md: 12px;
        --radius-pill: 999px;
      }

      * {
        box-sizing: border-box;
      }

      html,
      body {
        margin: 0;
        padding: 0;
        height: 100%;
        background: radial-gradient(circle at top left, #182340, #050712 60%);
        color: var(--text);
        font-family: "Plus Jakarta Sans", system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      body {
        display: flex;
        justify-content: center;
        align-items: stretch;
      }

      #app {
        max-width: 1320px;
        width: 100%;
        padding: 18px 18px 24px;
        display: flex;
        flex-direction: column;
        gap: 16px;
      }

      header.top-nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 14px 18px;
        border-radius: var(--radius-lg);
        background: linear-gradient(
          120deg,
          rgba(70, 194, 255, 0.1),
          rgba(39, 44, 74, 0.6),
          rgba(6, 10, 35, 0.95)
        );
        box-shadow: var(--shadow-soft);
      }

      .brand {
        display: flex;
        align-items: center;
        gap: 12px;
      }

      .brand-mark {
        width: 32px;
        height: 32px;
        border-radius: 12px;
        background: radial-gradient(circle at 30% 20%, #ffffff, #46c2ff 40%, #070b2a 80%);
        position: relative;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5);
      }

      .brand-mark::before,
      .brand-mark::after {
        content: "";
        position: absolute;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.6);
      }

      .brand-mark::before {
        inset: 8px 7px;
        border-left-color: transparent;
      }

      .brand-mark::after {
        inset: 10px 8px;
        border-right-color: transparent;
        opacity: 0.6;
      }

      .brand-text-title {
        font-size: 16px;
        font-weight: 600;
        letter-spacing: 0.02em;
      }

      .brand-text-subtitle {
        font-size: 12px;
        color: var(--text-muted);
      }

      .nav-controls {
        display: inline-flex;
        align-items: center;
        gap: 10px;
      }

      .pill {
        border-radius: var(--radius-pill);
        padding: 6px 12px;
        background: rgba(7, 11, 42, 0.9);
        border: 1px solid rgba(115, 126, 180, 0.5);
        font-size: 11px;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-muted);
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }

      .pill-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: #16c47f;
        box-shadow: 0 0 0 6px rgba(22, 196, 127, 0.18);
      }

      .btn-secondary {
        border-radius: var(--radius-pill);
        padding: 7px 16px;
        border: 1px solid rgba(105, 114, 168, 0.8);
        background: rgba(10, 13, 40, 0.9);
        color: var(--text);
        font-size: 12px;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }

      .btn-secondary:hover {
        background: rgba(17, 22, 63, 0.95);
      }

      .main-layout {
        flex: 1;
        display: grid;
        grid-template-columns: 1.4fr 1.0fr 1.2fr;
        gap: 16px;
        min-height: 0;
      }

      .column {
        background: linear-gradient(145deg, var(--bg-elevated), var(--bg-elevated-softer));
        border-radius: var(--radius-lg);
        padding: 12px 12px 10px;
        box-shadow: var(--shadow-soft);
        border: 1px solid rgba(32, 40, 84, 0.9);
        display: flex;
        flex-direction: column;
        min-height: 0;
      }

      .column-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 6px;
      }

      .column-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-muted);
      }

      .badge {
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 999px;
        background: rgba(33, 40, 92, 0.9);
        border: 1px solid rgba(72, 84, 176, 0.9);
        color: var(--text-muted);
      }

      .column-body {
        flex: 1;
        min-height: 0;
        overflow: hidden;
        position: relative;
      }

      .scroll-panel {
        position: absolute;
        inset: 0;
        padding: 4px 4px 6px;
        overflow-y: auto;
        scrollbar-width: thin;
        scrollbar-color: rgba(70, 194, 255, 0.55) rgba(5, 7, 18, 0.6);
      }

      .scroll-panel::-webkit-scrollbar {
        width: 7px;
      }
      .scroll-panel::-webkit-scrollbar-track {
        background: rgba(5, 7, 18, 0.8);
      }
      .scroll-panel::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent), var(--accent-strong));
        border-radius: 999px;
      }

      /* Conversation */
      #conversation-list {
        display: flex;
        flex-direction: column;
        gap: 8px;
        padding-bottom: 64px;
      }

      .msg {
        max-width: 92%;
        padding: 9px 11px;
        border-radius: 14px;
        font-size: 13px;
        line-height: 1.45;
        position: relative;
        box-shadow: 0 8px 22px rgba(1, 3, 22, 0.8);
        animation: fadeInUp 0.16s ease-out;
      }

      .msg-user {
        align-self: flex-end;
        background: linear-gradient(135deg, var(--accent-strong), #73ffe6);
        color: #02030d;
        border-bottom-right-radius: 4px;
      }

      .msg-assistant {
        align-self: flex-start;
        background: radial-gradient(circle at top left, #20275a, #070b25);
        border-bottom-left-radius: 4px;
        border: 1px solid rgba(71, 83, 168, 0.8);
      }

      .msg-reasoning {
        align-self: stretch;
        border-radius: 12px;
        border: 1px dashed rgba(107, 126, 211, 0.7);
        background: radial-gradient(circle at top, rgba(70, 194, 255, 0.18), rgba(7, 10, 34, 0.9));
        color: var(--text);
        font-size: 12px;
      }

      .msg-meta {
        margin-top: 3px;
        font-size: 10px;
        color: var(--text-muted);
        opacity: 0.8;
      }

      .markdown {
        font-size: 13px;
      }

      .markdown h1,
      .markdown h2,
      .markdown h3 {
        font-size: 14px;
        margin: 6px 0 4px;
      }

      .markdown p {
        margin: 3px 0;
      }

      .markdown ul,
      .markdown ol {
        padding-left: 18px;
        margin: 3px 0;
      }

      .markdown code {
        background: rgba(5, 10, 35, 0.9);
        padding: 1px 4px;
        border-radius: 4px;
        font-size: 12px;
      }

      .markdown pre {
        background: rgba(4, 7, 28, 0.95);
        padding: 8px 9px;
        border-radius: 8px;
        overflow-x: auto;
        border: 1px solid rgba(62, 77, 155, 0.95);
      }

      /* Input area */
      .input-shell {
        margin-top: 6px;
        padding: 7px 9px;
        border-radius: var(--radius-md);
        background: linear-gradient(145deg, rgba(7, 9, 32, 0.98), rgba(14, 19, 55, 0.98));
        border: 1px solid rgba(54, 66, 135, 0.9);
        display: flex;
        align-items: flex-end;
        gap: 8px;
        box-shadow: 0 14px 40px rgba(0, 0, 0, 0.7);
      }

      #user-input {
        flex: 1;
        resize: none;
        min-height: 42px;
        max-height: 80px;
        border: none;
        outline: none;
        background: transparent;
        color: var(--text);
        font-size: 13px;
        line-height: 1.4;
        font-family: inherit;
      }

      #send-btn {
        border-radius: 999px;
        border: none;
        padding: 8px 14px;
        background: linear-gradient(135deg, var(--accent), #73ffe6);
        color: #020111;
        font-size: 12px;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        box-shadow: 0 10px 26px rgba(16, 199, 213, 0.65);
      }

      #send-btn[disabled] {
        opacity: 0.5;
        cursor: default;
        box-shadow: none;
      }

      .send-icon {
        transform: translateY(0.5px);
      }

      /* Todos */
      #todos-list {
        display: flex;
        flex-direction: column;
        gap: 6px;
      }

      .todo-card {
        border-radius: 12px;
        padding: 7px 8px;
        background: radial-gradient(circle at top left, #202652, #070924);
        border: 1px solid rgba(55, 65, 130, 0.95);
        font-size: 12px;
        box-shadow: 0 10px 24px rgba(0, 0, 0, 0.65);
      }

      .todo-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 4px;
      }

      .todo-title {
        font-weight: 500;
        font-size: 12px;
      }

      .todo-id {
        font-size: 10px;
        color: var(--text-muted);
      }

      .todo-status-badge {
        border-radius: 999px;
        font-size: 10px;
        padding: 2px 8px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }

      .todo-status-pending {
        background: rgba(68, 90, 255, 0.16);
        color: rgba(173, 187, 255, 0.96);
        border: 1px solid rgba(102, 132, 255, 0.8);
      }

      .todo-status-in_progress {
        background: rgba(70, 194, 255, 0.2);
        color: #e8f7ff;
        border: 1px solid rgba(70, 194, 255, 0.9);
      }

      .todo-status-done {
        background: rgba(84, 233, 181, 0.15);
        color: #d6ffe9;
        border: 1px solid rgba(84, 233, 181, 0.9);
      }

      .todo-status-removed {
        background: rgba(91, 99, 138, 0.25);
        color: rgba(174, 179, 214, 0.9);
        border: 1px dashed rgba(124, 132, 175, 0.9);
      }

      .todo-body {
        margin-top: 4px;
        font-size: 11px;
        color: var(--text-muted);
      }

      .todo-footer {
        margin-top: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 10px;
        color: var(--text-muted);
      }

      /* Tools & summary */
      .tools-section {
        margin-bottom: 8px;
      }

      .tools-section h3 {
        margin: 0 0 5px;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-muted);
      }

      #tool-log {
        display: flex;
        flex-direction: column;
        gap: 4px;
      }

      .tool-entry {
        border-radius: 10px;
        padding: 6px 8px;
        background: radial-gradient(circle at top left, #202755, #050719);
        border: 1px solid rgba(53, 66, 136, 0.95);
        font-size: 11px;
      }

      .tool-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 2px;
      }

      .tool-name {
        font-weight: 500;
      }

      .tool-meta {
        font-size: 10px;
        color: var(--text-muted);
      }

      .tool-type-chip {
        border-radius: 999px;
        padding: 1px 7px;
        font-size: 9px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        background: rgba(12, 18, 55, 0.98);
        border: 1px solid rgba(68, 80, 158, 0.95);
        color: var(--text-muted);
      }

      .tool-payload {
        margin-top: 3px;
        padding: 5px 6px;
        border-radius: 7px;
        background: rgba(2, 5, 19, 0.95);
        border: 1px solid rgba(38, 49, 112, 0.95);
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono",
          "Courier New", monospace;
        font-size: 11px;
        max-height: 140px;
        overflow: auto;
        white-space: pre;
      }

      #summary-markdown {
        font-size: 12px;
        color: var(--text);
      }

      #summary-markdown .markdown {
        font-size: 12px;
      }

      /* Status */
      .status-bar {
        margin-top: 4px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-size: 11px;
        color: var(--text-muted);
      }

      .status-main {
        display: inline-flex;
        align-items: center;
        gap: 6px;
      }

      .thinking-dot {
        width: 8px;
        height: 8px;
        border-radius: 999px;
        background: var(--accent);
        box-shadow: 0 0 0 6px rgba(70, 194, 255, 0.3);
        animation: pulse 1.3s ease-out infinite;
      }

      .status-error {
        color: var(--danger);
      }

      /* Animations */
      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(4px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }

      @keyframes pulse {
        0% {
          transform: scale(1);
          box-shadow: 0 0 0 0 rgba(70, 194, 255, 0.4);
        }
        70% {
          transform: scale(1.06);
          box-shadow: 0 0 0 9px rgba(70, 194, 255, 0);
        }
        100% {
          transform: scale(1);
          box-shadow: 0 0 0 0 rgba(70, 194, 255, 0);
        }
      }

      /* Responsive */
      @media (max-width: 1040px) {
        .main-layout {
          grid-template-columns: 1.3fr 1.1fr;
          grid-template-rows: auto auto;
        }
        .column-tools {
          grid-column: 1 / -1;
          min-height: 260px;
        }
      }

      @media (max-width: 780px) {
        #app {
          padding: 10px;
        }
        .main-layout {
          grid-template-columns: 1fr;
        }
        .column {
          min-height: 220px;
        }
        header.top-nav {
          padding: 10px 12px;
        }
        .brand-text-subtitle {
          display: none;
        }
      }
    </style>
  </head>
  <body>
    <div id="app">
      <header class="top-nav">
        <div class="brand">
          <div class="brand-mark"></div>
          <div>
            <div class="brand-text-title">Agentic Mongo Assistant</div>
            <div class="brand-text-subtitle">Planner · Tools · Summarizer · Streaming</div>
          </div>
        </div>
        <div class="nav-controls">
          <div class="pill">
            <span class="pill-dot"></span>
            <span id="connection-status-text">Connected to OSS model</span>
          </div>
          <button class="btn-secondary" id="clear-btn" type="button">
            Clear view
          </button>
        </div>
      </header>

      <main class="main-layout">
        <!-- Conversation column -->
        <section class="column column-conversation">
          <div class="column-header">
            <div class="column-title">Conversation</div>
            <span class="badge" id="session-pill">Session: &mdash;</span>
          </div>
          <div class="column-body">
            <div class="scroll-panel">
              <div id="conversation-list"></div>
            </div>
          </div>
          <div class="status-bar">
            <div class="status-main">
              <span class="thinking-dot" id="stream-indicator" style="visibility: hidden"></span>
              <span id="status-text">Idle</span>
            </div>
            <div id="error-text" class="status-error"></div>
          </div>
          <div class="input-shell">
            <textarea
              id="user-input"
              placeholder="Ask the agent to inspect or remediate your Mongo data..."
            ></textarea>
            <button id="send-btn" type="button">
              <span class="send-icon">➤</span>
              <span>Send</span>
            </button>
          </div>
        </section>

        <!-- Todos column -->
        <section class="column column-todos">
          <div class="column-header">
            <div class="column-title">Plan & Todos</div>
            <span class="badge" id="cycle-pill">Cycle: 0</span>
          </div>
          <div class="column-body">
            <div class="scroll-panel">
              <div id="todos-list"></div>
            </div>
          </div>
        </section>

        <!-- Tools and summary column -->
        <section class="column column-tools">
          <div class="column-header">
            <div class="column-title">Tools & Outcome</div>
          </div>
          <div class="column-body">
            <div class="scroll-panel">
              <div class="tools-section">
                <h3>Tool calls & results</h3>
                <div id="tool-log"></div>
              </div>
              <div class="tools-section">
                <h3>Summary</h3>
                <div id="summary-markdown"></div>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>

    <!-- Markdown + sanitization -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
    <script>
      // --- Simple state ---
      const state = {
        sessionId: null,
        messages: [],
        todos: new Map(), // id -> todo object
        toolEvents: [],
        summary: null,
        reasoningBuffer: "",
        streaming: false,
        hasError: false,
        controller: null,
        cycle: 0,
      };

      // --- Helpers ---
      function uuid() {
        if (crypto.randomUUID) return crypto.randomUUID();
        return "xxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, function (c) {
          const r = (Math.random() * 16) | 0;
          const v = c === "x" ? r : (r & 0x3) | 0x8;
          return v.toString(16);
        });
      }

      function setStatus(text) {
        const el = document.getElementById("status-text");
        if (el) el.textContent = text;
      }

      function setError(text) {
        const el = document.getElementById("error-text");
        if (el) el.textContent = text || "";
        state.hasError = !!text;
      }

      function setStreaming(isStreaming) {
        state.streaming = isStreaming;
        const dot = document.getElementById("stream-indicator");
        const sendBtn = document.getElementById("send-btn");
        if (dot) dot.style.visibility = isStreaming ? "visible" : "hidden";
        if (sendBtn) sendBtn.disabled = isStreaming;
        setStatus(isStreaming ? "Agent is working..." : "Idle");
      }

      function safeMarkdownToHtml(markdownText) {
        if (!markdownText) return "";
        const html = marked.parse(markdownText, { breaks: true, gfm: true });
        return DOMPurify.sanitize(html);
      }

      function appendMessage(role, content, opts = {}) {
        state.messages.push({
          role,
          content,
          at: new Date(),
          ...opts,
        });
        renderConversation();
      }

      // --- Rendering: Conversation ---
      function renderConversation() {
        const list = document.getElementById("conversation-list");
        if (!list) return;
        list.innerHTML = "";

        for (const msg of state.messages) {
          const div = document.createElement("div");
          let cls = "msg msg-assistant";
          if (msg.role === "user") cls = "msg msg-user";
          else if (msg.role === "reasoning") cls = "msg msg-reasoning";
          div.className = cls;

          const body = document.createElement("div");
          if (msg.isMarkdown) {
            body.className = "markdown";
            body.innerHTML = safeMarkdownToHtml(msg.content);
          } else {
            body.textContent = msg.content;
          }
          div.appendChild(body);

          if (msg.meta) {
            const metaEl = document.createElement("div");
            metaEl.className = "msg-meta";
            metaEl.textContent = msg.meta;
            div.appendChild(metaEl);
          }

          list.appendChild(div);
        }

        // If we have a live reasoning buffer, show it as the last reasoning block.
        if (state.reasoningBuffer) {
          const div = document.createElement("div");
          div.className = "msg msg-reasoning";

          const body = document.createElement("div");
          body.className = "markdown";
          body.innerHTML = safeMarkdownToHtml(state.reasoningBuffer);
          div.appendChild(body);

          const metaEl = document.createElement("div");
          metaEl.className = "msg-meta";
          metaEl.textContent = "Streaming model reasoning...";
          div.appendChild(metaEl);

          list.appendChild(div);
        }

        const scrollPanel = list.parentElement;
        if (scrollPanel) {
          scrollPanel.scrollTop = scrollPanel.scrollHeight;
        }
      }

      // --- Rendering: Todos ---
      function renderTodos() {
        const list = document.getElementById("todos-list");
        const cyclePill = document.getElementById("cycle-pill");
        if (cyclePill) {
          cyclePill.textContent = `Cycle: ${state.cycle}`;
        }
        if (!list) return;
        list.innerHTML = "";

        const todosArr = Array.from(state.todos.values());
        if (todosArr.length === 0) {
          const empty = document.createElement("div");
          empty.style.fontSize = "11px";
          empty.style.color = "var(--text-muted)";
          empty.textContent = "Planner todos will appear here once generated.";
          list.appendChild(empty);
          return;
        }

        for (const todo of todosArr) {
          const card = document.createElement("div");
          card.className = "todo-card";
          if (todo._removed) {
            card.style.opacity = "0.7";
          }

          const header = document.createElement("div");
          header.className = "todo-header";
          const title = document.createElement("div");
          title.className = "todo-title";
          title.textContent = todo.title || "(untitled todo)";
          header.appendChild(title);

          const statusBadge = document.createElement("div");
          const statusKey =
            todo.status === "in_progress" || todo.status === "in progress"
              ? "in_progress"
              : todo.status;
          statusBadge.className = `todo-status-badge todo-status-${statusKey}`;
          statusBadge.textContent =
            statusKey === "in_progress"
              ? "IN PROGRESS"
              : statusKey === "done"
              ? "DONE"
              : statusKey === "removed"
              ? "REMOVED"
              : "PENDING";
          header.appendChild(statusBadge);

          card.appendChild(header);

          const idEl = document.createElement("div");
          idEl.className = "todo-id";
          idEl.textContent = `ID: ${todo.id}`;
          card.appendChild(idEl);

          if (todo.description) {
            const body = document.createElement("div");
            body.className = "todo-body";
            body.textContent = todo.description;
            card.appendChild(body);
          }

          const footer = document.createElement("div");
          footer.className = "todo-footer";
          const left = document.createElement("span");
          left.textContent = todo.cycle != null ? `Cycle ${todo.cycle}` : "Cycle –";
          const right = document.createElement("span");
          right.textContent = todo.updatedAt
            ? new Date(todo.updatedAt).toLocaleTimeString()
            : "";
          footer.appendChild(left);
          footer.appendChild(right);
          card.appendChild(footer);

          list.appendChild(card);
        }
      }

      // --- Rendering: Tools ---
      function renderTools() {
        const log = document.getElementById("tool-log");
        if (!log) return;
        log.innerHTML = "";
        if (state.toolEvents.length === 0) {
          const empty = document.createElement("div");
          empty.style.fontSize = "11px";
          empty.style.color = "var(--text-muted)";
          empty.textContent = "Tool calls and results will appear here.";
          log.appendChild(empty);
          return;
        }

        for (const ev of state.toolEvents) {
          const entry = document.createElement("div");
          entry.className = "tool-entry";

          const header = document.createElement("div");
          header.className = "tool-header";

          const left = document.createElement("div");
          const nameEl = document.createElement("span");
          nameEl.className = "tool-name";
          nameEl.textContent = ev.name;
          left.appendChild(nameEl);
          if (ev.cycle != null || ev.iteration != null) {
            const meta = document.createElement("span");
            meta.className = "tool-meta";
            const parts = [];
            if (ev.cycle != null) parts.push(`cycle ${ev.cycle}`);
            if (ev.iteration != null) parts.push(`step ${ev.iteration}`);
            meta.textContent = " · " + parts.join(" · ");
            left.appendChild(meta);
          }

          const chip = document.createElement("span");
          chip.className = "tool-type-chip";
          chip.textContent = ev.type === "call" ? "CALL" : "RESULT";

          header.appendChild(left);
          header.appendChild(chip);
          entry.appendChild(header);

          const payload = document.createElement("div");
          payload.className = "tool-payload";
          try {
            payload.textContent = JSON.stringify(ev.payload, null, 2);
          } catch {
            payload.textContent = String(ev.payload);
          }
          entry.appendChild(payload);

          log.appendChild(entry);
        }
      }

      // --- Rendering: Summary ---
      function renderSummary() {
        const container = document.getElementById("summary-markdown");
        if (!container) return;
        container.innerHTML = "";
        if (!state.summary) {
          const empty = document.createElement("div");
          empty.style.fontSize = "11px";
          empty.style.color = "var(--text-muted)";
          empty.textContent = "Once the agent finishes, a summary will appear here.";
          container.appendChild(empty);
          return;
        }
        const md = [];
        if (state.summary.summary) md.push(state.summary.summary);
        if (state.summary.details) {
          md.push("");
          md.push(state.summary.details);
        }
        const div = document.createElement("div");
        div.className = "markdown";
        div.innerHTML = safeMarkdownToHtml(md.join("\n\n"));
        container.appendChild(div);
      }

      // --- SSE handling over fetch ---
      async function streamChat(message) {
        if (!state.sessionId) {
          state.sessionId = uuid();
          const pill = document.getElementById("session-pill");
          if (pill) pill.textContent = `Session: ${state.sessionId.slice(0, 8)}`;
        }

        setError("");
        setStreaming(true);

        // reset per-request UI state, but preserve messages for context
        state.reasoningBuffer = "";
        state.toolEvents = [];
        renderTools();
        state.summary = null;
        renderSummary();

        const url = "/agent/chat/stream";
        const controller = new AbortController();
        state.controller = controller;

        const payload = {
          session_id: state.sessionId,
          message,
          metadata: {},
        };

        try {
          const res = await fetch(url, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify(payload),
            signal: controller.signal,
          });

          if (!res.ok || !res.body) {
            setError(`HTTP ${res.status} from server`);
            setStreaming(false);
            return;
          }

          const reader = res.body.getReader();
          const decoder = new TextDecoder("utf-8");
          let buffer = "";

          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, { stream: true });

            let boundary = buffer.indexOf("\n\n");
            while (boundary !== -1) {
              const chunk = buffer.slice(0, boundary);
              buffer = buffer.slice(boundary + 2);
              handleSseChunk(chunk);
              boundary = buffer.indexOf("\n\n");
            }
          }
        } catch (err) {
          if (err.name !== "AbortError") {
            console.error("Stream error", err);
            setError("Stream error. See console for details.");
          }
        } finally {
          setStreaming(false);
          state.controller = null;
        }
      }

      function handleSseChunk(raw) {
        if (!raw.trim()) return;
        const lines = raw.split(/\r?\n/);
        let eventType = "message";
        let dataLines = [];
        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.slice(6).trim();
          } else if (line.startsWith("data:")) {
            dataLines.push(line.slice(5).trim());
          }
        }
        const dataStr = dataLines.join("\n");
        let data;
        try {
          data = dataStr ? JSON.parse(dataStr) : {};
        } catch (e) {
          console.warn("Failed to parse SSE data JSON", e, dataStr);
          return;
        }
        handleEvent(eventType, data);
      }

      function handleEvent(type, data) {
        switch (type) {
          case "start":
            setStatus("Agent is working...");
            break;
          case "model_stream":
            if (typeof data.delta === "string") {
              state.reasoningBuffer += data.delta;
              renderConversation();
            }
            break;
          case "planner_reasoning":
            state.cycle = data.cycle ?? 0;
            // finalize buffer into a message, then clear it
            if (state.reasoningBuffer.trim()) {
              appendMessage("reasoning", state.reasoningBuffer, {
                isMarkdown: true,
                meta: `Model reasoning (live stream)`,
              });
              state.reasoningBuffer = "";
            }
            if (data.reasoning) {
              appendMessage("assistant", data.reasoning, {
                isMarkdown: true,
                meta: `Planner reasoning · cycle ${state.cycle}`,
              });
            }
            break;
          case "planner_todos":
            state.cycle = data.cycle ?? state.cycle ?? 0;
            state.todos.clear();
            if (Array.isArray(data.todos)) {
              for (const t of data.todos) {
                if (!t || !t.id) continue;
                state.todos.set(t.id, {
                  ...t,
                  cycle: state.cycle,
                  updatedAt: Date.now(),
                });
              }
            }
            renderTodos();
            break;
          case "agent_step":
            if (Array.isArray(data.todos)) {
              state.todos.clear();
              for (const t of data.todos) {
                if (!t || !t.id) continue;
                state.todos.set(t.id, {
                  ...t,
                  updatedAt: Date.now(),
                });
              }
              renderTodos();
            }
            if (data.step_explanation) {
              appendMessage("assistant", data.step_explanation, {
                isMarkdown: true,
                meta: "Execution step",
              });
            }
            break;
          case "tool_call":
            state.toolEvents.push({
              type: "call",
              name: data.name,
              payload: data.input,
              at: Date.now(),
              cycle: state.cycle,
              iteration: data.iteration,
            });
            renderTools();
            break;
          case "tool_result":
            state.toolEvents.push({
              type: "result",
              name: data.name,
              payload: data.result,
              at: Date.now(),
              cycle: state.cycle,
              iteration: data.iteration,
            });
            renderTools();
            break;
          case "summary":
            state.summary = data;
            renderSummary();
            if (data && data.summary) {
              appendMessage("assistant", data.summary, {
                isMarkdown: true,
                meta: data.needs_more_planning
                  ? "Interim summary (more planning required)"
                  : "Final summary",
              });
            }
            break;
          case "agent_abort":
            setError("Agent aborted: " + (data.reason || "max iterations reached"));
            break;
          case "error":
            setError(
              `Agent error (${data.status_code ?? "?"}): ${
                typeof data.detail === "string" ? data.detail : "see logs"
              }`
            );
            break;
          case "end":
            setStatus("Finished");
            break;
          default:
            break;
        }
      }

      // --- Event wiring ---
      function init() {
        const input = document.getElementById("user-input");
        const sendBtn = document.getElementById("send-btn");
        const clearBtn = document.getElementById("clear-btn");

        if (sendBtn && input) {
          sendBtn.addEventListener("click", () => {
            const text = input.value.trim();
            if (!text || state.streaming) return;
            appendMessage("user", text, { isMarkdown: false, meta: "You" });
            input.value = "";
            streamChat(text);
          });

          input.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              sendBtn.click();
            }
          });
        }

        if (clearBtn) {
          clearBtn.addEventListener("click", () => {
            if (state.controller) {
              state.controller.abort();
            }
            // Clear UI state but keep session id so context stays
            state.messages = [];
            state.todos.clear();
            state.toolEvents = [];
            state.summary = null;
            state.reasoningBuffer = "";
            state.hasError = false;
            setError("");
            setStreaming(false);
            renderConversation();
            renderTodos();
            renderTools();
            renderSummary();
          });
        }

        renderConversation();
        renderTodos();
        renderTools();
        renderSummary();
      }

      window.addEventListener("DOMContentLoaded", init);
    </script>
  </body>
</html>

