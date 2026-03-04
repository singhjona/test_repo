python```

import requests

url = "https://ai-icp-ccaas.channels.euw1.dev.aws.cloud.hsbc/llm/qwen/v1/chat/completions"

payload = {
    "messages": [
        {
            "role": "system",
            "content": ""
        },
        {
            "role": "user",
            "content": ""
        }
    ],
    "model": "/mnt/llm/llm_hosting/gpt-oss-20b/model_files",
    "reasoning_effort": "medium",
    "stream": true,
    "response_format": {
        "type": "json_schema",
        "json_schema": {
            "name": "structured_example",
            "description": "Example showing structured output (no prompt content)",
            "schema": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "integer"},
                                "value": {"type": "string"}
                            },
                            "required": ["id", "value"]
                        }
                    }
                },
                "required": ["result"]
            },
            "strict": True
        }
    }
}

headers = {
  "Content-Type": "application/json",
  "User-Agent": "insomnia/11.0.0"
}

response = requests.request("POST", url, json=payload, headers=headers,verify=False)

print(response.text)

```


Summary: server/tools.py

Location: [`server/tools.py`](server/tools.py)

Purpose
- Implements the FastMCP toolset used by the Mongo MCP server. Tools are registered on the local `mcp` ([`server/tools.py`](server/tools.py)) instance and intended to be invoked by the agent or external MCP clients.
- Emphasizes a guarded 3-step anomaly remediation workflow: profile -> check -> fix. See plans: [`plans/rename_mongo_tools_plan.md`](plans/rename_mongo_tools_plan.md) and related docs.

Key exported tools (functions are defined in this module; links resolve to this file)
- Read tooling
  - [`server.tools.run_mongo_db_read_code_tool`](server/tools.py) - public read-only runner for single or bulk Mongo specs. Delegates validation to [`server.mongo.validate_mongo_spec`](server/mongo.py) and execution to [`server.mongo.run_mongo_db_code`](server/mongo.py). Backwards-compatible alias: [`server.tools.run_mongo_db_code_tool`](server/tools.py).
- Write tooling (gated)
  - [`server.tools.run_mongo_db_write_code_tool`](server/tools.py) - based write runner; only registered when `ENABLE_WRITE_TOOLS` is true. Delegates to [`server.mongo.run_mongo_db_write_code`](server/mongo.py).
  - Higher-level write helpers: [`server.tools.insert_documents_tool`](server/tools.py), [`server.tools.create_collection_tool`](server/tools.py), and the remediation executor [`server.tools.fix_anomalies`](server/tools.py). These enforce conservative validation and caps via config constants.
- Discovery & inspection
  - [`server.tools.get_schema`](server/tools.py) - lightweight schema inference using bounded samples.
  - [`server.tools.describe_collection`](server/tools.py) - count + schema summary for one or multiple collections.
  - [`server.tools.describe_database`](server/tools.py) - returns collection stats (logged via [`server.tool_logging`](server/tool_logging.py)).
  - [`server.tools.get_documents`](server/tools.py) - bounded document sampling with selection modes.

Anomaly workflow (controls & intent)
- STEP 1: [`server.tools.profile_collection_anomalies`](server/tools.py)
  - Produces deterministic, bounded profiles: presence, nulls, type counts, numeric/string stats, top values, normalization/collision flags.
  - Emits LLM instructions + suggested checks; intended as the single authoritative profile before checks/writes.
- STEP 2: [`server.tools.anomalies_tools`](server/tools.py)
  - Runs a small list (1-5) of targeted checks derived from the profile. Supported check actions: `dependency_check`, `statistical_outlier`, `entropy_check`, `fuzzy_match_anomaly`, `velocity_check`.
- STEP 3: [`server.tools.fix_anomalies`](server/tools.py)
  - Human-in-the-loop immediate application of deterministic fixes. Validates preconditions, enforces global caps (`MAX_MODIFIED_DEFAULT` from [`server.config`](server/config.py)), gates via `ENABLE_WRITE_TOOLS`, records audit to `ANOMALY_APPROVALS_COLLECTION` (config constant).

Safeguards & integration points
- Validation: read tool defers to [`server.mongo.validate_mongo_spec`](server/mongo.py); fixes and write helpers call [`server.mongo.validate_mongo_write_spec`](server/mongo.py).
- Execution: runtime operations go through [`server.mongo.run_mongo_db_code`](server/mongo.py) and [`server.mongo.run_mongo_db_write_code`](server/mongo.py).
- Logging & telemetry: per-call logging via [`server.tool_logging.log_tool_invocation`](server/tool_logging.py) and timing via `tool_timing`.
- Config-driven caps and gating: constants like `MAX_RESULTS`, `MAX_BULK_ITEMS`, `ENABLE_WRITE_TOOLS`, `MAX_MODIFIED_DEFAULT`, `ANOMALY_APPROVALS_COLLECTION` in [`server.config`](server/config.py) control behavior and safety.

Implementation notes / TODOs (as seen in the code)
- Many helper paths validate/normalize inputs and return consistent {"success": bool, ...} structures.
- Bulk parallel execution uses thread pool (bounded).
- `fix_anomalies` persists an execution/audit doc and performs post-verification counts; all writes are audited.
- `create_collection_tool` emulates collection creation conservatively and records audit.
- `insert_documents_tool` sanitizes via `mongo._sanitize_document_for_insert` if present and records an insert audit to `INSERT_AUDIT_COLLECTION`.

Related files and symbols
- [`server.mongo`](server/mongo.py) - validation and DB runner used by these tools.
- [`server.config`](server/config.py) - configuration constants controlling caps/gates.
- [`server.tool_logging`](server/tool_logging.py) - logging helpers used on each tool invocation.

Recommended reading
- See the API reference in [`docs/mcp-server-tools.md`](docs/mcp-server-tools.md) for human-oriented descriptions and safety model.
- See plans: [`plans/rename_mongo_tools_plan.md`](plans/rename_mongo_tools_plan.md) for intended renames and separation of responsibilities between read/write tools.


Summary: server/app.py

Location: [`server/app.py`](server/app.py)

Purpose
- Assembles and exposes the FastAPI application used for agent endpoints and (when available) mounts the FastMCP ASGI app under `/mcp`.
- Provides helper entrypoints for running the FastMCP server directly (`run_server`) and for creating the MCP instance (`create_mcp`).

Primary symbols
- [`server.app.create_mcp`](server/app.py)
  - Returns the FastMCP instance created by importing [`server.tools.mcp`](server/tools.py). Intended for CLI/runner callers to obtain the registered MCP toolset.
- [`server.app.run_server`](server/app.py)
  - Configures logging (via [`server.config.configure_logging`](server/config.py) and [`server.tool_logging.configure_tool_logging`](server/tool_logging.py)), then calls `mcp.run(...)` to start the FastMCP server.
- `app: FastAPI` - the FastAPI application instance exported for uvicorn.

Startup / mounting behavior
- `@app.on_event("startup")` handler:
  - Calls logging configurators to ensure logging is active under uvicorn.
  - Calls [`create_mcp`](server/app.py) to get the MCP instance, then attempts to discover an ASGI app on the MCP by probing candidate attributes:
    - `["asgi_app", "app", "asgi", "get_asgi_app", "create_asgi_app", "streamable_http_app", "sse_app", "http_app"]`
  - If an ASGI app is found, mounts it at `/mcp` and logs the chosen attribute.
  - Logs the FastAPI `app.routes` for diagnostics.
  - Imports and includes the agent router: imports [`server.agent.router`](server/agent.py) and calls `app.include_router(...)`. Errors are logged with tracebacks to surface import-time issues (addresses plan guidance to avoid silent failures).
- `@app.on_event("shutdown")` handler:
  - Attempts to cancel and await in-flight agent background tasks by inspecting [`server.agent.ACTIVE_AGENT_TASKS`](server/agent.py).
  - Logs progress and any exceptions during shutdown cleanup.

Fallback route
- `/mcp` HTTP route: if the FastMCP ASGI app could not be mounted, the endpoint returns a 501 with guidance to start the server with `run_server()` or mount FastMCP ASGI manually.

Integration and safety notes
- Logging configuration is applied both in `run_server()` and in the FastAPI startup handler so that running via `uvicorn server.app:app` still activates tool logging and route diagnostics.
- The startup handler intentionally delays importing the agent router until after logging is configured so import errors are visible in logs rather than silently masked.
- The mounting logic is defensive: probes multiple attribute names for FastMCP compatibility and catches exceptions while attempting to call candidate factories.

Related files and symbols
- [`server.tools`](server/tools.py) - module that registers tools and provides the `mcp` instance returned by [`create_mcp`](server/app.py).
- [`server.agent`](server/agent.py) - agent router and background task lifecycle used at startup/shutdown.
- [`server.config.configure_logging`](server/config.py) and [`server.tool_logging.configure_tool_logging`](server/tool_logging.py) - logging setup invoked by `run_server` and startup handler.

Operational guidance
- For full MCP endpoints under `/mcp`, prefer starting with `run_server()` (which calls `mcp.run(...)`). When running under uvicorn, ensure logging config and FastMCP version compatibility for one of the probed attributes so the ASGI app mounts correctly.
- If WebSocket/SSE agent endpoints are missing or 404ing, consult the startup logs (route listing) and the agent router import exception (both are logged by the startup handler).

See also
- [`server/tools.py`](server/tools.py) - tools registered onto the `mcp` instance.
- [`server/agent.py`](server/agent.py) - agent endpoints and background task lifecycle.

**Server Structure**

- **Path**: `server/` - Contains the FastMCP server implementation and related modules.
- **Key files**:
  - `__init__.py`: package initializer.
  - `__main__.py`: module entrypoint/CLI.
  - `agent.py`: FastAPI router for agent endpoints and background task lifecycle management.
  - `app.py`: FastAPI app assembly, FastMCP mounting, startup/shutdown logic.
  - `config.py`: application configuration constants and logging setup.
  - `middleware.py`: FastAPI middleware and request hooks.
  - `mongo.py`: MongoDB validation, query execution and write helpers used by tools.
  - `tool_logging.py`: per-tool logging, invocation records and timing helpers.
  - `tools.py`: MCP tool registrations and the anomaly profile/check/fix workflow (primary focus of this summary).
  - `static/`: static assets (e.g. `chat.html`) served by the app when applicable.
- **Notes**: `tools.py` and `app.py` are central to the MCP runtime; `mongo.py` provides the DB-side validation/execution layer and `agent.py` manages HTTP/WebSocket endpoints and background tasks.


Code Snippet: server/tools.py

```python
@mcp.tool(
    description=(
        "Execute a SINGLE or BULK MongoDB READ operation(s) described by a JSON 'spec' (or list of specs). "
        "This tool is strictly read-only and delegates validation to the Mongo layer via mongo.validate_mongo_spec(). "
        "Supported read operations: find, aggregate, count, distinct, list_collections, collection_stats. "
        "Example: {'operation':'find','collection':'people','filter':{},'limit':10} will return up to 10 documents from the 'people' collection. "
        "Use the dedicated write tool 'run_mongo_db_write_code_tool' for any writes (gated by ENABLE_WRITE_TOOLS)."
    )
)
def run_mongo_db_read_code_tool(
    spec: Annotated[
        Dict[str, Any] | List[Dict[str, Any]],
        Field(description="Mongo read spec object OR list of spec objects. See docs for allowed read operations."),
    ],
    execution_mode: Annotated[Literal["parallel", "sequential"], Field(description="How to execute bulk items. Default 'parallel'.")] = "parallel",
) -> Dict[str, Any]:
    """Run Mongo read-only specs through the sandbox runner (single or bulk).

    Validation: each spec is validated with mongo.validate_mongo_spec() which is the single source of truth
    for allowed read operations. This function performs NO write-detection or write-gating itself.
    """
    tool_name = "run_mongo_db_read_code_tool"


@mcp.tool(
    description=(
        "Execute a SINGLE or BULK MongoDB READ operation(s) described by a JSON 'spec' (or list of specs). "
        "This tool is strictly read-only and delegates validation to the Mongo layer via mongo.validate_mongo_spec(). "
        "Supported read operations: find, aggregate, count, distinct, list_collections, collection_stats. "
        "Example: {'operation':'find','collection':'people','filter':{},'limit':10} will return up to 10 documents from the 'people' collection. "
        "Use the dedicated write tool 'run_mongo_db_write_code_tool' for any writes (gated by ENABLE_WRITE_TOOLS)."
    )
)
def run_mongo_db_read_code_tool(
    spec: Annotated[
        Dict[str, Any] | List[Dict[str, Any]],
        Field(description="Mongo read spec object OR list of spec objects. See docs for allowed read operations."),
    ],
    execution_mode: Annotated[
        Literal["parallel", "sequential"], 
        Field(description="How to execute bulk items. Default 'parallel'.")
    ] = "parallel",
) -> Dict[str, Any]:
    """Run Mongo read-only specs through the sandbox runner (single or bulk).

    Validation: each spec is validated with mongo.validate_mongo_spec() which is the single source of truth
    for allowed read operations. This function performs NO write-detection or write-gating itself.
    """
    tool_name = "run_mongo_db_read_code_tool"


@mcp.tool(
    description=(
        "Return a brief collection summary (count + lightweight schema) for one collection OR multiple collections. "
        "Example: describe_collection('ccaas.orders', sample_limit=200) or describe_collection(['ccaas.users','ccaas.orders'])."
    )
)
def describe_collection(
    collection: Annotated[
        str | List[str],
        Field(
            description=(
                "Collection name OR list of collection names. Accepts 'collection' or 'db.collection'.\n"
                "Examples: 'orders', 'ccaas.orders', ['ccaas.users','ccaas.orders']"
            )
        )
    ],
    sample_limit: Annotated[
        int,
        Field(
            description=(
                "Number of documents to sample for the schema summary per collection (bounded by MAX_RESULTS).\n"
                "Examples: 50, 100, 200"
            ),
            ge=1,
            le=MAX_RESULTS,
        ),
    ] = 100,
    execution_mode: Annotated[
        Literal["parallel", "sequential"],
        Field(description="How to execute bulk items. Default 'parallel'."),
    ] = "parallel",
) -> Dict[str, Any]:
    tool_name = "describe_collection"

def _describe_one(coll_in: str) -> Dict[str, Any]:


@mcp.tool(
    description=(
        "List collections in the default database ('ccaas') and return basic stats. "
        "Use this to discover available collections before querying a specific one."
    )
)
@logged_tool()
def describe_database() -> Dict[str, Any]:
    spec = {"operation": "collection_stats", "database": "ccaas"}
    out = mongo.run_mongo_db_code(spec)
    return out


@mcp.tool(
    description=(
        "Fetch a SMALL, bounded set of documents for inspection from one collection OR multiple collections. "
        "Selection modes: 'first', 'last', 'random', 'filter', 'skip', 'sample'. "
        "Example: get_documents('users', selection_mode='first', number_of_documents=5) "
        "or get_documents(['ccaas.users', 'ccaas.orders'], selection_mode='sample', number_of_documents=10)."
    )
)
def get_documents(
    collection: Annotated[
        str | List[str],
        Field(
            description=(
                "Collection name OR list of collection names. Accepts 'collection' or 'db.collection'.\n"
                "Examples: 'users', 'ccaas.users', ['ccaas.users','ccaas.orders']"
            )
        )
    ],
    selection_mode: Annotated[
        Literal["first", "last", "random", "filter", "skip", "sample"],
        Field(description="How to select documents for each collection."),
    ] = "first",
    selection_filter: Annotated[
        Dict[str, Any] | None,
        Field(description="MongoDB filter document or None."),
    ] = None,
    projection: Annotated[
        Dict[str, Any] | None,
        Field(description="Projection spec or None (only applies to 'find' based modes)."),
    ] = None,
    sort: Annotated[
        Any,
        Field(description="Sort spec for 'find' based modes. Note: 'last' forces sort by {'_id': -1}."),
    ] = None,
    number_of_documents: Annotated[
        int,
        Field(description="Number of documents to return (bounded by MAX_RESULTS).", ge=1, le=MAX_RESULTS),
    ] = 10,
    skip: Annotated[
        int,
        Field(description="Number of documents to skip before returning results (used in 'skip' mode).", ge=0),
    ] = 0,
    seed: Annotated[
        int | None,
        Field(description="Optional RNG seed for sampling (when relevant). Example: 42"),
    ] = None,
    execution_mode: Annotated[
        Literal["parallel", "sequential"],
        Field(description="How to execute bulk items. Default 'parallel'."),
    ] = "parallel",
) -> Dict[str, Any]:
    tool_name = "get_documents"

@mcp.tool(
    description=(
        "STEP 1 of 2 (MUST RUN FIRST): Create a bounded anomaly profile for a collection. "
        "This returns deterministic, structured metrics (field presence/missingness, null rates, type counts, "
        "simple numeric/string stats, and top values capped at 100). "
        "Use the returned profile to decide which SMALL number of targeted checks to run next using anomalies_tools().\n\n"
        "How to use (recommended plan):\n"
        "1) Call profile_collection_anomalies(collection)\n"
        "2) Review fields with flags like high_missing_ratio/high_null_ratio/type_inconsistent/outlier_suspected.\n"
        "3) Choose 1-5 narrowly scoped follow-up checks.\n"
        "4) Call anomalies_tools(collection, checks=[...]) with only those checks.\n\n"
        "Example: profile_collection_anomalies('ccaas.users').\n\n"
        "MANDATE: This tool MUST be run before any anomalies_tools() or fix_anomalies() calls for a given remediation flow. "
        "Failure to follow the required order invalidates the workflow and must be treated as a process failure."
    )
)
@logged_tool()
def profile_collection_anomalies(
    collection: Annotated[
        str,
        Field(
            description=(
                "Collection name to profile. Accepts 'collection' or 'db.collection'.\n"
                "Examples: 'users', 'ccaas.users'"
            )
        )
    ],
) -> Dict[str, Any]:
    """Mongo-first sampler/profile. Operates and returns bounded JSON.

    This implementation focuses on top-level fields on a bounded sample and computes
    presence, nulls, type counts, simple numeric/string stats and top values (<=100).
    """
    db_name, coll_name = mongo.normalize_collection_name(collection, default_db="ccaas")
    db_name = db_name or "ccaas"


@mcp.tool(
    description=(
        "STEP 2 of 2 (MUST RUN AFTER profiling): Execute SMALL, targeted anomaly checks.\n\n"
        "ABSOLUTE RULES:\n"
        "- NEVER run anomalies_tools() before profile_collection_anomalies().\n"
        "- ALWAYS derive checks from the profile output and keep the checks list small (typically 1-5) to stay bounded and interpretable.\n"
        "- Failure to follow the required order or to keep checks narrowly scoped invalidates the remediation workflow and must be treated as a process failure.\n\n"
        "Typical flow (plan):\n"
        "1) profile_collection_anomalies(...)\n"
        "2) Pick suspicious fields based on the profile's flags/metrics\n"
        "3) Run anomalies_tools(..., checks=[...])\n"
        "4) If needed, iterate with 1-2 additional checks (not dozens)\n\n"
        "Supported actions (current implementation):\n"
        "- dependency_check: validate an 'if X then Y' rule (find violations + sample a few _ids)\n"
        "- statistical_outlier: compute a percentile cutoff from a numeric sample and count/sample values above it\n"
        "- entropy_check: sample strings and compute Shannon entropy to surface gibberish-like values\n"
        "- fuzzy_match_anomaly: return near-duplicate values (edit distance) for a target string\n"
        "- velocity_check: aggregate counts into time buckets and flag spike/drops\n\n"
        "Examples:\n"
        "- [{'action':'dependency_check', 'if_field':'status', 'is_value':'active', 'then_field':'end_date', 'must_not_be':None}]\n"
        "- [{'action':'statistical_outlier', 'field':'amount', 'method':'percentile', 'threshold':99.5}]"
    )
)
@logged_tool()
def anomalies_tools(
    collection: Annotated[
        str,
        Field(
            description=(
                "Collection name to check. Accepts 'collection' or 'db.collection'.\n"
                "Examples: 'orders', 'ccaas.orders'"
            )
        )
    ],
    checks: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "List of check specifications. Each item MUST include 'action'.\n\n"
                "Keep this list SMALL (1-5). Derive checks from profile_collection_anomalies() output.\n\n"
                "Action: dependency_check\n"
                "- Purpose: enforce an 'if_field == is_value then then_field must NOT be must_not_be' rule\n"
                "- Required keys: action, if_field, then_field\n"
                "- Optional: is_value (scalar or list), must_not_be\n"
                "- Example 1 (null forbidden): {'action':'dependency_check', 'if_field':'status', 'is_value':'active', 'then_field':'end_date', 'must_not_be':None}\n"
                "- Example 2 (boolean forbidden): {'action':'dependency_check', 'if_field':'status', 'is_value':['cancelled', 'refunded'], 'then_field':'refund_processed', 'must_not_be':False}\n\n"
                "Action: statistical_outlier\n"
                "- Purpose: flag unusually high numeric values using a percentile cutoff computed from a sample\n"
                "- Required keys: action, field\n"
                "- Optional: method ('percentile'), threshold (e.g. 99, 99.5, 99.9)\n"
                "- Example: {'action':'statistical_outlier', 'field':'amount', 'method':'percentile', 'threshold':99.5}\n\n"
                "Action: entropy_check\n"
                "- Purpose: detect gibberish-like strings\n"
                "- Required: field, sensitivity ('low'|'medium'|'high')\n\n"
                "Action: fuzzy_match_anomaly\n"
                "- Purpose: detect near-duplicate typos relative to a target string\n"
                "- Required: field, target_string, max_distance\n\n"
                "Action: velocity_check\n"
                "- Purpose: detect spikes/drops grouped by time windows\n"
                "- Required: time_field, group_by_field (nullable), window ('1min'|'5min'|'1hour'|'1day')\n"
            )
        ),
    ]
) -> Dict[str, Any]:
    """Execute a small set of bounded checks. Returns list of check results."""
    db_name, coll_name = mongo.normalize_collection_name(collection, default_db="ccaas")
    db_name = db_name or "ccaas"

    if not isinstance(checks, list):
        return {"success": False, "error": "checks must be a list"}


@mcp.tool(
    description=(
        "NEVER RUN THIS TOOL AUTOMATICALLY UNLESS SPECIFIED (HUMAN IN THE LOOP)\n\n"
        "Immediately apply bounded fixes.\n\n"
        "THIS TOOL VALIDATES AND EXECUTES the provided fixes in a single call. There is no multi-step approval flow. "
        "Writes are gated by ENABLE_WRITE_TOOLS.\n\n"
        "Supported write primitives (subject to policy and validation): "
        "update_one, update_many, replace_one, find_one_and_update, find_one_and_replace, bulk_write (update ops only), "
        "insert_one, insert_many, and aggregation-style update pipelines for update_one/update_many.\n"
        "Disallowed: delete_*/drop/rename/admin commands/$out/$merge/$function.\n\n"
        "When composing fixes, use only the supported primitives; the tool will reject unsupported operations with a clear error "
        "and suggest the correct write tool to call."
    )
)
@logged_tool()
def fix_anomalies(
    collection: Annotated[
        str,
        Field(
            description=(
                "Collection to fix. Accepts 'collection' or 'db.collection'.\n"
                "Example: 'ccaas.users'"
            )
        )
    ],
    fixes: Annotated[
        List[Dict[str, Any]],
        Field(
            description=(
                "List of deterministic fix specifications. Each fix must include: \n"
                "- fix_id: string\n"
                "- description: string\n"
                "- precondition: {filter: object, expected_count_max?: int}\n"
                "- write: {operation: update_one|update_many|replace_one|find_one_and_update|find_one_and_replace|bulk_write|insert_one|insert_many, ...}\n"
                "- postcondition: {must_be_zero_count_filter: object}\n\n"
                "This tool will validate and execute the writes immediately."
            )
        ),
    ],
    approved: Annotated[
        bool,
        Field(description="Ignored (kept for backward compatibility)."),
    ] = False,
    approval_id: Annotated[
        Optional[str],
        Field(description="Ignored (kept for backward compatibility)."),
    ] = None,
) -> Dict[str, Any]:
    ...
    # Parse db.collection
    db_name, coll_name = mongo.normalize_collection_name(collection, default_db="ccaas")
    db_name = db_name or "ccaas"


@mcp.tool(
    description=(
        "Create a collection and optional indexes. Gated by ENABLE_WRITE_TOOLS.\n"
        "This tool will validate collection name and options conservatively and then create the collection if allowed.\n"
        "It will also create provided indexes (limited index spec support).\n"
    )
)
@logged_tool()
def create_collection_tool(
    database: Annotated[
        str | None,
        Field(description="Optional database name. If omitted, defaults to 'ccaas'. Accepts 'db' only."),
    ] = None,
    collection: Annotated[
        str,
        Field(
            description=(
                "Collection name (no database). Example: 'users' or pass 'db.collection' in database param if needed."
            )
        )
    ] = "",
    options: Annotated[
        Dict[str, Any] | None,
        Field(
            description=(
                "Optional creation options: allowed keys: capped(bool), size(int), max(int), validator(dict limited), indexes(list of index specs)"
            )
        ),
    ] = None,
) -> Dict[str, Any]:
    if not ENABLE_WRITE_TOOLS:
        return {"success": False, "error": "Write tools are disabled (set ENABLE_WRITE_TOOLS=true)"}

    # Normalize
    if database and isinstance(collection, str) and "." in collection:
        # user passed db.collection in collection param
        db_name, coll_name = mongo.normalize_collection_name(collection, default_db=database)
    else:
        db_name, coll_name = mongo.normalize_collection_name(
            f"{database + '.' if database else ''}{collection}", default_db="ccaas"
        )
    db_name = db_name or "ccaas"

    # Conservative name validation
    if not isinstance(coll_name, str) or not coll_name or not re.match(r"^[a-zA-Z0-9_\-]+$", coll_name):
        return {
            "success": False,
            "error": "Invalid collection name; allow only ascii letters, numbers, underscore, dash",
        }

    opts = options or {}
    allowed_opts = {"capped", "size", "max", "validator", "indexes"}
    for k in opts.keys():
        if k not in allowed_opts:
            return {"success": False, "error": f"Unsupported option: {k}"}

    # Build create collection spec (sanitized)
    # Note: create_collection is not implemented in run_mongo_db_write_code; we emulate by creating an index or a dummy update to ensure collection exists.
    _create_spec = {"operation": "create_collection", "database": db_name, "collection": coll_name, "options": {}}

    notes: List[str] = []
    indexes_created: List[Dict[str, Any]] = []


@mcp.tool(
    description=(
        "Insert documents into a collection (gated by ENABLE_WRITE_TOOLS).\n"
        "This tool validates and sanitizes documents, enforces batch size limits, and writes via the write runner.\n"
    )
)
@logged_tool()
def insert_documents_tool(
    collection: Annotated[
        str,
        Field(description="Target collection. Accepts 'collection' or 'db.collection'. Example: 'ccaas.users'"),
    ],
    documents: Annotated[
        List[Dict[str, Any]],
        Field(description="List of documents to insert. Non-empty list, each must be an object."),
    ],
    ordered: Annotated[
        bool, Field(description="If true, insert is ordered; on first failure stop. Default true")
    ] = True,
    bypass_document_validation: Annotated[
        bool, Field(description="Ignored for now; kept for API parity.")
    ] = False,
) -> Dict[str, Any]:
    if not ENABLE_WRITE_TOOLS:
        return {"success": False, "error": "Write tools are disabled (set ENABLE_WRITE_TOOLS=true)"}

    db_name, coll_name = mongo.normalize_collection_name(collection, default_db="ccaas")
    db_name = db_name or "ccaas"