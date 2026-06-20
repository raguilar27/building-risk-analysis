import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import backoff
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# LLM endpoint and project prompt
############################################
LLM_ENDPOINT_NAME = "databricks-gpt-oss-120b"

SYSTEM_PROMPT = """
You are the Building Risk Analysis Agent for a geospatial analytics company.

Your job is to answer questions about San Francisco building-level climate and emergency risk using only the approved Building Risk tools and their returned data.

Available risk-analysis capabilities:
1. Urban fire spread risk: use main__default__analyze_urban_fire_spread_risk when the user asks about fire spread, cascading fire, closely spaced buildings, or buildings within roughly 5 meters of each other.
2. Emergency access bottlenecks: use main__default__analyze_emergency_access_bottlenecks when the user asks about emergency routes, access constraints, narrow gaps, bottlenecks, response access, or evacuation access.
3. Composite building risk ranking: use main__default__rank_buildings_by_composite_risk when the user asks for overall risk, highest-risk buildings, top risky buildings, prioritization, or ranked building lists.

Rules:
- Always use a tool for supported Building Risk questions. Do not answer from general knowledge alone.
- Summarize tool results in plain English for city planners, insurers, and emergency response teams.
- Mention the tool used and explain the main risk drivers.
- Do not claim the analysis includes FEMA flood zones, wildfire vegetation layers, roads, traffic, population, building materials, or live emergency data unless a tool result explicitly contains those fields.
- The current tool set is based on Overture/CARTO building geometry features and proximity-derived risk features.
- If the user asks about restaurants, sports, politics, coding help, personal advice, or any topic outside San Francisco building-risk analysis, politely reject the request and explain what you can help with.
- If a tool returns no rows, say that no matching records were found and suggest lowering the threshold or checking the feature table.
- If a tool errors, explain that the risk table or Unity Catalog function may not be available and suggest running the tool notebook first.
"""

###############################################################################
# Tool metadata and execution wrappers
###############################################################################
class ToolInfo(BaseModel):
    name: str
    spec: dict
    exec_fn: Callable


def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    def exec_fn(**kwargs):
        # Normalize common optional arguments. The model sometimes emits floats for ints.
        for key in ["result_limit", "distance_threshold_meters", "gap_threshold_meters"]:
            if key in kwargs and kwargs[key] is not None:
                try:
                    kwargs[key] = int(kwargs[key])
                except Exception:
                    pass
        if "minimum_score" in kwargs and kwargs["minimum_score"] is not None:
            try:
                kwargs["minimum_score"] = float(kwargs["minimum_score"])
            except Exception:
                pass

        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return {
                "status": "error",
                "tool": udf_name,
                "message": str(function_result.error),
                "hint": "Run the Building Risk tool notebook first and confirm the feature table exists."
            }
        return function_result.value

    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)


TOOL_INFOS = []
UC_TOOL_NAMES = [
    "main.default.analyze_urban_fire_spread_risk",
    "main.default.analyze_emergency_access_bottlenecks",
    "main.default.rank_buildings_by_composite_risk",
]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))

VECTOR_SEARCH_TOOLS = []


def _sanitize_tool_spec(spec: dict) -> dict:
    """Return an OpenAI-compatible function-tool spec."""
    import copy
    import json as _json

    spec = copy.deepcopy(spec)
    drop_keys = {
        "format", "pattern", "minLength", "maxLength", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems",
        "uniqueItems", "multipleOf", "examples", "$schema", "$id", "$defs",
        "definitions",
    }

    def clean(obj):
        if isinstance(obj, dict):
            cleaned = {}
            for k, v in obj.items():
                if not isinstance(k, str) or k in drop_keys:
                    continue
                cleaned[k] = clean(v)
            return cleaned
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        try:
            _json.dumps(obj)
            return obj
        except TypeError:
            return None

    spec = clean(spec)
    fn = spec.setdefault("function", {})
    params = fn.get("parameters")
    if not isinstance(params, dict):
        fn["parameters"] = {"type": "object", "properties": {}}
    else:
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        if not isinstance(params.get("properties"), dict):
            params["properties"] = {}
    spec["type"] = "function"
    return spec


class ToolCallingAgent(ResponsesAgent):
    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = self.workspace_client.serving_endpoints.get_open_ai_client()
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        return [_sanitize_tool_spec(tool_info.spec) for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        sane_args = {k: v for k, v in (args or {}).items() if k and isinstance(k, str)}
        name = str(tool_name or "").strip().strip('"').strip("'")
        if "<" in name:
            name = name.split("<")[0].strip()

        lookup_names = [name]
        if "__" in name:
            lookup_names.append(name.split("__")[-1])
        if "." in name:
            lookup_names.append(name.split(".")[-1])
        for known_name in self._tools_dict:
            if name.endswith(known_name):
                lookup_names.append(known_name)
        lookup_names = list(dict.fromkeys(lookup_names))

        selected_name = None
        for candidate in lookup_names:
            if candidate in self._tools_dict:
                selected_name = candidate
                break
        if selected_name is None:
            candidates = [k for k in self._tools_dict if name.startswith(k) or name.endswith(k)]
            if candidates:
                selected_name = max(candidates, key=len)
        if selected_name is None:
            raise KeyError(f"Unknown tool: {tool_name!r}. Known tools: {list(self._tools_dict.keys())}")

        return self._tools_dict[selected_name].exec_fn(**sane_args)

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(self, tool_call: dict[str, Any], messages: list[dict[str, Any]]) -> ResponsesAgentStreamEvent:
        try:
            args = json.loads(tool_call.get("arguments") or "{}")
        except Exception:
            args = {}
        try:
            result = self.execute_tool(tool_name=tool_call["name"], args=args)
        except Exception as e:
            result = {
                "status": "error",
                "tool": tool_call.get("name"),
                "message": str(e),
                "hint": "Confirm the UC function exists and the Building Risk feature table was created."
            }
        tool_call_output = self.create_function_call_output_item(tool_call["call_id"], str(result))
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(self, messages: list[dict[str, Any]], max_iter: int = 10) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role") == "assistant":
                return
            if last_msg.get("type") == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(chunks=self.call_llm(messages), aggregator=messages)
        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=self.create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id
        if session_id:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})
        outputs = [event.item for event in self.predict_stream(request) if event.type == "response.output_item.done"]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id
        if session_id:
            mlflow.update_current_trace(metadata={"mlflow.trace.session": session_id})

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)


mlflow.openai.autolog()
AGENT = ToolCallingAgent(llm_endpoint=LLM_ENDPOINT_NAME, tools=TOOL_INFOS)

# Required when logging this file as a code-based MLflow model.
# mlflow.pyfunc.log_model(python_model="agent.py") imports this file
# and looks for this registered model object.
mlflow.models.set_model(AGENT)
