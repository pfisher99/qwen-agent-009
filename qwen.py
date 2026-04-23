from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from local_run_command_tool import OPENAI_TOOLS, TOOL_NAME, execute_run_command


PROJECT_ROOT = Path(__file__).resolve().parent
base_url = "http://192.168.11.90:8000"
model_name = "Qwen3.5-9B-local"
api_key = "EMPTY"
ENABLE_THINKING = True
REQUEST_TIMEOUT_SECONDS = 120
MAX_TOOL_ROUNDS_PER_TURN = 8
system_prompt = Path.read_text(PROJECT_ROOT / "system_prompt.txt", encoding="utf-8")

generate_cfg = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}


class StepLogger:
    def __init__(self) -> None:
        self.session_dir = create_step_log_dir()
        self.step_number = 0

    def write_model_call(
        self,
        assistant_message: dict[str, Any],
        tool_messages: list[dict[str, Any]] | None = None,
    ) -> Path:
        self.step_number += 1
        path = self.session_dir / f"step{self.step_number}.txt"
        text = format_step_log(
            self.step_number,
            assistant_message,
            tool_messages or [],
        )
        path.write_text(text, encoding="utf-8")
        return path


def make_json_safe(value: Any) -> Any:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return make_json_safe(model_dump(mode="json"))
        except TypeError:
            return make_json_safe(model_dump())

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        return make_json_safe(dict_method())

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    return str(value)


def stringify_any(value: Any, strip: bool = False) -> str:
    safe_value = make_json_safe(value)
    if safe_value is None:
        text = ""
    elif isinstance(safe_value, str):
        text = safe_value
    else:
        text = json.dumps(safe_value, ensure_ascii=False, indent=2)

    if strip:
        return text.strip()
    return text


def normalize_model_server(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def create_step_log_dir() -> Path:
    root = PROJECT_ROOT / "step_logs"
    root.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = root / stamp
    suffix = 1
    while session_dir.exists():
        suffix += 1
        session_dir = root / f"{stamp}-{suffix}"

    session_dir.mkdir()
    return session_dir


def json_default(value: Any) -> Any:
    return make_json_safe(value)


def format_text_block(value: Any) -> str:
    return stringify_any(value, strip=True)


def format_step_log(
    step_number: int,
    assistant_message: dict[str, Any],
    tool_messages: list[dict[str, Any]],
) -> str:
    sections = [f"Step {step_number}"]

    reasoning = format_text_block(assistant_message.get("reasoning"))
    if reasoning:
        sections.append(f"<think>\n{reasoning}\n</think>")

    content = format_text_block(assistant_message.get("content"))
    if content:
        sections.append(content)

    tool_calls = assistant_message.get("tool_calls") or []
    for index, tool_call in enumerate(tool_calls, start=1):
        function = tool_call.get("function") or {}
        name = str(function.get("name") or "").strip() or TOOL_NAME
        arguments = format_text_block(function.get("arguments"))
        tool_call_lines = [f"[tool_call {index}] {name}"]
        if arguments:
            tool_call_lines.append(arguments)
        sections.append("\n".join(tool_call_lines))

    for index, tool_message in enumerate(tool_messages, start=1):
        content = format_text_block(tool_message.get("content"))
        if not content:
            continue
        sections.append(f"[tool_result {index}]\n{content}")

    return "\n\n".join(section for section in sections if section).strip() + "\n"


def build_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def sanitize_messages_for_api(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean_messages: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role") or "")
        clean: dict[str, Any] = {"role": role}

        if role in {"system", "user", "assistant"}:
            if "content" in message:
                clean["content"] = make_json_safe(message.get("content", ""))
            if role == "assistant" and message.get("tool_calls"):
                clean["tool_calls"] = make_json_safe(message["tool_calls"])
        elif role == "tool":
            clean["content"] = stringify_any(message.get("content", ""))
            clean["tool_call_id"] = stringify_any(
                message.get("tool_call_id", ""), strip=True
            )
        else:
            clean["content"] = make_json_safe(message.get("content", ""))

        clean_messages.append(clean)

    return clean_messages


def build_request_body(
    messages: list[dict[str, Any]], with_tools: bool, stream: bool
) -> dict[str, Any]:
    body = {
        "model": model_name,
        "messages": sanitize_messages_for_api(messages),
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
        **generate_cfg,
    }
    if with_tools:
        body["tools"] = OPENAI_TOOLS
        body["tool_choice"] = "auto"
    return body


def append_tool_call_delta(
    tool_calls: list[dict[str, Any]], delta_calls: list[dict[str, Any]]
) -> None:
    for delta_call in delta_calls:
        index = int(delta_call.get("index", len(tool_calls)))
        while len(tool_calls) <= index:
            tool_calls.append(
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                }
            )

        target = tool_calls[index]
        if delta_call.get("id"):
            target["id"] = delta_call["id"]
        if delta_call.get("type"):
            target["type"] = delta_call["type"]

        function_delta = delta_call.get("function") or {}
        if function_delta.get("name"):
            target["function"]["name"] = function_delta["name"]
        if function_delta.get("arguments"):
            target["function"]["arguments"] += function_delta["arguments"]


def emit_debug_raw(event: dict[str, Any]) -> None:
    print("\n[DEBUG RAW CHUNK]", file=sys.stderr)
    print(
        json.dumps(make_json_safe(event), ensure_ascii=False, indent=2),
        file=sys.stderr,
        flush=True,
    )


def stream_chat_completion(
    messages: list[dict[str, Any]],
    with_tools: bool,
    debug_raw: bool,
    debug_think: bool,
) -> tuple[dict[str, Any], str | None]:
    url = f"{normalize_model_server(base_url)}/chat/completions"
    body = build_request_body(messages, with_tools=with_tools, stream=True)
    request = urllib.request.Request(
        url=url,
        data=json.dumps(body).encode("utf-8"),
        headers=build_headers(),
        method="POST",
    )

    assistant_message: dict[str, Any] = {
        "role": "assistant",
        "content": "",
        "reasoning": "",
        "tool_calls": [],
    }
    finish_reason: str | None = None
    think_started = False
    printed_output = False

    try:
        response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS)
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Request failed: HTTP {exc.code}\n{error_text}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"Request failed: {exc}") from exc

    with response:
        for raw_line in response:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue

            payload = line[5:].strip()
            if payload == "[DONE]":
                break

            event = json.loads(payload)
            if debug_raw:
                emit_debug_raw(event)

            choice = (event.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            finish_reason = choice.get("finish_reason") or finish_reason

            reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if reasoning_delta:
                assistant_message["reasoning"] += reasoning_delta
                if debug_think:
                    if not think_started:
                        print("\n<think>", file=sys.stderr)
                        think_started = True
                    print(reasoning_delta, end="", file=sys.stderr, flush=True)

            content_delta = delta.get("content") or ""
            if content_delta:
                assistant_message["content"] += content_delta
                print(content_delta, end="", flush=True)
                printed_output = True

            delta_tool_calls = delta.get("tool_calls") or []
            if delta_tool_calls:
                append_tool_call_delta(assistant_message["tool_calls"], delta_tool_calls)

            if finish_reason:
                break

    if debug_think and think_started:
        print("\n</think>", file=sys.stderr)
    if printed_output:
        print()

    if not assistant_message["tool_calls"]:
        assistant_message.pop("tool_calls")
    if not assistant_message["reasoning"]:
        assistant_message.pop("reasoning")

    return assistant_message, finish_reason


def execute_tool_calls(assistant_message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_messages: list[dict[str, Any]] = []

    for tool_call in assistant_message.get("tool_calls", []):
        function = tool_call.get("function") or {}
        tool_name = function.get("name", "")
        arguments = function.get("arguments", "")

        print(f"[tool] {tool_name}", flush=True)

        if tool_name == TOOL_NAME:
            tool_output = execute_run_command(arguments)
        else:
            tool_output = json.dumps(
                {"ok": False, "error": f"Unknown tool: {tool_name}"},
                ensure_ascii=False,
                indent=2,
            )

        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": stringify_any(tool_output),
            }
        )

    return tool_messages


def chat_until_complete(
    messages: list[dict[str, Any]],
    with_tools: bool,
    debug_raw: bool,
    debug_think: bool,
    step_logger: StepLogger,
) -> list[dict[str, Any]]:
    tool_rounds = 0
    force_no_tools = False

    while True:
        assistant_message, finish_reason = stream_chat_completion(
            messages,
            with_tools=with_tools and not force_no_tools,
            debug_raw=debug_raw,
            debug_think=debug_think,
        )
        messages.append(assistant_message)

        tool_messages: list[dict[str, Any]] = []
        if (
            with_tools
            and not force_no_tools
            and finish_reason == "tool_calls"
            and assistant_message.get("tool_calls")
        ):
            tool_rounds += 1
            tool_messages = execute_tool_calls(assistant_message)
            messages.extend(tool_messages)
            if tool_rounds >= MAX_TOOL_ROUNDS_PER_TURN:
                print(
                    "[tool] tool round limit reached; asking for a summary without more tools.",
                    flush=True,
                )
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Tool round limit reached for this turn. Summarize the "
                            "results so far and ask what to do next. Do not call tools."
                        ),
                    }
                )
                force_no_tools = True

        step_logger.write_model_call(assistant_message, tool_messages)

        if not tool_messages:
            return messages


def run_once(query: str, debug_raw: bool = False, debug_think: bool = False) -> None:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    step_logger = StepLogger()
    messages.append({"role": "user", "content": query})
    print(f"Step logs: {step_logger.session_dir}")
    chat_until_complete(
        messages,
        with_tools=True,
        debug_raw=debug_raw,
        debug_think=debug_think,
        step_logger=step_logger,
    )


def repl(debug_raw: bool = False, debug_think: bool = False) -> None:
    messages: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    step_logger = StepLogger()

    print(f"Model: {model_name}")
    print(f"Server: {normalize_model_server(base_url)}")
    print("Tooling: enabled")
    print(f"Step logs: {step_logger.session_dir}")
    print(f"Think debug: {'on' if debug_think else 'off'}")
    print(f"Raw debug: {'on' if debug_raw else 'off'}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        query = input("user question: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": query})
        chat_until_complete(
            messages,
            with_tools=True,
            debug_raw=debug_raw,
            debug_think=debug_think,
            step_logger=step_logger,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chat with the local OpenAI-compatible server without Qwen-Agent."
    )
    parser.add_argument(
        "--debug-raw",
        action="store_true",
        help="Print each raw streamed event to stderr.",
    )
    parser.add_argument(
        "--debug-think",
        action="store_true",
        help="Mirror streamed reasoning deltas to stderr inside <think>...</think>.",
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Optional prompt to run once. If omitted, starts the TUI.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.query:
        run_once(
            " ".join(args.query),
            debug_raw=args.debug_raw,
            debug_think=args.debug_think,
        )
    else:
        repl(debug_raw=args.debug_raw, debug_think=args.debug_think)
