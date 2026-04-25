from __future__ import annotations

import argparse
import atexit
import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from local_msfconsole_tool import (
    OPENAI_TOOL as MSFCONSOLE_OPENAI_TOOL,
    TOOL_NAME as MSFCONSOLE_TOOL_NAME,
    execute_msfconsole,
)
from local_run_command_tool import (
    OPENAI_TOOLS as RUN_COMMAND_OPENAI_TOOLS,
    TOOL_NAME as RUN_COMMAND_TOOL_NAME,
    execute_run_command,
)
from local_web_tools import (
    OPENAI_TOOLS as WEB_OPENAI_TOOLS,
    OPEN_URL_TOOL_NAME,
    WEB_SEARCH_TOOL_NAME,
    execute_open_url,
    execute_web_search,
)


PROJECT_ROOT = Path(__file__).resolve().parent
base_url = "http://192.168.11.90:8000"
model_name = "Qwen3.5-9B-local"
api_key = "EMPTY"
ENABLE_THINKING = True
REQUEST_TIMEOUT_SECONDS = 300
TOKENIZER_TIMEOUT_SECONDS = 60
MAX_TOOL_ROUNDS_PER_TURN = 200
DEFAULT_MAX_OUTPUT_TOKENS = 80000
MAX_OUTPUT_TOKENS = min(
    int(os.environ.get("QWEN_MAX_OUTPUT_TOKENS", str(DEFAULT_MAX_OUTPUT_TOKENS))),
    DEFAULT_MAX_OUTPUT_TOKENS,
)
DEFAULT_MAX_CONTEXT_TOKENS = 160000
MAX_CONTEXT_TOKENS = min(
    int(os.environ.get("QWEN_MAX_CONTEXT_TOKENS", str(DEFAULT_MAX_CONTEXT_TOKENS))),
    DEFAULT_MAX_CONTEXT_TOKENS,
)
WEB_TOOLS_ENABLED = os.environ.get("QWEN_ENABLE_WEB_TOOLS") == "1"
base_system_prompt = Path.read_text(PROJECT_ROOT / "system_prompt.txt", encoding="utf-8")
web_tools_prompt = (
    "Web tools are enabled for this run."
    if WEB_TOOLS_ENABLED
    else "Web tools are disabled for this run; do not call web_search or open_url."
)
system_prompt = f"{base_system_prompt}\n{web_tools_prompt}"
OPENAI_TOOLS = [
    *RUN_COMMAND_OPENAI_TOOLS,
    MSFCONSOLE_OPENAI_TOOL,
    *(WEB_OPENAI_TOOLS if WEB_TOOLS_ENABLED else []),
]
ANSI_CSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_OSC_PATTERN = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
ANSI_ESC_PATTERN = re.compile(r"\x1b[@-Z\\-_]")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
TERMINAL_RESET_SEQUENCE = "\x1b[0m\x1b[?25h\x1b[?2004l"

generate_cfg = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "max_tokens": MAX_OUTPUT_TOKENS,
    "presence_penalty": 1.5,
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


class TerminalGuard:
    def __init__(self) -> None:
        self.fd: int | None = None
        self.attrs: list[Any] | None = None
        if not sys.stdin.isatty():
            return

        try:
            import termios

            self.fd = sys.stdin.fileno()
            self.attrs = termios.tcgetattr(self.fd)
        except (ImportError, OSError, ValueError):
            self.fd = None
            self.attrs = None

    def restore_mode(self) -> None:
        if self.fd is None or self.attrs is None:
            return

        try:
            import termios

            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.attrs)
        except (ImportError, OSError, ValueError):
            pass

    def reset_display(self) -> None:
        for stream in (sys.stdout, sys.stderr):
            if not stream.isatty():
                continue
            try:
                stream.write(TERMINAL_RESET_SEQUENCE)
                stream.flush()
            except OSError:
                pass

    def restore(self) -> None:
        self.restore_mode()
        self.reset_display()


TERMINAL_GUARD = TerminalGuard()
atexit.register(TERMINAL_GUARD.restore)


def strip_terminal_sequences(text: str) -> str:
    text = ANSI_OSC_PATTERN.sub("", text)
    text = ANSI_CSI_PATTERN.sub("", text)
    text = ANSI_ESC_PATTERN.sub("", text)
    text = CONTROL_CHARS_PATTERN.sub("", text)
    return text.replace("\r\n", "\n").replace("\r", "\n")


def terminal_sequence_is_complete(text: str) -> bool:
    if text.startswith("\x1b["):
        return bool(ANSI_CSI_PATTERN.match(text))
    if text.startswith("\x1b]"):
        return "\x07" in text or "\x1b\\" in text
    return len(text) >= 2 and "@" <= text[1] <= "_"


def split_incomplete_terminal_sequence(text: str) -> tuple[str, str]:
    escape_index = text.rfind("\x1b")
    if escape_index == -1:
        return text, ""

    tail = text[escape_index:]
    if terminal_sequence_is_complete(tail):
        return text, ""
    return text[:escape_index], tail


class TerminalOutputSanitizer:
    def __init__(self) -> None:
        self.pending = ""

    def clean(self, value: Any) -> str:
        text = self.pending + stringify_any(value)
        text, self.pending = split_incomplete_terminal_sequence(text)
        return strip_terminal_sequences(text)

    def clear(self) -> None:
        self.pending = ""


STDOUT_TERMINAL_SANITIZER = TerminalOutputSanitizer()
STDERR_TERMINAL_SANITIZER = TerminalOutputSanitizer()


def sanitize_terminal_text(value: Any) -> str:
    return strip_terminal_sequences(stringify_any(value))


def print_stream_text(value: Any, *, file: Any = None) -> None:
    target = file or sys.stdout
    sanitizer = STDERR_TERMINAL_SANITIZER if target is sys.stderr else STDOUT_TERMINAL_SANITIZER
    print(sanitizer.clean(value), end="", file=target, flush=True)


def clear_terminal_output_sanitizers() -> None:
    STDOUT_TERMINAL_SANITIZER.clear()
    STDERR_TERMINAL_SANITIZER.clear()


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


def normalize_model_root(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned[:-3]
    return cleaned


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


def format_tool_call_for_console(tool_name: Any, arguments: Any) -> str:
    name = sanitize_terminal_text(tool_name).strip() or "tool"
    call_arguments = sanitize_terminal_text(arguments).strip()
    if not call_arguments:
        return f"[tool] {name}"
    return f"[tool] {name} {call_arguments}"


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
        name = str(function.get("name") or "").strip() or "tool"
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


def compact_completed_turn(
    messages: list[dict[str, Any]], turn_start_index: int
) -> None:
    if turn_start_index < 0 or turn_start_index >= len(messages):
        return

    user_message = messages[turn_start_index]
    final_message = messages[-1]
    if user_message.get("role") != "user" or final_message.get("role") != "assistant":
        return

    messages[turn_start_index:] = [
        {"role": "user", "content": make_json_safe(user_message.get("content", ""))},
        {
            "role": "assistant",
            "content": stringify_any(final_message.get("content", ""), strip=True),
        },
    ]


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


def build_tokenizer_body(messages: list[dict[str, Any]], with_tools: bool) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "chat_template_kwargs": {"enable_thinking": ENABLE_THINKING},
    }
    if with_tools:
        body["tools"] = OPENAI_TOOLS
        body["tool_choice"] = "auto"
    return body


def count_chat_tokens(messages: list[dict[str, Any]], with_tools: bool) -> int:
    url = f"{normalize_model_root(base_url)}/tokenize"
    request = urllib.request.Request(
        url=url,
        data=json.dumps(build_tokenizer_body(messages, with_tools)).encode("utf-8"),
        headers=build_headers(),
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=TOKENIZER_TIMEOUT_SECONDS) as response:
            result = json.loads(response.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"Tokenizer failed: HTTP {exc.code}\n{error_text}") from exc
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Tokenizer failed: {exc}") from exc

    count = result.get("count")
    if isinstance(count, int):
        return count

    tokens = result.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)

    raise SystemExit(f"Tokenizer response did not include a token count: {result}")


def default_preserve_from(messages: list[dict[str, Any]]) -> int:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "user":
            return index
    return 1 if messages and messages[0].get("role") == "system" else 0


def history_start_index(messages: list[dict[str, Any]]) -> int:
    return 1 if messages and messages[0].get("role") == "system" else 0


def trim_old_history_to_token_budget(
    messages: list[dict[str, Any]],
    with_tools: bool,
    preserve_from: int,
) -> tuple[list[dict[str, Any]], int, int, int]:
    history_start = history_start_index(messages)
    preserve_from = max(history_start, min(preserve_from, len(messages)))
    token_count = count_chat_tokens(messages, with_tools)
    removed_messages = 0

    while token_count > MAX_CONTEXT_TOKENS and history_start < preserve_from:
        drop_until = preserve_from
        for index in range(history_start + 1, preserve_from):
            if messages[index].get("role") == "user":
                drop_until = index
                break

        removed_count = drop_until - history_start
        del messages[history_start:drop_until]
        preserve_from -= removed_count
        removed_messages += removed_count
        token_count = count_chat_tokens(messages, with_tools)

    return messages, preserve_from, removed_messages, token_count


def enforce_message_token_budget(
    messages: list[dict[str, Any]],
    with_tools: bool,
    preserve_from: int | None = None,
) -> int:
    if not messages:
        return 0

    clean_messages = sanitize_messages_for_api(messages)
    if preserve_from is None:
        preserve_from = default_preserve_from(clean_messages)
    preserve_from = max(0, min(preserve_from, len(clean_messages) - 1))

    original_count = count_chat_tokens(clean_messages, with_tools)
    clean_messages, preserve_from, removed_messages, token_count = (
        trim_old_history_to_token_budget(clean_messages, with_tools, preserve_from)
    )

    if token_count > MAX_CONTEXT_TOKENS:
        raise SystemExit(
            "Message history cannot fit within "
            f"{MAX_CONTEXT_TOKENS} tokens even after dropping old history "
            f"(current count: {token_count})."
        )

    if removed_messages:
        print(
            "[history] dropped old history "
            f"{original_count} -> {token_count} tokens "
            f"(removed {removed_messages} old message"
            f"{'' if removed_messages == 1 else 's'}).",
            flush=True,
        )

    messages[:] = clean_messages
    return preserve_from


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
                    print_stream_text(reasoning_delta, file=sys.stderr)

            content_delta = delta.get("content") or ""
            if content_delta:
                assistant_message["content"] += content_delta
                print_stream_text(content_delta)
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
    clear_terminal_output_sanitizers()
    TERMINAL_GUARD.restore()

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

        print(format_tool_call_for_console(tool_name, arguments), flush=True)

        if tool_name == RUN_COMMAND_TOOL_NAME:
            tool_output = execute_run_command(arguments)
        elif tool_name == MSFCONSOLE_TOOL_NAME:
            tool_output = execute_msfconsole(arguments)
        elif tool_name == WEB_SEARCH_TOOL_NAME and WEB_TOOLS_ENABLED:
            tool_output = execute_web_search(arguments)
        elif tool_name == OPEN_URL_TOOL_NAME and WEB_TOOLS_ENABLED:
            tool_output = execute_open_url(arguments)
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
    turn_start_index = len(messages) - 1
    tool_rounds = 0
    force_no_tools = False

    while True:
        tools_enabled_for_request = with_tools and not force_no_tools
        turn_start_index = enforce_message_token_budget(
            messages,
            with_tools=tools_enabled_for_request,
            preserve_from=turn_start_index,
        )
        assistant_message, finish_reason = stream_chat_completion(
            messages,
            with_tools=tools_enabled_for_request,
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
            # Keep completed tool transcripts in history so later turns can
            # reason from exact commands, outputs, and tool state transitions.
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
    print(f"Web tools: {'enabled' if WEB_TOOLS_ENABLED else 'disabled'}")
    print(f"Step logs: {step_logger.session_dir}")
    print(f"Think debug: {'on' if debug_think else 'off'}")
    print(f"Raw debug: {'on' if debug_raw else 'off'}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        TERMINAL_GUARD.restore()
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
