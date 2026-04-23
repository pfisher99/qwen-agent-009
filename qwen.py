from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from qwen_agent.agents import Assistant
    from qwen_agent.tools.base import BaseTool, register_tool
    from qwen_agent.utils.output_beautify import typewriter_print
except ImportError as exc:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "qwen-agent is not installed. Install it with: pip install -U qwen-agent"
    ) from exc


PROJECT_ROOT = Path(__file__).resolve().parent
base_url = "http://127.0.0.1:8000"
model_name = "Qwen3.5-9B-local"
api_key = "EMPTY"
MAX_TOOL_OUTPUT_CHARS = 6000

system_prompt = Path.read_text(PROJECT_ROOT / "system_prompt.txt", encoding="utf-8")


def clip_text(value: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(value) <= limit:
        return value
    return f"{value[:limit]}\n...[truncated to {limit} characters]"


def load_tool_params(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    if not isinstance(params, str):
        raise TypeError(f"Unsupported params type: {type(params).__name__}")

    text = params.strip()
    if not text:
        return {}
    return json.loads(text)


def build_shell_command(command: str) -> tuple[list[str], str]:
    if os.name == "nt":
        return ["powershell.exe", "-NoProfile", "-Command", command], "powershell"

    shell_path = shutil.which("bash") or shutil.which("sh") or "/bin/sh"
    shell_name = Path(shell_path).name
    return [shell_path, "-lc", command], shell_name


@register_tool("run_command")
class RunCommandTool(BaseTool):
    description = (
        "Run a local shell or CLI command and return the exit code, stdout, stderr, "
        "and working directory."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell or CLI command to execute.",
            },
            "working_directory": {
                "type": "string",
                "description": (
                    "Optional directory to run the command in. "
                    "Defaults to the folder containing qwen_agent.py."
                ),
            },
            "timeout_seconds": {
                "type": "integer",
                "description": "Optional timeout in seconds. Defaults to 60.",
            },
        },
        "required": ["command"],
    }

    def call(self, params: Any, **kwargs: Any) -> str:
        data = load_tool_params(params)
        command = str(data["command"]).strip()
        if not command:
            raise ValueError("command must not be empty")

        working_directory = str(Path(data.get("working_directory") or PROJECT_ROOT).expanduser())
        timeout_seconds = int(data.get("timeout_seconds", 60))
        timeout_seconds = max(1, min(timeout_seconds, 600))

        shell_cmd, shell_name = build_shell_command(command)

        try:
            completed = subprocess.run(
                shell_cmd,
                cwd=working_directory,
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"Command timed out after {timeout_seconds} seconds.",
                    "shell": shell_name,
                    "command": command,
                    "working_directory": working_directory,
                    "stdout": clip_text(exc.stdout or ""),
                    "stderr": clip_text(exc.stderr or ""),
                },
                ensure_ascii=False,
                indent=2,
            )
        except (FileNotFoundError, NotADirectoryError, PermissionError, OSError) as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "shell": shell_name,
                    "command": command,
                    "working_directory": working_directory,
                },
                ensure_ascii=False,
                indent=2,
            )

        return json.dumps(
            {
                "ok": completed.returncode == 0,
                "shell": shell_name,
                "command": command,
                "working_directory": working_directory,
                "exit_code": completed.returncode,
                "stdout": clip_text(completed.stdout),
                "stderr": clip_text(completed.stderr),
            },
            ensure_ascii=False,
            indent=2,
        )


tools: list[Any] = ["run_command"]


def normalize_model_server(url: str) -> str:
    cleaned = url.rstrip("/")
    if cleaned.endswith("/v1"):
        return cleaned
    return f"{cleaned}/v1"


def build_agent() -> Assistant:
    llm_cfg = {
        "model": model_name,
        "model_server": normalize_model_server(base_url),
        "api_key": api_key,
    }
    return Assistant(
        llm=llm_cfg,
        system_message=system_prompt,
        function_list=tools,
        name="Local Qwen Agent",
        description="A simple one-file Qwen-Agent wired to a local OpenAI-compatible API.",
    )


def run_once(query: str) -> None:
    bot = build_agent()
    messages = [{"role": "user", "content": query}]
    response = []
    response_plain_text = ""
    for response in bot.run(messages=messages):
        response_plain_text = typewriter_print(response, response_plain_text)
    if response_plain_text:
        print()


def repl() -> None:
    bot = build_agent()
    messages: list[dict[str, Any]] = []

    print(f"Model: {model_name}")
    print(f"Server: {normalize_model_server(base_url)}")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        query = input("\nyou> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": query})
        response = []
        response_plain_text = ""

        print("assistant> ", end="")
        for response in bot.run(messages=messages):
            response_plain_text = typewriter_print(response, response_plain_text)
        print()

        messages.extend(response)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_once(" ".join(sys.argv[1:]))
    else:
        repl()
