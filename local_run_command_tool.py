from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_TOOL_OUTPUT_CHARS = 12000
TOOL_NAME = "run_command"
TOOL_DESCRIPTION = (
    "Run a local shell or CLI command and return the exit code, stdout, stderr, "
    "and working directory."
)
TOOL_PARAMETERS = {
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
                "Defaults to the current project folder."
            ),
        },
        "timeout_seconds": {
            "type": "integer",
            "description": "Optional timeout in seconds. Defaults to 60.",
        },
    },
    "required": ["command"],
}
OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": TOOL_PARAMETERS,
        },
    }
]


def output_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def clip_text(value: Any, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = output_to_text(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated to {limit} characters]"


def load_tool_params(params: Any) -> dict[str, Any]:
    if isinstance(params, dict):
        return params
    if isinstance(params, (bytes, bytearray)):
        params = bytes(params).decode("utf-8", errors="replace")
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


def execute_run_command(params: Any) -> str:
    data = load_tool_params(params)
    command = str(data["command"]).strip()
    if not command:
        raise ValueError("command must not be empty")

    working_directory = str(
        Path(data.get("working_directory") or PROJECT_ROOT).expanduser()
    )
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
