from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
TOOL_NAME = "msfconsole"
DEFAULT_MAX_TOOL_OUTPUT_CHARS = 30000
DEFAULT_MAX_BUFFER_CHARS = 200000
MSFCONSOLE_COMMAND = ["msfconsole", "-q"]
ANSI_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ANSI_OSC_PATTERN = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)")
ANSI_ESC_PATTERN = re.compile(r"\x1b[@-Z\\-_]")
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

TOOL_DESCRIPTION = (
    "Control one persistent msfconsole session for testing. "
    "Use start to launch it, send to write one command, read to collect more "
    "output, status to inspect the session, and stop when finished."
)
TOOL_PARAMETERS = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["start", "send", "read", "status", "stop"],
            "description": (
                "Session action. Defaults to send when command is provided, "
                "otherwise status."
            ),
        },
        "command": {
            "type": "string",
            "description": "Command to send to msfconsole when action is send.",
        },
        "wait_seconds": {
            "type": "number",
            "description": (
                "Seconds to wait before collecting output. Defaults to 3 for "
                "start/send/read and is capped at 120."
            ),
        },
    },
}
OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": TOOL_PARAMETERS,
    },
}


def positive_int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, str(default)))
    except ValueError:
        return default
    return max(1, value)


MAX_TOOL_OUTPUT_CHARS = positive_int_from_env(
    "MSFCONSOLE_MAX_TOOL_OUTPUT_CHARS",
    DEFAULT_MAX_TOOL_OUTPUT_CHARS,
)
MAX_BUFFER_CHARS = positive_int_from_env(
    "MSFCONSOLE_MAX_BUFFER_CHARS",
    DEFAULT_MAX_BUFFER_CHARS,
)


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


def output_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def render_terminal_text(text: str) -> str:
    text = text.replace("\r\n", "\n")
    lines: list[str] = []
    line: list[str] = []
    column = 0

    for char in text:
        if char == "\r":
            line = []
            column = 0
            continue
        if char == "\n":
            lines.append("".join(line).rstrip())
            line = []
            column = 0
            continue
        if char == "\b" or char == "\x7f":
            column = max(0, column - 1)
            continue
        if char == "\t":
            char = " "
        elif ord(char) < 32:
            continue

        if column < len(line):
            line[column] = char
        else:
            if column > len(line):
                line.extend(" " for _ in range(column - len(line)))
            line.append(char)
        column += 1

    if line:
        lines.append("".join(line).rstrip())
    return "\n".join(lines)


def clean_output(value: Any) -> str:
    text = output_to_text(value)
    text = ANSI_OSC_PATTERN.sub("", text)
    text = ANSI_PATTERN.sub("", text)
    text = ANSI_ESC_PATTERN.sub("", text)
    text = render_terminal_text(text)
    return CONTROL_CHARS_PATTERN.sub("", text)


def clip_text(value: Any, limit: int | None = MAX_TOOL_OUTPUT_CHARS) -> str:
    text = clean_output(value)
    if limit is None:
        return text
    if len(text) <= limit:
        return text
    marker = "\n...[truncated middle output]...\n"
    content_budget = max(0, limit - len(marker))
    head_chars = content_budget // 3
    tail_chars = content_budget - head_chars
    if tail_chars <= 0:
        return text[:content_budget]
    return f"{text[:head_chars]}{marker}{text[-tail_chars:]}"


def strip_command_echo(output: str, command: str) -> str:
    if not output:
        return output

    lines = output.splitlines(keepends=True)
    if not lines:
        return output

    first_line = lines[0].strip()
    clean_command = command.strip()
    if first_line == clean_command or first_line.endswith(clean_command):
        return "".join(lines[1:]).lstrip("\n")
    return output


def bounded_wait_seconds(value: Any, default: float = 3.0) -> float:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        seconds = default
    return max(0.0, min(seconds, 120.0))


def command_has_path_separator(command: str) -> bool:
    return os.sep in command or bool(os.altsep and os.altsep in command)


def resolve_msfconsole_command() -> list[str] | None:
    executable = MSFCONSOLE_COMMAND[0]
    if Path(executable).is_absolute() or command_has_path_separator(executable):
        if Path(executable).exists():
            return MSFCONSOLE_COMMAND
        return None

    resolved = shutil.which(executable)
    if not resolved:
        return None
    return [resolved, *MSFCONSOLE_COMMAND[1:]]


class MsfConsoleSession:
    def __init__(self) -> None:
        self.process: subprocess.Popen[bytes] | None = None
        self.master_fd: int | None = None
        self.reader_thread: threading.Thread | None = None
        self._output = ""
        self._output_truncated = False
        self._process_lock = threading.RLock()
        self._output_lock = threading.Lock()

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def buffered_output_chars(self) -> int:
        with self._output_lock:
            return len(self._output)

    def append_output(self, text: Any) -> None:
        output = output_to_text(text)
        if not output:
            return
        with self._output_lock:
            self._output += output
            if MAX_BUFFER_CHARS is not None and len(self._output) > MAX_BUFFER_CHARS:
                self._output = self._output[-MAX_BUFFER_CHARS:]
                self._output_truncated = True

    def drain_output(self) -> str:
        with self._output_lock:
            output = self._output
            truncated = self._output_truncated
            self._output = ""
            self._output_truncated = False

        if truncated:
            output = "[buffer truncated before this output]\n" + output
        return clip_text(output)

    def status(self) -> dict[str, Any]:
        process = self.process
        return {
            "ok": True,
            "running": self.is_running(),
            "pid": process.pid if process is not None else None,
            "returncode": process.poll() if process is not None else None,
            "buffered_output_chars": self.buffered_output_chars(),
        }

    def start(self, wait_seconds: float = 3.0) -> dict[str, Any]:
        with self._process_lock:
            if self.is_running():
                time.sleep(wait_seconds)
                return {
                    "ok": True,
                    "running": True,
                    "already_running": True,
                    "pid": self.process.pid if self.process is not None else None,
                    "output": self.drain_output(),
                }

            command = resolve_msfconsole_command()
            if command is None:
                return {
                    "ok": False,
                    "running": False,
                    "error": "msfconsole was not found on PATH.",
                    "hint": "Install or ensure msfconsole is on PATH.",
                }

            self._cleanup_handles()
            self._output = ""
            self._output_truncated = False
            try:
                if os.name == "posix":
                    self._start_with_pty(command)
                else:
                    self._start_with_pipes(command)
            except OSError as exc:
                self._cleanup_handles()
                return {
                    "ok": False,
                    "running": False,
                    "error": str(exc),
                    "command": command,
                }

        time.sleep(wait_seconds)
        return {
            "ok": True,
            "running": self.is_running(),
            "pid": self.process.pid if self.process is not None else None,
            "command": command,
            "output": self.drain_output(),
        }

    def _start_with_pty(self, command: list[str]) -> None:
        import pty

        master_fd, slave_fd = pty.openpty()
        self._disable_pty_echo(slave_fd)
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
        try:
            self.process = subprocess.Popen(
                command,
                cwd=PROJECT_ROOT,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
        finally:
            os.close(slave_fd)

        self.master_fd = master_fd
        self.reader_thread = threading.Thread(
            target=self._read_pty_output,
            name="msfconsole-output-reader",
            daemon=True,
        )
        self.reader_thread.start()

    def _disable_pty_echo(self, slave_fd: int) -> None:
        try:
            import termios

            attrs = termios.tcgetattr(slave_fd)
            attrs[3] &= ~termios.ECHO
            if hasattr(termios, "ECHOCTL"):
                attrs[3] &= ~termios.ECHOCTL
            termios.tcsetattr(slave_fd, termios.TCSANOW, attrs)
        except (ImportError, OSError):
            pass

    def _start_with_pipes(self, command: list[str]) -> None:
        self.process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self.reader_thread = threading.Thread(
            target=self._read_pipe_output,
            name="msfconsole-output-reader",
            daemon=True,
        )
        self.reader_thread.start()

    def _read_pty_output(self) -> None:
        while True:
            master_fd = self.master_fd
            if master_fd is None:
                return
            try:
                chunk = os.read(master_fd, 4096)
            except OSError:
                return
            if not chunk:
                return
            self.append_output(chunk)

    def _read_pipe_output(self) -> None:
        process = self.process
        if process is None or process.stdout is None:
            return

        while True:
            try:
                chunk = process.stdout.read(1)
            except OSError:
                return
            if not chunk:
                return
            self.append_output(chunk)

    def send(self, command: str, wait_seconds: float = 3.0) -> dict[str, Any]:
        command = command.rstrip("\n")
        if not command.strip():
            return {"ok": False, "running": self.is_running(), "error": "command is empty"}

        started = False
        startup_output = ""
        if not self.is_running():
            start_result = self.start(wait_seconds=wait_seconds)
            if not start_result.get("ok"):
                return start_result
            started = True
            startup_output = str(start_result.get("output") or "")

        prior_output = self.drain_output()
        try:
            self._write_input(f"{command}\n")
        except OSError as exc:
            return {
                "ok": False,
                "running": self.is_running(),
                "error": str(exc),
                "command": command,
            }

        time.sleep(wait_seconds)
        output = strip_command_echo(self.drain_output(), command)
        return {
            "ok": True,
            "running": self.is_running(),
            "started": started,
            "command": command,
            "startup_output": startup_output,
            "prior_output": prior_output,
            "output": output,
        }

    def read(self, wait_seconds: float = 3.0) -> dict[str, Any]:
        time.sleep(wait_seconds)
        return {
            "ok": True,
            "running": self.is_running(),
            "output": self.drain_output(),
        }

    def stop(self, wait_seconds: float = 3.0) -> dict[str, Any]:
        with self._process_lock:
            process = self.process
            if process is None:
                return {"ok": True, "running": False, "output": self.drain_output()}

            if process.poll() is None:
                try:
                    self._write_input("exit -y\n")
                    process.wait(timeout=max(wait_seconds, 1.0))
                except (OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        try:
                            process.wait(timeout=3)
                        except subprocess.TimeoutExpired:
                            pass

            output = self.drain_output()
            self._cleanup_handles()
            return {"ok": True, "running": False, "output": output}

    def _write_input(self, text: str) -> None:
        data = text.encode("utf-8", errors="replace")
        if self.master_fd is not None:
            os.write(self.master_fd, data)
            return

        process = self.process
        if process is None or process.stdin is None:
            raise OSError("msfconsole session is not writable")
        process.stdin.write(data)
        process.stdin.flush()

    def _cleanup_handles(self) -> None:
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        self.process = None
        self.reader_thread = None


_SESSION = MsfConsoleSession()


def execute_msfconsole(params: Any) -> str:
    try:
        data = load_tool_params(params)
        action = str(data.get("action") or ("send" if "command" in data else "status"))
        action = action.strip().lower()
        wait_seconds = bounded_wait_seconds(data.get("wait_seconds"), default=3.0)

        if action == "start":
            result = _SESSION.start(wait_seconds=wait_seconds)
        elif action == "send":
            result = _SESSION.send(str(data.get("command") or ""), wait_seconds=wait_seconds)
        elif action == "read":
            result = _SESSION.read(wait_seconds=wait_seconds)
        elif action == "status":
            result = _SESSION.status()
        elif action == "stop":
            result = _SESSION.stop(wait_seconds=wait_seconds)
        else:
            result = {"ok": False, "error": f"Unknown msfconsole action: {action}"}
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}

    return json.dumps(result, ensure_ascii=False, indent=2)


atexit.register(lambda: _SESSION.stop(wait_seconds=1.0))
