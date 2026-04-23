from __future__ import annotations

import sys
from typing import Any

try:
    from qwen_agent.agents import Assistant
    from qwen_agent.utils.output_beautify import typewriter_print
except ImportError as exc:  # pragma: no cover - import guard for local setup
    raise SystemExit(
        "qwen-agent is not installed. Install it with: pip install -U qwen-agent"
    ) from exc


base_url = "http://127.0.0.1:8000"
model_name = "Qwen3.5-9B-local"
api_key = "EMPTY"

system_prompt = "You are a helpful local Qwen agent."
tools: list[Any] = []


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
