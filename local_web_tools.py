from __future__ import annotations

import html
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from typing import Any


WEB_SEARCH_TOOL_NAME = "web_search"
OPEN_URL_TOOL_NAME = "open_url"
REQUEST_TIMEOUT_SECONDS = 20
MAX_FETCH_BYTES = 2_000_000
MAX_TEXT_CHARS = 60000
MAX_LINKS = 40
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

WEB_SEARCH_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": WEB_SEARCH_TOOL_NAME,
        "description": (
            "Search DuckDuckGo over the public internet. This is disabled unless "
            "QWEN_ENABLE_WEB_TOOLS=1 is set before starting the agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum results to return. Defaults to 5, capped at 10.",
                },
            },
            "required": ["query"],
        },
    },
}

OPEN_URL_OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": OPEN_URL_TOOL_NAME,
        "description": (
            "Open an http(s) URL and return parsed text plus links when the "
            "response is HTML. This is disabled unless QWEN_ENABLE_WEB_TOOLS=1 "
            "is set before starting the agent."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "HTTP or HTTPS URL to open.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Maximum text characters to return. Defaults to 60000.",
                },
            },
            "required": ["url"],
        },
    },
}

OPENAI_TOOLS = [WEB_SEARCH_OPENAI_TOOL, OPEN_URL_OPENAI_TOOL]


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


def json_result(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False, indent=2)


def bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        number = default
    return max(minimum, min(number, maximum))


def normalize_whitespace(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    blocks: list[str] = []
    previous_blank = False
    for line in lines:
        if not line:
            if not previous_blank:
                blocks.append("")
            previous_blank = True
            continue
        blocks.append(line)
        previous_blank = False
    return "\n".join(blocks).strip()


def clip_text(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n...[truncated to {limit} characters]"


def make_request(url: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
    )


def fetch_url(url: str) -> tuple[str, str, bytes]:
    request = make_request(url)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        final_url = response.geturl()
        content_type = response.headers.get("Content-Type", "")
        body = response.read(MAX_FETCH_BYTES + 1)

    if len(body) > MAX_FETCH_BYTES:
        body = body[:MAX_FETCH_BYTES]
    return final_url, content_type, body


def decode_body(body: bytes, content_type: str) -> str:
    charset_match = re.search(r"charset=([^;\s]+)", content_type, flags=re.I)
    charset = charset_match.group(1).strip("\"'") if charset_match else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def clean_duckduckgo_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.path == "/l/":
        query = urllib.parse.parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        if uddg:
            return urllib.parse.unquote(uddg)
    return url


class DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict[str, str]] = []
        self._in_result_link = False
        self._current_href = ""
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {key: value or "" for key, value in attrs}
        classes = set(attr_map.get("class", "").split())
        if tag == "a" and "result__a" in classes:
            self._in_result_link = True
            self._current_href = attr_map.get("href", "")
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._current_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._in_result_link:
            return

        title = normalize_whitespace("".join(self._current_text))
        url = clean_duckduckgo_url(html.unescape(self._current_href))
        if title and url:
            self.results.append({"title": title, "url": url})

        self._in_result_link = False
        self._current_href = ""
        self._current_text = []


class HtmlTextParser(HTMLParser):
    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title = ""
        self.text_parts: list[str] = []
        self.links: list[dict[str, str]] = []
        self._skip_depth = 0
        self._in_title = False
        self._current_link: dict[str, str] | None = None
        self._current_link_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
            return

        attr_map = {key: value or "" for key, value in attrs}
        if tag == "title":
            self._in_title = True
        elif tag == "a" and attr_map.get("href"):
            self._current_link = {
                "url": urllib.parse.urljoin(self.base_url, attr_map["href"]),
                "text": "",
            }
            self._current_link_text = []
        elif tag in {"p", "div", "section", "article", "header", "footer", "li", "br"}:
            self.text_parts.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title:
            self.title += data
            return
        self.text_parts.append(data)
        if self._current_link is not None:
            self._current_link_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
            return
        if tag == "title":
            self._in_title = False
        elif tag == "a" and self._current_link is not None:
            link_text = normalize_whitespace("".join(self._current_link_text))
            if link_text and len(self.links) < MAX_LINKS:
                self._current_link["text"] = link_text
                self.links.append(self._current_link)
            self._current_link = None
            self._current_link_text = []
        elif tag in {"p", "div", "section", "article", "li"}:
            self.text_parts.append("\n")
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text_parts.append("\n\n")

    def parsed(self) -> dict[str, Any]:
        return {
            "title": normalize_whitespace(self.title),
            "text": normalize_whitespace("".join(self.text_parts)),
            "links": self.links,
        }


def validate_http_url(url: str) -> str:
    cleaned = str(url or "").strip()
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("url must be an absolute http(s) URL")
    return cleaned


def execute_web_search(params: Any) -> str:
    try:
        data = load_tool_params(params)
        query = str(data.get("query") or "").strip()
        if not query:
            raise ValueError("query must not be empty")
        max_results = bounded_int(data.get("max_results"), 5, 1, 10)
        search_url = "https://duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
        final_url, content_type, body = fetch_url(search_url)
        text = decode_body(body, content_type)
        parser = DuckDuckGoParser()
        parser.feed(text)
        result = {
            "ok": True,
            "query": query,
            "url": final_url,
            "results": parser.results[:max_results],
        }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    return json_result(result)


def execute_open_url(params: Any) -> str:
    try:
        data = load_tool_params(params)
        url = validate_http_url(str(data.get("url") or ""))
        max_chars = bounded_int(data.get("max_chars"), MAX_TEXT_CHARS, 1000, 200000)
        final_url, content_type, body = fetch_url(url)
        decoded = decode_body(body, content_type)
        if "html" in content_type.lower() or decoded.lstrip().startswith(("<!DOCTYPE", "<html", "<HTML")):
            parser = HtmlTextParser(final_url)
            parser.feed(decoded)
            parsed = parser.parsed()
            text = clip_text(parsed["text"], max_chars)
            result = {
                "ok": True,
                "url": final_url,
                "content_type": content_type,
                "title": parsed["title"],
                "text": text,
                "links": parsed["links"],
            }
        else:
            result = {
                "ok": True,
                "url": final_url,
                "content_type": content_type,
                "text": clip_text(normalize_whitespace(decoded), max_chars),
                "links": [],
            }
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, ValueError) as exc:
        result = {"ok": False, "error": str(exc)}
    return json_result(result)
