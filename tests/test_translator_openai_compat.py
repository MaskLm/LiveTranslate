"""OpenAI-compatible path against a local fake server (LM Studio-like)."""

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import model_form
from translator import Translator

KNOWN = {
    "model", "messages", "max_tokens", "temperature", "stream",
    "stream_options", "response_format", "top_p", "seed",
    "frequency_penalty", "presence_penalty",
}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mode = self.server.mode
        if mode.get("strict") and set(body) - KNOWN:
            return self._json(400, {"error": {"message": "unrecognized argument"}})
        rf = body.get("response_format")
        if rf and mode.get("no_schema"):
            return self._json(400, {"error": {"message": "bad response_format"}})
        txt = mode.get("text", "你好世界")
        if rf:
            txt = json.dumps({"t": txt}, ensure_ascii=False)
        if not body.get("stream"):
            return self._json(200, {
                "id": "x", "object": "chat.completion", "created": 1,
                "model": body["model"],
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": txt}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4,
                          "total_tokens": 9},
            })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        base = {"id": "x", "object": "chat.completion.chunk", "created": 1,
                "model": body["model"]}

        def ev(d):
            self.wfile.write(b"data: " + json.dumps(d, ensure_ascii=False).encode() + b"\n\n")
            self.wfile.flush()

        for i in range(0, len(txt), 2):
            ev({**base, "choices": [{"index": 0, "delta": {"content": txt[i:i + 2]},
                                     "finish_reason": None}]})
        ev({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def _json(self, code, d):
        b = json.dumps(d, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


@pytest.fixture
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    srv.mode = {}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _translator(srv, streaming, json_response):
    md = model_form.build_model_dict(dict(
        name="m", api_base=f"http://127.0.0.1:{srv.server_port}/v1", api_key="x",
        model="hunyuan-mt-chimera-7b", proxy="none",
        streaming=streaming, json_response=json_response))
    return Translator(
        api_base=md["api_base"], api_key=md["api_key"], model=md["model"],
        streaming=md.get("streaming", True), proxy=md.get("proxy", "none"),
        no_system_role=md.get("no_system_role", False),
        no_think=md.get("no_think", True),
        thinking_style=md.get("thinking_style"),
        json_response=md.get("json_response", False),
        overrides=md.get("overrides"), extra_body=md.get("extra_body"),
        api_format=md.get("api_format", "auto"),
    )


COMBOS = list(itertools.product((True, False), (True, False)))


@pytest.mark.parametrize("streaming,json_response", COMBOS)
@pytest.mark.parametrize("strict", [False, True])
def test_all_checkbox_combos(server, streaming, json_response, strict):
    server.mode = {"strict": strict}
    tr = _translator(server, streaming, json_response)
    assert tr.translate("hello", "en") == "你好世界"
    parts = list(tr.translate_iter("hello", "en"))
    assert parts[-1] == "你好世界"
    if streaming and not json_response:
        assert len(parts) > 2  # live partials reach the UI
    else:
        assert parts == ["你好世界"]


@pytest.mark.parametrize("streaming", [True, False])
def test_json_schema_rejected_falls_back(server, streaming):
    server.mode = {"no_schema": True}
    tr = _translator(server, streaming, True)
    # server returns plain text once response_format is dropped
    assert tr.translate("hello", "en") == "你好世界"
    assert tr._drop_response_format


@pytest.mark.parametrize("streaming", [True, False])
def test_think_blocks_stripped(server, streaming):
    server.mode = {"text": "<think>hmm hmm</think>你好世界"}
    tr = _translator(server, streaming, False)
    assert tr.translate("hello", "en") == "你好世界"
    assert list(tr.translate_iter("hello", "en"))[-1] == "你好世界"


def test_strip_think_unclosed():
    assert Translator._strip_think("<think>never ends") == ""
    assert Translator._strip_think("a</think>b") == "b"
