"""Native Anthropic path with Claude Haiku 5.5-shaped responses (real SDK, fake server)."""

import itertools
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import model_form
from translator import Translator

TEXT = "你好世界"


def _message(body, text):
    return {
        "id": "msg_1", "type": "message", "role": "assistant",
        "model": body["model"], "stop_reason": "end_turn", "stop_sequence": None,
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "text", "text": text},
        ],
        "usage": {"input_tokens": 10, "output_tokens": 7},
    }


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append((self.path, body))
        thinking = body.get("thinking") or {}
        if thinking.get("type") in ("between_tools", "enabled"):
            return self._json(400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": "bad thinking"}})
        for k in ("temperature", "top_p", "top_k"):
            if k in body:
                return self._json(400, {"type": "error", "error": {
                    "type": "invalid_request_error", "message": f"{k} not allowed"}})
        if body["messages"][-1]["role"] != "user":
            return self._json(400, {"type": "error", "error": {
                "type": "invalid_request_error", "message": "prefill"}})
        text = TEXT
        fmt = (body.get("output_config") or {}).get("format")
        if fmt:
            text = json.dumps({"t": TEXT}, ensure_ascii=False)
        if not body.get("stream"):
            return self._json(200, _message(body, text))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def ev(name, d):
            self.wfile.write(
                f"event: {name}\ndata: {json.dumps({'type': name, **d}, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        msg = _message(body, "")
        msg["content"] = []
        msg["usage"] = {"input_tokens": 10, "output_tokens": 1}
        ev("message_start", {"message": msg})
        ev("content_block_start", {"index": 0, "content_block": {
            "type": "thinking", "thinking": "", "signature": ""}})
        ev("content_block_delta", {"index": 0, "delta": {
            "type": "signature_delta", "signature": "sig"}})
        ev("content_block_stop", {"index": 0})
        ev("content_block_start", {"index": 1, "content_block": {"type": "text", "text": ""}})
        for i in range(0, len(text), 2):
            ev("content_block_delta", {"index": 1, "delta": {
                "type": "text_delta", "text": text[i:i + 2]}})
        ev("content_block_stop", {"index": 1})
        ev("message_delta", {"delta": {"stop_reason": "end_turn", "stop_sequence": None},
                             "usage": {"output_tokens": 7}})
        ev("message_stop", {})

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
    srv.bodies = []
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()


def _translator(srv, streaming, json_response, model="claude-haiku-5-5"):
    md = model_form.build_model_dict(dict(
        name="m", api_base=f"http://127.0.0.1:{srv.server_port}/anthropic", api_key="k",
        model=model, proxy="none", api_format="anthropic",
        streaming=streaming, json_response=json_response))
    return Translator(
        api_base=md["api_base"], api_key=md["api_key"], model=md["model"],
        streaming=md.get("streaming", True), proxy=md.get("proxy", "none"),
        thinking_style=md.get("thinking_style"),
        json_response=md.get("json_response", False),
        overrides=md.get("overrides"), extra_body=md.get("extra_body"),
        api_format=md.get("api_format", "auto"),
        anthropic_fallbacks=md.get("anthropic_fallbacks", True),
    )


@pytest.mark.parametrize("streaming,json_response",
                         list(itertools.product((True, False), (True, False))))
def test_haiku_all_combos(server, streaming, json_response):
    tr = _translator(server, streaming, json_response)
    assert tr.translate("hello", "en") == TEXT
    parts = list(tr.translate_iter("hello", "en"))
    assert parts[-1] == TEXT
    if streaming and not json_response:
        assert len(parts) > 2
    else:
        assert parts == [TEXT]
    assert tr.last_usage == (10, 7)
    for path, body in server.bodies:
        assert path.endswith("/v1/messages")
        assert body["output_config"]["effort"] == "low"
        assert "thinking" not in body and "betas" not in body
        assert body["system"]


def test_empty_text_is_explicit_error(server, monkeypatch):
    import translator_anthropic as tla
    monkeypatch.setattr(tla, "extract_text", lambda c: "")
    tr = _translator(server, False, False)
    with pytest.raises(tla.TranslationAPIError, match="no text"):
        tr.translate("hello", "en")
