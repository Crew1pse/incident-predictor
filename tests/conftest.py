import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from training.train import train


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("model")
    train(minutes=30_000, out_dir=str(d))
    return str(d)


class FakeServer:
    """Локальный HTTP-сервер: имитирует Prometheus и Telegram, запоминает запросы."""
    def __init__(self, responder):
        self.requests = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def _handle(self):
                n = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(n).decode() if n else ""
                outer.requests.append((self.command, self.path, body))
                code, payload = responder(self.command, self.path, body)
                data = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = _handle

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def close(self):
        self.srv.shutdown()


@pytest.fixture
def fake_server():
    servers = []

    def make(responder):
        s = FakeServer(responder)
        servers.append(s)
        return s
    yield make
    for s in servers:
        s.close()
