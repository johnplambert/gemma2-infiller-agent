import pytest
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class MockHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _ = self.rfile.read(length) if length > 0 else b""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"url": "http://127.0.0.1/post", "files": {"file": "named temp file contents\\n"}}')

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"url": "http://127.0.0.1"}')

    def log_message(self, format, *args):
        pass

class ServerHolder:
    def __init__(self, port):
        self.port = port
        self.url = f"http://127.0.0.1:{port}"

    def __call__(self, suffix=""):
        suffix = suffix.lstrip("/")
        return f"{self.url}/{suffix}" if suffix else self.url

    def joinurl(self, suffix=""):
        return self.__call__(suffix)

@pytest.fixture(scope="session")
def httpbin():
    server = HTTPServer(("127.0.0.1", 0), MockHandler)
    port = server.server_port
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    yield ServerHolder(port)
    server.shutdown()

@pytest.fixture(scope="session")
def httpbin_secure(httpbin):
    return httpbin

