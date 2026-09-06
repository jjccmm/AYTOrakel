"""Launch the dependency-free AYTOrakel season editor on localhost."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from threading import Timer
from urllib.parse import urlparse
import webbrowser


ROOT = Path(__file__).resolve().parent
HTML_PATH = ROOT / "ayto_data_viewer.html"
DATA_PATH = ROOT / "ayto_data.json"


class ViewerHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        path = urlparse(self.path).path
        if path in {"/", "/ayto_data_viewer.html"}:
            self._send(200, "text/html; charset=utf-8", HTML_PATH.read_bytes())
        elif path == "/ayto_data.json":
            self._send(200, "application/json; charset=utf-8", DATA_PATH.read_bytes())
        else:
            self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_PUT(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        if urlparse(self.path).path != "/ayto_data.json":
            self._send(404, "text/plain; charset=utf-8", b"Not found")
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 50_000_000:
                raise ValueError("Invalid JSON size")
            raw = self.rfile.read(size)
            data = json.loads(raw.decode("utf-8"))
            if not isinstance(data, dict):
                raise ValueError("The top JSON level must be an object")
            formatted = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
            temporary = DATA_PATH.with_name(f".{DATA_PATH.name}.tmp")
            temporary.write_bytes(formatted)
            os.replace(temporary, DATA_PATH)
            self._send(204, "text/plain; charset=utf-8", b"")
        except (UnicodeDecodeError, json.JSONDecodeError, OSError, ValueError) as error:
            self._send(400, "text/plain; charset=utf-8", str(error).encode("utf-8"))

    def log_message(self, message: str, *args: object) -> None:
        print(message % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the local AYTOrakel data viewer")
    parser.add_argument("--port", type=int, default=0, help="Local port; 0 selects a free port")
    parser.add_argument("--no-browser", action="store_true", help="Do not open the browser automatically")
    args = parser.parse_args()

    server = ThreadingHTTPServer(("127.0.0.1", args.port), ViewerHandler)
    port = server.server_address[1]
    url = f"http://127.0.0.1:{port}/?save=1"
    print(f"AYTOrakel Data Viewer: {url}")
    print("Stop with Ctrl+C")
    if not args.no_browser:
        Timer(0.3, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nViewer stopped")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
