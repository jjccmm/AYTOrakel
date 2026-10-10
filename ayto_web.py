"""Open the static AYTO puzzle locally; production only needs GitHub Pages."""
from __future__ import annotations

import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Timer
import webbrowser


class PuzzleHandler(SimpleHTTPRequestHandler):
    # Windows registers .mjs as plain text; browsers require a JavaScript MIME type.
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript", ".js": "text/javascript"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the interactive AYTO puzzle")
    parser.add_argument('--port', type=int, default=0, help="Local port; 0 chooses a free port")
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    handler = partial(PuzzleHandler, directory=str(root))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    url = f'http://127.0.0.1:{server.server_address[1]}/'
    print(f'AYTO puzzle: {url}', flush=True)
    print('Stop with Ctrl+C.', flush=True)
    if not args.no_browser:
        timer = Timer(.3, webbrowser.open, args=(url,))
        timer.daemon = True
        timer.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
