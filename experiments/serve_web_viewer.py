#!/usr/bin/env python3
"""Serve the hosted web viewer locally with small write endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_WEB_ROOT = Path.home() / "projects" / "evasion-paths-experiments"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from web_export import publish_single_sim_bundle, rebuild_catalog_from_manifests


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the hosted web viewer with local write endpoints.")
    parser.add_argument("--web-root", type=str, default=str(DEFAULT_WEB_ROOT))
    parser.add_argument("--info-file", type=str, default="", help="Optional JSON file to write server URL metadata.")
    return parser


def _json_response(handler: SimpleHTTPRequestHandler, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def main() -> None:
    args = _make_parser().parse_args()
    web_root = Path(args.web_root).resolve()
    if not web_root.exists():
        raise SystemExit(f"Web root does not exist: {web_root}")
    rebuild_catalog_from_manifests(web_root)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(web_root), **handler_kwargs)

        def do_POST(self) -> None:
            req_path = urlsplit(self.path).path
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(content_length).decode("utf-8") or "{}")
            except Exception as exc:  # pragma: no cover - local debug path
                _json_response(self, 400, {"ok": False, "error": f"Invalid JSON payload: {exc}"})
                return

            try:
                if req_path == "/__export_sim":
                    source_dir = Path(str(payload.get("source_dir", ""))).resolve()
                    published = publish_single_sim_bundle(
                        source_dir=source_dir,
                        web_root=web_root,
                        category=str(payload.get("category", "rl")),
                        motion_model=str(payload.get("motion_model", "RL")),
                        display_name=str(payload.get("display_name", source_dir.parent.name or source_dir.name)),
                        experiment_id=(str(payload.get("experiment_id", "")).strip() or None),
                        source_run=str(payload.get("source_run", "")),
                        checkpoint=str(payload.get("checkpoint", "")),
                    )
                    _json_response(
                        self,
                        200,
                        {"ok": True, "published_path": str(published), "web_root": str(web_root)},
                    )
                    return

                _json_response(self, 404, {"ok": False, "error": "Unknown endpoint"})
            except Exception as exc:  # pragma: no cover - local debug path
                _json_response(self, 500, {"ok": False, "error": str(exc)})

        def log_message(self, format: str, *log_args) -> None:
            msg = format % log_args
            if "GET /data/" in msg or "GET /favicon.ico" in msg:
                return
            super().log_message(format, *log_args)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler))
    port = int(httpd.server_address[1])
    url = f"http://127.0.0.1:{port}/index.html"

    if args.info_file:
        info_path = Path(args.info_file).resolve()
        info_path.parent.mkdir(parents=True, exist_ok=True)
        info_path.write_text(json.dumps({"port": port, "url": url}, indent=2), encoding="utf-8")

    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
