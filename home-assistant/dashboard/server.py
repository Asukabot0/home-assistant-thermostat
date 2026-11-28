"""Pure-stdlib read-only HTTP & Server-Sent Events server for the controller dashboard."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import signal
import threading
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

from .collector import DashboardPaths, collect_snapshot
from .demo import SCENARIO_IDS, build_demo_snapshot

LOG = logging.getLogger("thermostat.dashboard")
STATIC_DIR = Path(__file__).resolve().parent / "static"

STATIC_FILES: dict[str, tuple[str, str]] = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/assets/styles.css": ("styles.css", "text/css; charset=utf-8"),
    "/assets/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/assets/favicon.svg": ("favicon.svg", "image/svg+xml"),
}

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "SAMEORIGIN",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline'; "
        "script-src 'self'; "
        "img-src 'self' data:; "
        "connect-src 'self'"
    ),
}


@dataclass(frozen=True)
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8780
    paths: DashboardPaths | None = None
    demo: bool = False
    default_scenario: str = "recover_strengthen"
    stream_interval_s: float = 2.0


class DashboardHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], config: ServerConfig):
        self.config = config
        self.shutdown_event = threading.Event()
        super().__init__(server_address, DashboardRequestHandler)

    def get_snapshot(self, scenario: str | None = None) -> dict[str, Any]:
        if self.config.demo:
            chosen = scenario if scenario in SCENARIO_IDS else self.config.default_scenario
            cfg_dir = self.config.paths.shadow_config_dir if self.config.paths else None
            return build_demo_snapshot(chosen, config_dir=cfg_dir)
        paths = self.config.paths or DashboardPaths.from_env()
        return collect_snapshot(paths)


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer  # type: ignore[assignment]
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOG.info("%s - %s", self.address_string(), format % args)

    def _send_security_headers(self) -> None:
        for key, value in SECURITY_HEADERS.items():
            self.send_header(key, value)

    def _send_bytes(
        self,
        body: bytes,
        content_type: str,
        status: HTTPStatus = HTTPStatus.OK,
        *,
        cache_control: str = "no-store",
        head_only: bool = False,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache_control)
        self._send_security_headers()
        self.end_headers()
        if not head_only:
            self.wfile.write(body)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK, *, head_only: bool = False) -> None:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self._send_bytes(body, "application/json; charset=utf-8", status, head_only=head_only)

    def _reject_mutation(self) -> None:
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        body = b'{"error":"read_only_dashboard"}'
        self.send_header("Content-Length", str(len(body)))
        self._send_security_headers()
        self.end_headers()
        self.wfile.write(body)

    do_POST = _reject_mutation
    do_PUT = _reject_mutation
    do_PATCH = _reject_mutation
    do_DELETE = _reject_mutation

    def do_HEAD(self) -> None:
        self._dispatch(head_only=True)

    def do_GET(self) -> None:
        self._dispatch(head_only=False)

    def _dispatch(self, *, head_only: bool) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        params = parse_qs(parsed.query)
        scenario = params.get("scenario", [None])[0]

        if route in STATIC_FILES:
            filename, content_type = STATIC_FILES[route]
            target = (STATIC_DIR / filename).resolve()
            if not target.is_file() or target.parent != STATIC_DIR.resolve():
                self._send_json({"error": "asset_not_found"}, HTTPStatus.NOT_FOUND, head_only=head_only)
                return
            body = target.read_bytes()
            cache = "no-cache" if filename == "index.html" else "public, max-age=60"
            self._send_bytes(body, content_type, cache_control=cache, head_only=head_only)
            return

        if route == "/api/snapshot":
            try:
                snapshot = self.server.get_snapshot(scenario)
            except Exception as exc:
                LOG.warning("Snapshot error: %s", type(exc).__name__)
                self._send_json({"error": "snapshot_unavailable"}, HTTPStatus.INTERNAL_SERVER_ERROR, head_only=head_only)
                return
            self._send_json(snapshot, head_only=head_only)
            return

        if route == "/api/health":
            try:
                snapshot = self.server.get_snapshot(scenario)
            except Exception as exc:
                LOG.warning("Health check error: %s", type(exc).__name__)
                self._send_json({"ok": False, "error": type(exc).__name__}, HTTPStatus.INTERNAL_SERVER_ERROR, head_only=head_only)
                return
            self._send_json(
                {
                    "ok": True,
                    "mode": snapshot["mode"],
                    "controller_health": snapshot["controller"]["health"],
                    "data_present": snapshot["controller"]["data_present"],
                    "generated_at": snapshot["generated_at"],
                },
                head_only=head_only,
            )
            return

        if route == "/api/stream":
            if head_only:
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self._send_security_headers()
                self.end_headers()
                return
            self._serve_sse(scenario)
            return

        self._send_json({"error": "not_found"}, HTTPStatus.NOT_FOUND, head_only=head_only)

    def _serve_sse(self, scenario: str | None) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self._send_security_headers()
        self.end_headers()

        interval = max(0.5, float(self.server.config.stream_interval_s))
        try:
            while not self.server.shutdown_event.is_set():
                snapshot = self.server.get_snapshot(scenario)
                payload = json.dumps(snapshot, ensure_ascii=False, allow_nan=False)
                chunk = f"event: snapshot\ndata: {payload}\n\n".encode("utf-8")
                self.wfile.write(chunk)
                self.wfile.flush()
                if self.server.shutdown_event.wait(interval):
                    break
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            return


def create_server(config: ServerConfig) -> DashboardHTTPServer:
    return DashboardHTTPServer((config.host, config.port), config)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = DashboardPaths.from_env()
    parser = argparse.ArgumentParser(
        prog="python3 -m dashboard",
        description="Read-only web dashboard for the Home Assistant three-room thermostat controller.",
    )
    parser.add_argument("--host", default=os.getenv("THERMOSTAT_DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("THERMOSTAT_DASHBOARD_PORT", "8780")))
    parser.add_argument(
        "--demo",
        action="store_true",
        default=os.getenv("THERMOSTAT_DASHBOARD_DEMO", "").lower() in {"1", "true", "yes"},
        help="Serve deterministic demo scenarios (explicit opt-in only).",
    )
    parser.add_argument("--scenario", default="recover_strengthen", help="Initial scenario when --demo is enabled.")
    parser.add_argument("--shadow-config-dir", type=Path, default=defaults.shadow_config_dir)
    parser.add_argument("--pilot-dir", type=Path, default=defaults.pilot_data_dir)
    parser.add_argument("--shadow-data-dir", type=Path, default=defaults.shadow_data_dir)
    parser.add_argument("--shadow-output-dir", type=Path, default=defaults.shadow_output_dir)
    parser.add_argument("--neural-output-dir", type=Path, default=defaults.neural_output_dir)
    parser.add_argument("--stream-interval", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    paths = DashboardPaths(
        shadow_config_dir=args.shadow_config_dir,
        pilot_data_dir=args.pilot_dir,
        shadow_data_dir=args.shadow_data_dir,
        shadow_output_dir=args.shadow_output_dir,
        neural_output_dir=args.neural_output_dir,
    )
    config = ServerConfig(
        host=args.host,
        port=args.port,
        paths=paths,
        demo=args.demo,
        default_scenario=args.scenario,
        stream_interval_s=args.stream_interval,
    )
    httpd = create_server(config)

    def _stop(*_args: Any) -> None:
        httpd.shutdown_event.set()
        threading.Thread(target=httpd.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    mode_tag = f"DEMO ({args.scenario})" if config.demo else "LIVE (read-only)"
    LOG.info("Thermostat dashboard listening on http://%s:%d [%s]", config.host, config.port, mode_tag)
    try:
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
