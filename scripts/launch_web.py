"""Double-click launcher: one local server per workspace and automatic browser."""
import argparse
import json
import logging
import os
import secrets
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/webapp"
APP_ID = "openjev-local-workbench-v1"


def request_json(url, *, token=None, body=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-OpenJev-Token"] = token
    request = urllib.request.Request(url, data=json.dumps(body).encode() if body is not None else None, headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=3) as response:
        return json.load(response)


def existing_server():
    try:
        info = json.loads((RUN / "server.json").read_text(encoding="utf-8"))
        # The port is local; never follow a URL supplied by the record.
        port = int(info["port"])
        if not 1 <= port <= 65535:
            return None
        url = f"http://127.0.0.1:{port}"
        health = request_json(url + "/health")
        if health.get("app") == APP_ID and health.get("workspace") == str(ROOT):
            return {**info, "url": url}
    except (OSError, ValueError, KeyError):
        pass
    return None


def show_error(message):
    logging.error(message)
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, message, "OpenJev 启动提示", 0x10)
    elif sys.stderr:
        print(message, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--stop", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    RUN.mkdir(parents=True, exist_ok=True)
    # pythonw has no standard streams. Capture errors from dependencies as well.
    stream = (RUN / "server.log").open("a", encoding="utf-8", buffering=1)
    sys.stdout = stream
    sys.stderr = stream
    logging.basicConfig(stream=stream, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTHONUTF8", "1")
    try:
        from filelock import FileLock, Timeout
        lock = FileLock(RUN / "instance.lock")
        if args.stop:
            info = existing_server()
            if info:
                request_json(info["url"] + "/api/shutdown", token=info["token"], body={})
            return
        try:
            lock.acquire(timeout=0)
        except Timeout:
            # Another double click can happen while the first server is starting.
            for _ in range(60):
                info = existing_server()
                if info:
                    if not args.no_browser:
                        webbrowser.open(info["url"])
                    return
                time.sleep(.5)
            show_error("服务仍在启动，请稍后重试。日志位于 outputs\\webapp\\server.log。")
            return
        try:
            import uvicorn
            from openjev.webapp import ModelManager, create_web_app
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Reserve the socket before announcing the URL, eliminating port races.
            try:
                listener.bind(("127.0.0.1", 8765))
            except OSError:
                listener.bind(("127.0.0.1", 0))
            listener.listen(128)
            port = listener.getsockname()[1]
            token = secrets.token_urlsafe(32)
            info = {"pid": os.getpid(), "port": port, "token": token, "started_at": time.time()}
            (RUN / "server.json").write_text(json.dumps(info), encoding="utf-8")
            manager = ModelManager()
            server = None

            def shutdown():
                manager.stop()
                server.should_exit = True

            app = create_web_app(manager, token, shutdown)
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_config=None, access_log=False))

            def open_when_ready():
                for _ in range(60):
                    if server.should_exit:
                        return
                    if server.started:
                        webbrowser.open(f"http://127.0.0.1:{port}")
                        return
                    time.sleep(.25)

            if not args.no_browser:
                threading.Thread(target=open_when_ready, daemon=True).start()
            logging.info("OpenJev listening on http://127.0.0.1:%s", port)
            try:
                server.run(sockets=[listener])
            finally:
                manager.stop()
                listener.close()
                (RUN / "server.json").unlink(missing_ok=True)
        finally:
            lock.release()
    except Exception:
        logging.exception("Launcher failed")
        show_error("启动失败。请查看 outputs\\webapp\\server.log 中的错误信息。")


if __name__ == "__main__":
    main()
