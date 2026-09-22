"""Local browser interface; loads only complete fine-tuned checkpoints."""
import gc
import json
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from openjev.protocol import MODEL_ID, parse_request, strict_loads

ROOT = Path(__file__).resolve().parents[1]
APP_ID = "openjev-local-workbench-v1"
logger = logging.getLogger(__name__)


def available_checkpoints(output):
    found = []
    for path in Path(output).glob("checkpoint-*"):
        try:
            step = int(path.name.removeprefix("checkpoint-"))
        except ValueError:
            continue
        required = ["COMPLETE", "decision_config.json", "decision_head.safetensors",
                    "backbone/config.json", "tokenizer/tokenizer_config.json", "tokenizer/tokenizer.json"]
        if all((path / name).is_file() for name in required) and any((path / "backbone").glob("*.safetensors")):
            found.append((step, path.resolve()))
    return [path for _, path in sorted(found)]


def training_status(output):
    output = Path(output)
    result = {"active": False, "step": 0, "planned_steps": None}
    try:
        config = json.loads((output / "run-config.json").read_text(encoding="utf-8"))
        manifest = json.loads((ROOT / "data/prepared/manifest.json").read_text(encoding="utf-8"))
        count = manifest["splits"]["train"]["questions"]
        accum = config["gradient_accumulation_steps"]
        result["planned_steps"] = config["epochs"] * ((count + accum - 1) // accum)
        if config.get("max_steps"):
            result["planned_steps"] = min(result["planned_steps"], config["max_steps"])
    except (OSError, ValueError, KeyError):
        pass
    try:
        # Read only the tail, even after long training runs.
        with (output / "metrics.jsonl").open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8192))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                row = json.loads(line)
                if "step" in row and "loss" in row:
                    result.update(step=row["step"], loss=row["loss"])
                    break
            except ValueError:
                continue
    except OSError:
        pass
    try:
        import psutil
        launch = json.loads((output / "launch.json").read_text(encoding="utf-8-sig"))
        pids = [launch.get("launcher_pid")] + [p["pid"] for p in launch.get("processes_observed", [])]
        for pid in pids:
            if not pid:
                continue
            try:
                process = psutil.Process(pid)
                if "scripts.train" in process.cmdline() and Path(process.cwd()).resolve() == ROOT:
                    result["active"] = True
                    break
            except (psutil.Error, OSError):
                continue
    except (ImportError, OSError, ValueError, KeyError):
        pass
    return result


class ModelUnavailable(Exception):
    pass


class ModelManager:
    def __init__(self, output=None, engine_factory=None, device_resolver=None):
        self.output = Path(output or ROOT / "outputs/decision-full")
        self.engine_factory = engine_factory
        self.device_resolver = device_resolver
        self.engine = None
        self._state_lock = threading.Lock()
        self._operation = threading.Lock()
        self._stop = threading.Event()
        self._state = {"status": "waiting", "message": "等待第一个完整微调断点，保存后将自动加载。",
                       "checkpoint": None, "device": None, "max_length": None}

    def snapshot(self):
        with self._state_lock:
            result = dict(self._state)
        checkpoints = available_checkpoints(self.output)
        result.update(model=MODEL_ID, latest_checkpoint=checkpoints[-1].name if checkpoints else None,
                      training=training_status(self.output))
        return result

    def _update(self, **values):
        with self._state_lock:
            self._state.update(values)

    def start(self):
        def watch():
            while not self._stop.is_set():
                if available_checkpoints(self.output):
                    self.request_load("auto")
                    return
                self._stop.wait(5)
        threading.Thread(target=watch, name="checkpoint-watcher", daemon=True).start()

    def stop(self):
        self._stop.set()

    def request_load(self, device="auto"):
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("请选择自动、显卡或CPU。")
        checkpoints = available_checkpoints(self.output)
        if not checkpoints:
            raise ModelUnavailable("尚无完整微调断点，训练保存后会自动加载。")
        checkpoint = checkpoints[-1]
        with self._state_lock:
            if self._state["status"] == "loading":
                return
            self._state.update(status="loading", message=f"正在加载 {checkpoint.name}，请稍候。")
        threading.Thread(target=self._load, args=(checkpoint, device), daemon=True, name="model-loader").start()

    def _choose_device(self, preference):
        if self.device_resolver:
            return self.device_resolver(preference)
        import torch
        if preference == "cpu":
            return "cpu"
        if preference == "cuda":
            if not torch.cuda.is_available():
                raise ValueError("未检测到可用显卡，请选择CPU。")
            if not torch.cuda.is_bf16_supported():
                raise ValueError("当前显卡不支持模型所需的 BF16 运算，请选择CPU。")
            return "cuda"
        if training_status(self.output)["active"]:
            return "cpu"
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                and torch.cuda.mem_get_info()[0] >= 6 * 1024 ** 3):
            return "cuda"
        return "cpu"

    def _load(self, checkpoint, preference):
        with self._operation:
            try:
                device = self._choose_device(preference)
                self.engine = None
                gc.collect()
                if self.engine_factory is None:
                    import torch
                    from openjev.inference import Engine
                    torch.set_num_threads(min(8, max(1, (os.cpu_count() or 4) // 2)))
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    engine = Engine(checkpoint, device=device)
                else:
                    engine = self.engine_factory(checkpoint, device=device)
                self.engine = engine
                self._update(status="ready", message="模型已就绪，可以开始判断。", checkpoint=checkpoint.name,
                             device=device, max_length=engine.max_length)
            except Exception as exc:
                logger.exception("Model load failed")
                self.engine = None
                self._update(status="error", message=f"模型加载失败：{exc}", checkpoint=None, device=None)

    def predict(self, value):
        parse_request(value)
        if not self._operation.acquire(blocking=False):
            raise ModelUnavailable("模型正在加载或处理其他请求，请稍后重试。")
        try:
            if self.engine is None:
                raise ModelUnavailable("模型尚未就绪，请等待加载完成。")
            return self.engine.predict(value)
        finally:
            self._operation.release()


def create_web_app(manager, token, shutdown=None):
    @asynccontextmanager
    async def lifespan(app):
        manager.start()
        yield
        manager.stop()

    app = FastAPI(title="OpenJev 决策工作台", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    assets = ROOT / "web"
    app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.middleware("http")
    async def local_session(request, call_next):
        if request.method == "POST" and not secrets.compare_digest(request.headers.get("x-openjev-token", ""), token):
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail": "页面会话已失效，请刷新后重试。"}, status_code=403)
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = "default-src 'self'; style-src 'self'; script-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'"
        return response

    @app.get("/")
    def index():
        return FileResponse(assets / "index.html")

    @app.get("/health")
    def health():
        return {"status": "ok", "app": APP_ID, "workspace": str(ROOT)}

    @app.get("/api/session")
    def session():
        return {"token": token}

    @app.get("/api/status")
    def status():
        return manager.snapshot()

    async def read_body(request):
        raw = await request.body()
        if len(raw) > 1024 * 1024:
            raise HTTPException(413, "请求过大，请缩短输入。")
        try:
            return strict_loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/v1/systemone")
    async def predict(request: Request):
        value = await read_body(request)
        try:
            return await run_in_threadpool(manager.predict, value)
        except ModelUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except (ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except Exception as exc:
            logger.exception("Inference failed")
            message = "显存不足，请在页面选择CPU并重新加载。" if "out of memory" in str(exc).lower() else f"判断失败：{exc}"
            raise HTTPException(500, message) from exc

    @app.post("/api/reload")
    async def reload(request: Request):
        value = await read_body(request)
        try:
            if not isinstance(value, dict):
                raise ValueError("加载设置应为对象。")
            manager.request_load(value.get("device", "auto"))
            return {"accepted": True}
        except ModelUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post("/api/shutdown")
    def close():
        if shutdown:
            shutdown()
        return {"status": "stopping"}

    return app
