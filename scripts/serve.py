import argparse
import threading
from fastapi import FastAPI, HTTPException, Request as HttpRequest
from openjev.protocol import strict_loads


def create_app(engine):
    app = FastAPI(title="OpenJev 本地决策服务", version="1.0")
    lock = threading.Lock()

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "openjev-qwen3.5-0.8b-v1"}

    @app.post("/v1/systemone")
    async def systemone(request: HttpRequest):
        from starlette.concurrency import run_in_threadpool
        try:
            value = strict_loads((await request.body()).decode("utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        def predict():
            # One GPU request at a time; avoids competing allocations on 16 GB.
            with lock:
                return engine.predict(value)
        try:
            return await run_in_threadpool(predict)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    import uvicorn
    from openjev.inference import Engine
    engine = Engine(args.checkpoint, device=args.device)
    uvicorn.run(create_app(engine), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
