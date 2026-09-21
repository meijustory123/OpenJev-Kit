"""Resumable PyTorch wheel download with official SHA256 verification.

Pinned hash read from https://download.pytorch.org/whl/cu128/torch/.
Supports the official CDN and its Aliyun mirror; bytes must match the official hash.
"""
import concurrent.futures
import argparse
import hashlib
import time
import urllib.request
from pathlib import Path

URL = "https://download.pytorch.org/whl/cu128/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl"
MIRROR_URL = "https://mirrors.aliyun.com/pytorch-wheels/cu128/torch-2.10.0%2Bcu128-cp311-cp311-win_amd64.whl"
SHA256 = "3523fda6e2cfab2b04ae09b1424681358e508bb3faa11ceb67004113d5e7acad"
SIZE = 2867372330
CHUNK = 8 * 1024 * 1024


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["official", "aliyun"], default="aliyun")
    args = parser.parse_args()
    download_url = URL if args.source == "official" else MIRROR_URL
    from scripts.network_route import prefer_responsive_cdn
    prefer_responsive_cdn()
    directory = Path(".uv-cache/downloads")
    directory.mkdir(parents=True, exist_ok=True)
    parts = directory / "torch-parts"
    parts.mkdir(exist_ok=True)
    destination = directory / "torch-2.10.0+cu128-cp311-cp311-win_amd64.whl"
    if destination.exists():
        with destination.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() == SHA256:
                print(f"Already verified: {destination}")
                return
        raise ValueError("已有 wheel 校验不匹配，请另存后再下载")

    def fetch(index):
        start = index * CHUNK
        end = min(SIZE, start + CHUNK) - 1
        path = parts / f"{index:04d}.part"
        if path.exists() and path.stat().st_size == end - start + 1:
            return path
        for attempt in range(6):
            try:
                request = urllib.request.Request(download_url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=90) as response:
                    expected = f"bytes {start}-{end}/{SIZE}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise ValueError(f"Range mismatch: {response.headers.get('Content-Range')}")
                    payload = response.read(end - start + 2)
                if len(payload) != end - start + 1:
                    raise ValueError("分块下载未完成")
                path.write_bytes(payload)
                return path
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(attempt + 1)

    total = (SIZE + CHUNK - 1) // CHUNK
    errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        futures = [pool.submit(fetch, i) for i in range(total)]
        for done, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as exc:
                errors.append(str(exc))
            if done % 20 == 0 or done == total:
                print(f"Downloaded {done}/{total} chunks", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} 个分块未完成，可重新运行续传: {errors[:3]}")
    temporary = destination.with_suffix(".pending")
    hasher = hashlib.sha256()
    with temporary.open("wb") as output:
        for i in range(total):
            data = (parts / f"{i:04d}.part").read_bytes()
            hasher.update(data)
            output.write(data)
    if hasher.hexdigest() != SHA256:
        raise ValueError("SHA256 校验失败，未安装")
    temporary.replace(destination)
    print(f"SHA256 verified: {destination}", flush=True)


if __name__ == "__main__":
    main()
