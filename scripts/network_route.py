"""Prefer a responsive DNS answer for official package hosts, within this process.

Some CDN addresses time out on this host. TLS hostname/certificate validation
stays enabled; no hosts file, proxy, system setting or certificate is changed.
"""
import concurrent.futures
import socket
import ssl
import threading
import time


def prefer_responsive_cdn():
    original = socket.getaddrinfo
    selected = {}
    lock = threading.Lock()
    allowed = {"pypi.org", "files.pythonhosted.org", "download.pytorch.org"}

    def lookup(host, port, *args, **kwargs):
        records = original(host, port, *args, **kwargs)
        if host not in allowed or port not in (443, "443"):
            return records
        with lock:
            if host not in selected:
                addresses = {record[4][0] for record in records if record[0] == socket.AF_INET}

                def probe(address):
                    started = time.monotonic()
                    try:
                        with socket.create_connection((address, 443), timeout=4) as sock:
                            with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                                return time.monotonic() - started, address
                    except (OSError, ssl.SSLError):
                        return float("inf"), address

                with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(addresses))) as pool:
                    probes = list(pool.map(probe, addresses))
                best = min(probes, default=(float("inf"), None))
                selected[host] = best[1] if best[0] != float("inf") else None
                print(f"CDN route checked: {host}", flush=True)
        return sorted(records, key=lambda r: r[4][0] != selected[host])

    socket.getaddrinfo = lookup
