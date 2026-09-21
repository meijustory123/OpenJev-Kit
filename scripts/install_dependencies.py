"""pip wrapper for this host's uneven CDN connectivity."""
import runpy
import sys
from scripts.network_route import prefer_responsive_cdn


if __name__ == "__main__":
    prefer_responsive_cdn()
    sys.argv = ["pip", "install", "--cache-dir", ".pip-cache", "--index-url",
                "https://pypi.org/simple", "--disable-pip-version-check", *sys.argv[1:]]
    runpy.run_module("pip", run_name="__main__")
