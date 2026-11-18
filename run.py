"""
run.py - the single entry point for Drone Oversight.

    python run.py          App mode: builds the UI if needed, serves everything
                           from one port (8000) and opens the browser.
    python run.py --dev    Developer mode: API with auto-reload + Vite dev
                           server (http://localhost:5173) via `npm start`.
    python run.py --api    Backend only, with auto-reload (used by `npm start`).

The packaged Windows executable (DroneOversight.spec) always runs App mode.
"""
from __future__ import annotations

import argparse
import multiprocessing
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
FROZEN = getattr(sys, "frozen", False)
ROOT = Path(sys.executable).parent if FROZEN else Path(__file__).resolve().parent


def _utf8_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def check_python_deps() -> None:
    missing = []
    for pkg in ("fastapi", "uvicorn", "numpy", "python_multipart", "PIL", "cv2", "scipy", "pyproj"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[!] Installing missing packages: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])


def _npm() -> str:
    path = shutil.which("npm")
    if not path:
        raise SystemExit("[!] Node.js/npm was not found. Install Node.js 22.12+ and try again.")
    return path


def ensure_ui_built() -> None:
    """Build dashboard -> dist/ when missing or older than the sources."""
    index = ROOT / "dist" / "index.html"
    sources = [p for p in (ROOT / "dashboard").rglob("*") if p.is_file()]
    sources += [ROOT / "package.json", ROOT / "vite.config.js"]
    newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0)
    if index.exists() and index.stat().st_mtime >= newest:
        return
    npm = _npm()
    if not (ROOT / "node_modules").is_dir():
        print("[*] Installing frontend packages...")
        subprocess.check_call([npm, "ci"], cwd=ROOT)
    print("[*] Building the dashboard...")
    subprocess.check_call([npm, "run", "build"], cwd=ROOT)


def _open_browser_when_ready(url: str, probe: str, timeout_s: int = 90) -> None:
    def worker() -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                urllib.request.urlopen(probe, timeout=1).close()
                print(f"[*] Opening {url}")
                webbrowser.open(url)
                return
            except Exception:
                time.sleep(0.5)
    threading.Thread(target=worker, daemon=True).start()


def run_app(open_browser: bool = True) -> None:
    import uvicorn
    from api import app

    if not FROZEN:
        ensure_ui_built()
    url = f"http://localhost:{PORT}"
    print(f"[*] Drone Oversight running at {url}  (Ctrl+C to stop)")
    if open_browser:
        _open_browser_when_ready(url, f"{url}/missions")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")


def run_api_reload() -> None:
    import uvicorn

    uvicorn.run("api:app", host=HOST, port=PORT, reload=True, app_dir=str(ROOT))


def run_dev(open_browser: bool = True) -> None:
    npm = _npm()
    if not (ROOT / "node_modules").is_dir():
        subprocess.check_call([npm, "ci"], cwd=ROOT)
    if open_browser:
        _open_browser_when_ready("http://localhost:5173", "http://localhost:5173/")
    subprocess.call([npm, "start"], cwd=ROOT)


def main() -> None:
    multiprocessing.freeze_support()  # needed by PyInstaller
    _utf8_output()
    parser = argparse.ArgumentParser(description="Drone Oversight launcher")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dev", action="store_true", help="API (auto-reload) + Vite dev server")
    group.add_argument("--api", action="store_true", help="backend only, with auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    args = parser.parse_args()

    if not FROZEN:
        check_python_deps()
    if args.api:
        run_api_reload()
    elif args.dev:
        run_dev(open_browser=not args.no_browser)
    else:
        run_app(open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
