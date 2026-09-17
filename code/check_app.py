#!/usr/bin/env python3
"""Launch check for the Shiny app: start server, verify HTTP 200, screenshot both tabs with headless Chrome driven over the DevTools protocol (real-time wait so websocket outputs render).

Usage: python code/check_app.py [--out-dir work/shiny-app-tabs/data] [--port 8123] [--wait-s 25]
Exits non-zero on failure (server not reachable, traceback in server log, or missing screenshot).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]


def wait_http(url: str, timeout: float) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            time.sleep(0.5)
    return False


def cdp_screenshot(url: str, out_png: Path, wait_s: float, port: int = 9333) -> dict:
    """Open url in headless Chrome via the DevTools protocol, wait real time, screenshot, and probe the DOM."""
    import asyncio
    import base64
    import json
    import websockets

    chrome = subprocess.Popen([CHROME, "--headless=new", "--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                               "--hide-scrollbars", "--window-size=1600,2600", f"--remote-debugging-port={port}", "about:blank"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        targets = None
        while time.time() - t0 < 30:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json", timeout=2) as r:
                    targets = json.loads(r.read())
                if targets:
                    break
            except Exception:  # noqa: BLE001
                time.sleep(0.3)
        ws_url = next(t["webSocketDebuggerUrl"] for t in targets if t.get("type") == "page")

        async def drive():
            async with websockets.connect(ws_url, max_size=50_000_000) as ws:
                mid = 0

                async def call(method, **params):
                    nonlocal mid
                    mid += 1
                    await ws.send(json.dumps({"id": mid, "method": method, "params": params}))
                    while True:
                        msg = json.loads(await ws.recv())
                        if msg.get("id") == mid:
                            return msg.get("result", {})
                await call("Page.enable")
                await call("Emulation.setDeviceMetricsOverride", width=1600, height=2600, deviceScaleFactor=1, mobile=False)
                await call("Page.navigate", url=url)
                await asyncio.sleep(wait_s)
                probe = await call("Runtime.evaluate", returnByValue=True, expression="""(() => ({
                    selects: Array.from(document.querySelectorAll('select')).map(s => [s.id, s.options.length, s.value]),
                    plots: Array.from(document.querySelectorAll('.shiny-plot-output img')).length,
                    widgets: Array.from(document.querySelectorAll('.js-plotly-plot')).length,
                    spinners: Array.from(document.querySelectorAll('.recalculating')).length,
                    statsRows: Array.from(document.querySelectorAll('table.table tbody tr')).length,
                    errors: Array.from(document.querySelectorAll('.shiny-output-error')).map(e => e.textContent).slice(0, 5)
                }))()""")
                shot = await call("Page.captureScreenshot", format="png", captureBeyondViewport=False)
                out_png.write_bytes(base64.b64decode(shot["data"]))
                return probe.get("result", {}).get("value", {})
        return asyncio.run(drive())
    finally:
        chrome.terminate()
        try:
            chrome.wait(timeout=10)
        except subprocess.TimeoutExpired:
            chrome.kill()


def run_tab(tab: str, port: int, out_png: Path, log: Path, wait_s: float) -> bool:
    env = {**os.environ, "DR_APP_DEFAULT_TAB": tab, "MPLBACKEND": "Agg"}
    with log.open("a") as lf:
        lf.write(f"\n===== {tab} =====\n"); lf.flush()
        srv = subprocess.Popen([PY, "-m", "shiny", "run", str(ROOT / "code" / "app.py"), "--port", str(port), "--host", "127.0.0.1"],
                               stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT)
        try:
            url = f"http://127.0.0.1:{port}/"
            ok = wait_http(url, 60)
            print(f"[{'PASS' if ok else 'FAIL'}] {tab}: server responds 200 at {url}")
            if not ok:
                return False
            if not Path(CHROME).exists():
                print(f"[SKIP] {tab}: Chrome not found, no screenshot")
                return True
            probe = cdp_screenshot(url, out_png, wait_s)
            print(f"[info] {tab}: DOM probe {probe}")
            ok = out_png.exists() and out_png.stat().st_size > 20_000
            filled = all(n > 0 for _, n, _ in probe.get("selects", [])) and probe.get("plots", 0) >= 2 and probe.get("widgets", 0) >= 1 \
                and not probe.get("errors")
            print(f"[{'PASS' if ok else 'FAIL'}] {tab}: screenshot {out_png} ({out_png.stat().st_size if out_png.exists() else 0} bytes)")
            print(f"[{'PASS' if filled else 'FAIL'}] {tab}: outputs rendered (selects populated, >=2 plots, >=1 plotly widget, no output errors)")
            return ok and filled
        finally:
            srv.terminate()
            try:
                srv.wait(timeout=10)
            except subprocess.TimeoutExpired:
                srv.kill()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=ROOT / "work" / "shiny-app-tabs" / "data")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--wait-s", type=float, default=25.0)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    log = a.out_dir / "server_log.txt"
    log.write_text("")
    results = [run_tab("tab_context", a.port, a.out_dir / "tab1_context.png", log, a.wait_s),
               run_tab("tab_rule", a.port, a.out_dir / "tab2_rule.png", log, a.wait_s)]
    text = log.read_text()
    tb = "Traceback" in text or "Error" in text and "ERROR" in text
    print(f"[{'FAIL' if tb else 'PASS'}] no Python traceback in server log ({len(text.splitlines())} lines)")
    results.append(not tb)
    print(f"\n{sum(results)}/{len(results)} checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
