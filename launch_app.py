"""Start one local AcciDEN server, or reuse the copy already running."""
import json
import os
from pathlib import Path
import socket
import sys
import time
import threading
import webbrowser
from urllib.request import ProxyHandler, build_opener

PORT = int(os.getenv("ACCIDEN_PORT", "8000"))
BASE_URL = f"http://127.0.0.1:{PORT}"


def is_acciden_running():
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open(BASE_URL + "/api/build-info", timeout=2) as response:
            data = json.load(response)
        return (data.get("build") == "admin-only-nav-5"
                and os.path.normcase(data.get("folder", "")) == os.path.normcase(str(Path(__file__).resolve().parent)))
    except (OSError, ValueError):
        return False


def open_dashboards():
    # Admin owns the two response tabs so each sidebar can reuse their handles.
    webbrowser.open_new_tab(f"{BASE_URL}/admin")


def open_when_ready(server, stopped):
    while not stopped.wait(0.5):
        if server.started:
            print("Server ready. Opening Admin. Its sidebar opens or switches dashboards.")
            open_dashboards()
            return


def main():
    global PORT, BASE_URL
    os.chdir(Path(__file__).resolve().parent)
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
    try:
        listener.bind(("127.0.0.1", PORT))
    except OSError as error:
        listener.close()
        if error.errno not in (48, 98, 10048) and getattr(error, "winerror", None) != 10048:
            print(f"Could not start AcciDEN: {error}")
            return 1
        # Allow an existing copy a few seconds to finish model warm-up.
        for _ in range(6):
            if is_acciden_running():
                print("AcciDEN is already running. Keep its existing server window open.")
                print(f"Use your existing tabs at {BASE_URL}; no new tabs were opened.")
                return 0
            time.sleep(1)
        # An older folder may still own port 8000. Start this folder on a
        # different port instead of silently reopening that older dashboard.
        for candidate in range(PORT + 1, PORT + 21):
            listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            try:
                listener.bind(("127.0.0.1", candidate))
                PORT = candidate
                BASE_URL = f"http://127.0.0.1:{PORT}"
                print(f"An older server is running. Opening this updated folder at {BASE_URL}")
                break
            except OSError:
                listener.close()
        else:
            print("No available local port. Close an earlier server and try again.")
            return 1
    try:
        import uvicorn
        print("AcciDEN starting. Keep this window open.")
        for dashboard in ("admin", "ambulance", "police"):
            print(f"  {dashboard.title()}: {BASE_URL}/{dashboard}")
        config = uvicorn.Config("server:combined_app", host="127.0.0.1", port=PORT, reload=False)
        server = uvicorn.Server(config)
        stopped = threading.Event()
        threading.Thread(target=open_when_ready, args=(server, stopped), daemon=True).start()
        try:
            server.run(sockets=[listener])
        finally:
            stopped.set()
    finally:
        listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
