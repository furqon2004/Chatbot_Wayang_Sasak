"""Run the existing API and UI together; stop both if either service exits."""
from contextlib import suppress
import signal
import subprocess
import sys
import time


def main():
    stopping = False
    processes = []

    def stop(signum, frame):
        nonlocal stopping
        stopping = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)

    commands = (
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
         "--port", "8000", "--workers", "1", "--no-access-log"],
        [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
         "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"],
    )
    try:
        for command in commands:
            if stopping:
                return 0
            processes.append(subprocess.Popen(command))
        while not stopping:
            for process in processes:
                code = process.poll()
                if code is not None:
                    print("Application service exited; stopping container", file=sys.stderr, flush=True)
                    return code if code > 0 else 1
            time.sleep(0.25)
        return 0
    except OSError:
        print("Application service could not start", file=sys.stderr, flush=True)
        return 1
    finally:
        for process in processes:
            if process.poll() is None:
                with suppress(ProcessLookupError):
                    process.terminate()
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    raise SystemExit(main())
