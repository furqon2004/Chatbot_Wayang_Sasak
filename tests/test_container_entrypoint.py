import signal
import subprocess

import pytest


class Process:
    def __init__(self, code=None, stubborn=False):
        self.returncode = code
        self.stubborn = stubborn
        self.waited = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        if not self.stubborn:
            self.returncode = 0

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("service", timeout)
        self.waited = True
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


@pytest.mark.parametrize("failed_index,exit_code,expected", [
    (0, 0, 1), (0, 7, 7), (1, 0, 1), (1, 7, 7), (0, -9, 1), (1, -9, 1),
])
def test_service_exit_stops_its_sibling(monkeypatch, failed_index, exit_code, expected):
    from scripts import container_entrypoint as runner

    processes = [Process(), Process()]
    processes[failed_index].returncode = exit_code
    pending = iter(processes)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args: next(pending))
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)

    assert runner.main() == expected
    assert all(process.poll() is not None and process.waited for process in processes)


def test_container_stop_terminates_both_services(monkeypatch):
    from scripts import container_entrypoint as runner

    processes = [Process(), Process()]
    pending = iter(processes)
    handlers = {}
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args: next(pending))
    monkeypatch.setattr(runner.signal, "signal", lambda sig, handler: handlers.update({sig: handler}))
    monkeypatch.setattr(runner.time, "sleep", lambda _: handlers[signal.SIGTERM](signal.SIGTERM, None))

    assert runner.main() == 0
    assert all(process.poll() == 0 and process.waited for process in processes)


def test_start_failure_cleans_up_the_process_already_started(monkeypatch):
    from scripts import container_entrypoint as runner

    process = Process()
    def start(*args):
        if not hasattr(start, "started"):
            start.started = True
            return process
        raise OSError("cannot start second process")
    monkeypatch.setattr(runner.subprocess, "Popen", start)
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)

    assert runner.main() == 1
    assert process.poll() == 0 and process.waited


def test_unresponsive_service_is_killed_and_reaped(monkeypatch):
    from scripts import container_entrypoint as runner

    processes = [Process(code=2), Process(stubborn=True)]
    pending = iter(processes)
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args: next(pending))
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)

    assert runner.main() == 2
    assert processes[1].killed and processes[1].waited
