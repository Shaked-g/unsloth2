"""Unit tests for scripts/smoke_local.py's own helper functions.

Deliberately does NOT invoke the full smoke_local.main() (which itself runs `pytest
tests/`) -- doing so from inside a pytest run would recursively re-run the whole suite.
Instead this exercises smoke_local's PASS/FAIL detection logic and its in-process
demo-build check directly, so a bug in the smoke orchestrator itself would be caught
here rather than only by eyeballing its printed output."""

from __future__ import annotations

import importlib.util
import os
import sys

SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "scripts")


def _load_smoke_local_module():
    spec = importlib.util.spec_from_file_location("smoke_local", os.path.join(SCRIPTS_DIR, "smoke_local.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_run_step_reports_pass_for_successful_command():
    smoke_local = _load_smoke_local_module()
    ok = smoke_local._run_step("trivial success", [sys.executable, "-c", "print('ok')"])
    assert ok is True


def test_run_step_reports_fail_for_failing_command():
    smoke_local = _load_smoke_local_module()
    ok = smoke_local._run_step("trivial failure", [sys.executable, "-c", "import sys; sys.exit(1)"])
    assert ok is False


def test_check_demo_build_succeeds_with_no_weights():
    smoke_local = _load_smoke_local_module()
    assert smoke_local._check_demo_build() is True


def test_main_returns_zero_when_all_steps_pass(monkeypatch):
    smoke_local = _load_smoke_local_module()
    monkeypatch.setattr(smoke_local, "_run_step", lambda name, cmd: True)
    monkeypatch.setattr(smoke_local, "_check_demo_build", lambda: True)
    assert smoke_local.main() == 0


def test_main_returns_nonzero_when_any_step_fails(monkeypatch):
    smoke_local = _load_smoke_local_module()
    call_count = {"n": 0}

    def flaky_run_step(name, cmd):
        call_count["n"] += 1
        return call_count["n"] != 2  # second step fails

    monkeypatch.setattr(smoke_local, "_run_step", flaky_run_step)
    monkeypatch.setattr(smoke_local, "_check_demo_build", lambda: True)
    assert smoke_local.main() == 1
