"""Smoke tests for the installed command-line interface."""

import json
from pathlib import Path

import pytest

from sloserve.cli import main

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = str(PROJECT_ROOT / "configs" / "base.yaml")


def test_config_check_command(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["config-check", "--config", str(PROJECT_ROOT / "configs" / "base.yaml")])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert '"policy": "fcfs"' in captured.out
    assert '"model_revision": "c1899de289a04d12100db370d81485cdf75e47ca"' in captured.out
    assert '"tokenizer_revision": "c1899de289a04d12100db370d81485cdf75e47ca"' in captured.out


def test_serve_command_shell(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["serve-command", "--config", BASE_CONFIG])

    assert exit_code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith("vllm serve Qwen/Qwen3-0.6B")
    assert "--revision c1899de289a04d12100db370d81485cdf75e47ca" in out
    assert "--enforce-eager" in out


def test_serve_command_json_is_argv(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["serve-command", "--config", BASE_CONFIG, "--format", "json"])

    assert exit_code == 0
    argv = json.loads(capsys.readouterr().out)
    assert argv[0] == "vllm"
    assert argv[1] == "serve"
    assert "--gpu-memory-utilization" in argv
