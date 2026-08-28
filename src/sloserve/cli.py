"""Command-line entry point for SLOServe development utilities."""

from __future__ import annotations

import argparse
import json
import shlex
from collections.abc import Sequence

from sloserve.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level CLI parser."""
    parser = argparse.ArgumentParser(prog="sloserve")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check_parser = subparsers.add_parser("config-check", help="validate an experiment YAML file")
    check_parser.add_argument("--config", required=True, help="path to the YAML config")

    serve_parser = subparsers.add_parser(
        "serve-command",
        help="print the pinned vllm serve command derived from config",
    )
    serve_parser.add_argument("--config", required=True, help="path to the YAML config")
    serve_parser.add_argument(
        "--format",
        choices=["shell", "json", "lines"],
        default="shell",
        help="shell: one quoted command; json: argv list; lines: one arg per line",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run a SLOServe development command."""
    args = build_parser().parse_args(argv)
    if args.command == "config-check":
        config = load_config(args.config)
        summary = {
            "model": config.backend.model,
            "model_revision": config.backend.model_revision,
            "policy": config.router.policy.value,
            "random_seed": config.workload.random_seed,
            "repetitions": config.workload.repetitions,
            "tokenizer_revision": config.backend.tokenizer_revision,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "serve-command":
        config = load_config(args.config)
        serve_args = config.vllm_serve_args()
        if args.format == "json":
            print(json.dumps(["vllm", *serve_args]))
        elif args.format == "lines":
            print("\n".join(serve_args))
        else:
            print(shlex.join(["vllm", *serve_args]))
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
