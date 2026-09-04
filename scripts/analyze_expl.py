"""Audit expL raw facts, aggregate seed clusters, and enforce preregistered failure gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sloserve.analysis.expl import analyze_expl, require_gate_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results/raw/week7-expL"),
        help="directory containing all 24 completed expL points",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/sweeps/expL-adaptive-clipping.yaml"),
    )
    parser.add_argument(
        "--expkb-reference",
        type=Path,
        default=Path("results/raw/week6-expK-B"),
        help="saved real expK-B no-clip request facts used for the preregistered depth replay",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    # Persist every structured verdict before making a failed preregistered gate a hard process
    # failure. Raw-fact reconciliation errors remain immediate because the input is not auditable.
    analysis = analyze_expl(
        args.results,
        sweep_config=args.config,
        expkb_reference=args.expkb_reference,
        enforce_gates=False,
    )
    output = args.results / "analysis.json"
    output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(output)
    require_gate_results(analysis["gates"])


if __name__ == "__main__":
    main()
