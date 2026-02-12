from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

from mem2.analysis.score_parity import (
    flatten_eval_records,
    official_score_sum,
    strict_score_sum,
)
from mem2.cli.run import deep_merge, load_yaml
from mem2.core.arcmemo_compat import validate_arcmemo_compat_config
from mem2.io.hashing import stable_hash
from mem2.io.json_io import write_json
from mem2.orchestrator.runner import run_sync
from mem2.orchestrator.wiring import resolve_components


def _load_cfg(path: Path) -> dict:
    cfg = load_yaml(path)
    if "_base_" in cfg:
        base_cfg = load_yaml((path.parent / cfg["_base_"]).resolve())
        cfg = deep_merge(base_cfg, {k: v for k, v in cfg.items() if k != "_base_"})
    return cfg


def _extract_prompt_fingerprints(attempts) -> list[str]:
    fps = []
    for a in attempts:
        fp = (a.metadata or {}).get("prompt_fingerprint")
        if fp:
            fps.append(str(fp))
    return fps


def _prepare_offline_cfg(strict_cfg: dict, n_problems: int) -> dict:
    cfg = copy.deepcopy(strict_cfg)
    cfg["run"]["strict_arcmemo_compat"] = False
    cfg["run"]["run_type"] = "parity_offline"
    cfg["pipeline"]["provider"] = "mock"
    cfg["components"]["provider"]["profile_name"] = "mock"
    cfg["components"]["benchmark"]["limit"] = n_problems
    cfg["run"]["max_passes"] = 1
    cfg["run"]["retry_policy"]["max_passes"] = 1
    return cfg


def _prepare_live_cfg(strict_cfg: dict, n_problems: int) -> dict:
    cfg = copy.deepcopy(strict_cfg)
    cfg["run"]["run_type"] = "parity_live"
    cfg["components"]["benchmark"]["limit"] = n_problems
    return cfg


def _ensure_openai_key_if_file_present() -> bool:
    if os.getenv("OPENAI_API_KEY"):
        return True
    key_file = Path("/root/workspace/api_key.txt")
    if key_file.exists():
        key = key_file.read_text(encoding="utf-8").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ARC default parity lock harness")
    parser.add_argument(
        "--config",
        default="configs/experiments/arcmemo_arc_strict.yaml",
        help="Strict ArcMemo config path",
    )
    parser.add_argument("--offline-problems", type=int, default=2)
    parser.add_argument("--live-problems", type=int, default=1)
    parser.add_argument("--allow-live", action="store_true")
    args = parser.parse_args()

    strict_cfg = _load_cfg(Path(args.config))
    validate_arcmemo_compat_config(strict_cfg)

    offline_cfg = _prepare_offline_cfg(strict_cfg, args.offline_problems)
    comp_a = resolve_components(offline_cfg)
    bundle_a = run_sync(offline_cfg, comp_a)

    comp_b = resolve_components(offline_cfg)
    bundle_b = run_sync(offline_cfg, comp_b)

    fp_a = _extract_prompt_fingerprints(bundle_a.attempts)
    fp_b = _extract_prompt_fingerprints(bundle_b.attempts)

    flat_rows = flatten_eval_records(bundle_a.eval_records)
    offline_report = {
        "strict_config_hash": stable_hash(strict_cfg),
        "offline_config_hash": stable_hash(offline_cfg),
        "prompt_fingerprint_count": len(fp_a),
        "prompt_fingerprint_reproducible": fp_a == fp_b,
        "summary": bundle_a.summary,
        "official_score_recomputed": official_score_sum(flat_rows),
        "strict_score_recomputed": strict_score_sum(
            flat_rows, include_train=True, step_selection="last", aggregate_step_method="any"
        ),
    }
    write_json("outputs/parity/offline_parity_report.json", offline_report)

    live_report = {
        "attempted": False,
        "executed": False,
        "reason": "live verification not requested",
    }
    if args.allow_live:
        live_report["attempted"] = True
        if not _ensure_openai_key_if_file_present():
            live_report["reason"] = "OPENAI_API_KEY not configured (env or /root/workspace/api_key.txt)"
        else:
            try:
                live_cfg = _prepare_live_cfg(strict_cfg, args.live_problems)
                comp_live = resolve_components(live_cfg)
                bundle_live = run_sync(live_cfg, comp_live)
                attempt_count = int(bundle_live.summary.get("attempt_count", 0))
                live_report = {
                    "attempted": True,
                    "executed": True,
                    "executed_with_attempts": attempt_count > 0,
                    "reason": None,
                    "live_config_hash": stable_hash(live_cfg),
                    "summary": bundle_live.summary,
                }
                if attempt_count == 0:
                    live_report["reason"] = "live run returned zero attempts; inspect provider/model logs."
            except Exception as exc:  # pragma: no cover - depends on environment/provider setup
                live_report = {
                    "attempted": True,
                    "executed": False,
                    "executed_with_attempts": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
    write_json("outputs/parity/live_parity_report.json", live_report)

    print("offline report -> outputs/parity/offline_parity_report.json")
    print("live report    -> outputs/parity/live_parity_report.json")
    print("offline parity reproducible:", offline_report["prompt_fingerprint_reproducible"])
    print("live attempted:", live_report["attempted"], "executed:", live_report["executed"])


if __name__ == "__main__":
    main()

