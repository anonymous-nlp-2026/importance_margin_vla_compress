"""D025 Gate Evaluation for IMM MVP.

Compares baseline and IMM evaluation JSONs against gate criteria:
  G1: Δ_k median improvement ≥ 2x
  G2: Top-k preservation rate improvement ≥ 10pp
  G3-deploy: ACIS+Prune(50%) action_loss / baseline ≤ 1.10  [primary, decides MVP pass/fail]
  G3-standard: standard action_loss / baseline               [reference, non-blocking]
  G3a-bypass: bypass action_loss / baseline                   [reference, distribution mismatch confound]
  G4: L·ε·k/Δ_k ratio median < 1.0 (certified bound)
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def get_condition_keys(results: dict) -> list[str]:
    return sorted(k for k in results if not k.startswith("_"))


def get_perturbed_keys(results: dict, prune: bool = False) -> list[str]:
    prune_str = "True" if prune else "False"
    return sorted(
        k for k in results
        if not k.startswith("_")
        and k.startswith("eps")
        and not k.startswith("eps0.0_")
        and k.endswith(f"prune{prune_str}")
    )


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def evaluate_gate(baseline: dict, imm: dict, bypass: dict | None = None,
                  acis_prune: dict | None = None,
                  condition: str | None = None) -> dict:
    """Evaluate gate criteria (D025).

    G3-deploy (ACIS+Prune / baseline) is the primary action loss gate.
    G3-standard and G3a-bypass are reference metrics that don't affect overall verdict.
    """
    clean_key = "eps0.0_pruneFalse"
    bl_clean_loss = baseline.get(clean_key, {}).get("action_loss")

    # G3-deploy (D025 primary): ACIS+Prune(50%) action_loss / baseline
    g3_deploy_ratio = None
    if acis_prune is not None:
        acis_prune_loss = acis_prune.get(clean_key, {}).get("action_loss")
        if bl_clean_loss and acis_prune_loss and bl_clean_loss > 0:
            g3_deploy_ratio = acis_prune_loss / bl_clean_loss

    # G3-standard (reference): standard eval action_loss / baseline
    g3_standard_ratio = None
    imm_clean_loss = imm.get(clean_key, {}).get("action_loss")
    if bl_clean_loss and imm_clean_loss and bl_clean_loss > 0:
        g3_standard_ratio = imm_clean_loss / bl_clean_loss

    # G3a-bypass (reference): bypass eval action_loss / baseline
    g3a_bypass_ratio = None
    if bypass is not None:
        bypass_clean_loss = bypass.get(clean_key, {}).get("action_loss")
        if bl_clean_loss and bypass_clean_loss and bl_clean_loss > 0:
            g3a_bypass_ratio = bypass_clean_loss / bl_clean_loss

    if condition is not None:
        conditions = [condition]
    else:
        bl_perturbed = set(get_perturbed_keys(baseline))
        imm_perturbed = set(get_perturbed_keys(imm))
        conditions = sorted(bl_perturbed & imm_perturbed)

    # Collect per-condition values
    margin_ratios = []
    pres_diffs = []
    led_ratios_imm = []

    per_condition = {}
    for cond in conditions:
        bl_data = baseline.get(cond, {})
        imm_data = imm.get(cond, {})

        bl_margin = bl_data.get("margin_median")
        imm_margin = imm_data.get("margin_median")
        bl_pres = bl_data.get("preservation_rate")
        imm_pres = imm_data.get("preservation_rate")
        imm_led = imm_data.get("l_eps_delta_ratio_median")

        cond_info = {}

        if bl_margin and imm_margin and bl_margin > 0:
            ratio = imm_margin / bl_margin
            margin_ratios.append(ratio)
            cond_info["margin_ratio"] = ratio
            cond_info["bl_margin_median"] = bl_margin
            cond_info["imm_margin_median"] = imm_margin

        if bl_pres is not None and imm_pres is not None:
            diff = imm_pres - bl_pres
            pres_diffs.append(diff)
            cond_info["pres_diff"] = diff
            cond_info["bl_preservation"] = bl_pres
            cond_info["imm_preservation"] = imm_pres

        if imm_led is not None:
            led_ratios_imm.append(imm_led)
            cond_info["imm_led_median"] = imm_led

        per_condition[cond] = cond_info

    # Aggregate
    g1_value = mean(margin_ratios) if margin_ratios else None
    g2_value = mean(pres_diffs) if pres_diffs else None
    g4_value = statistics.median(led_ratios_imm) if led_ratios_imm else None

    # Gate judgments
    g1_pass = g1_value is not None and g1_value >= 2.0
    g2_pass = g2_value is not None and g2_value >= 0.10
    g3_deploy_pass = g3_deploy_ratio is not None and g3_deploy_ratio <= 1.10
    g4_pass = g4_value is not None and g4_value < 1.0

    # D025: overall uses G3-deploy; G3-standard and G3a-bypass are reference only
    overall = g1_pass and g2_pass and g3_deploy_pass and g4_pass

    gate = {
        "G1_margin_median_ratio": {
            "value": g1_value,
            "threshold": "≥ 2.0",
            "pass": g1_pass,
        },
        "G2_preservation_rate_diff": {
            "value": g2_value,
            "threshold": "≥ 0.10",
            "pass": g2_pass,
        },
        "G3_deploy_ratio": {
            "value": g3_deploy_ratio,
            "threshold": "≤ 1.10",
            "pass": g3_deploy_pass,
            "note": "D025 primary: ACIS+Prune(50%) action_loss / baseline",
        },
        "G3_standard_ratio": {
            "value": g3_standard_ratio,
            "role": "reference",
            "note": "Standard eval action_loss / baseline (non-blocking)",
        },
        "G4_led_ratio_median": {
            "value": g4_value,
            "threshold": "< 1.0",
            "pass": g4_pass,
        },
    }

    if g3a_bypass_ratio is not None:
        gate["G3a_bypass_ratio"] = {
            "value": g3a_bypass_ratio,
            "role": "reference",
            "note": "Bypass action_loss / baseline (non-blocking, distribution mismatch confound)",
        }

    return {
        "conditions_evaluated": conditions,
        "per_condition": per_condition,
        "gate": gate,
        "overall_pass": overall,
    }


def format_val(v, fmt=".4f"):
    if v is None:
        return "N/A"
    return f"{v:{fmt}}"


def print_report(result: dict, output_json: str | None = None):
    if "error" in result:
        print(f"ERROR: {result['error']}")
        print(f"  Baseline keys: {result.get('baseline_keys')}")
        print(f"  IMM keys:      {result.get('imm_keys')}")
        sys.exit(1)

    gate = result["gate"]
    conditions = result["conditions_evaluated"]

    print("=" * 64)
    print("  D025 GATE EVALUATION REPORT")
    print("=" * 64)
    print()
    if conditions:
        print(f"Conditions evaluated: {', '.join(conditions)}")
    else:
        print("Conditions evaluated: (none — clean-only eval data)")
    print()

    # Per-condition detail
    if per_condition := result["per_condition"]:
        print("--- Per-Condition Detail ---")
        for cond, info in per_condition.items():
            print(f"\n  [{cond}]")
            if "margin_ratio" in info:
                print(f"    Δ_k median:  BL={format_val(info['bl_margin_median'])}  IMM={format_val(info['imm_margin_median'])}  ratio={format_val(info['margin_ratio'], '.2f')}x")
            if "pres_diff" in info:
                print(f"    Preservation: BL={format_val(info['bl_preservation'])}  IMM={format_val(info['imm_preservation'])}  diff={format_val(info['pres_diff'], '+.4f')}")
            if "imm_led_median" in info:
                print(f"    L·ε·k/Δ_k:  IMM={format_val(info['imm_led_median'])}")

    # Gate summary
    print()
    print("--- Gate Criteria ---")
    print()

    # Primary gates (affect verdict)
    primary = [
        ("G1", "Δ_k median improvement ≥ 2x", gate["G1_margin_median_ratio"]),
        ("G2", "Top-k preservation ≥ +10pp", gate["G2_preservation_rate_diff"]),
        ("G3-deploy", "ACIS+Prune(50%) action loss / baseline ≤ 1.10", gate["G3_deploy_ratio"]),
        ("G4", "LED ratio median < 1.0 (certified bound)", gate["G4_led_ratio_median"]),
    ]

    for tag, desc, info in primary:
        status = "PASS" if info["pass"] else "FAIL"
        val = info["value"]
        thr = info["threshold"]
        if tag == "G1":
            val_str = f"{format_val(val, '.2f')}x"
        elif tag == "G2":
            val_str = format_val(val, "+.4f")
        else:
            val_str = format_val(val, ".4f")
        print(f"  [{status}] {tag}: {desc}")
        print(f"         value={val_str}  threshold={thr}")

    # Reference metrics (don't affect verdict)
    print()
    print("  --- Reference metrics (non-blocking) ---")

    g3_std = gate["G3_standard_ratio"]
    print(f"  [REF]  G3-standard: standard action_loss / baseline")
    print(f"         value={format_val(g3_std['value'], '.4f')}")

    if "G3a_bypass_ratio" in gate:
        g3a = gate["G3a_bypass_ratio"]
        print(f"  [REF]  G3a-bypass: bypass action_loss / baseline (distribution mismatch confound)")
        print(f"         value={format_val(g3a['value'], '.4f')}")

    print()
    print("=" * 64)
    overall = result["overall_pass"]
    verdict = "PASS" if overall else "FAIL"
    print(f"  OVERALL VERDICT: {verdict}")
    print("  (G3-standard and G3a-bypass do NOT affect verdict)")
    print("=" * 64)

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nDetailed results saved to {out_path}")

    return 0 if overall else 1


def main():
    parser = argparse.ArgumentParser(
        description="D025 Gate Evaluation: compare baseline vs IMM eval results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  python analyze_gate.py --baseline bl.json --imm imm.json --acis-prune ap.json
  python analyze_gate.py --baseline bl.json --imm imm.json --acis-prune ap.json --bypass bypass.json
  python analyze_gate.py --baseline bl.json --imm imm.json --acis-prune ap.json --condition eps0.01_pruneFalse
  python analyze_gate.py --baseline bl.json --imm imm.json --acis-prune ap.json --output gate_report.json
""",
    )
    parser.add_argument("--baseline", required=True, help="Path to baseline eval JSON")
    parser.add_argument("--imm", required=True, help="Path to IMM eval JSON (standard eval, for G1/G2/G4 and G3-standard reference)")
    parser.add_argument("--acis-prune", default=None, help="Path to ACIS+Prune(50%%) eval JSON (for G3-deploy primary gate)")
    parser.add_argument("--bypass", default=None, help="Path to bypass-ACIS eval JSON (optional, for G3a reference metric)")
    parser.add_argument("--condition", default=None, help="Evaluate a single condition (default: aggregate all perturbed non-pruned)")
    parser.add_argument("--output", default=None, help="Save detailed results to JSON")
    args = parser.parse_args()

    baseline = load_results(args.baseline)
    imm = load_results(args.imm)
    acis_prune = load_results(args.acis_prune) if args.acis_prune else None
    bypass = load_results(args.bypass) if args.bypass else None

    result = evaluate_gate(baseline, imm, bypass=bypass, acis_prune=acis_prune,
                           condition=args.condition)
    exit_code = print_report(result, output_json=args.output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
