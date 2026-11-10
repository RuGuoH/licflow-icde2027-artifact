"""Recompute paper results from the saved experimental inputs and outputs."""

import json
import re
import sys
from itertools import combinations
from pathlib import Path
from statistics import mean

import yaml

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "benchmark" / "tools"))

from ontology_metrics import _micro_f1, ontology_to_atoms, prepare_ontology_for_compare
from osadl_fact_f1 import osadl_fact_f1


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_yaml(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def atoms(entry):
    return ontology_to_atoms(prepare_ontology_for_compare(entry))


def jaccard(left, right):
    return 1.0 - len(left & right) / len(left | right) if left or right else 0.0


def modeling():
    root = DATA / "modeling"
    groups = read(root / "license_groups.json")
    excluded = read(root / "excluded_reference_atoms.json")
    all_ids = [name for ids in groups.values() for name in ids]
    reference = {
        name: atoms(load_yaml(root / "expert" / f"{name}.yaml")) - set(excluded[name])
        for name in all_ids
    }
    profiles = {path.stem: load_yaml(path) for path in (root / "osadl").glob("*.yaml")}

    def summarize(source, condition, ids):
        scores, facts, distances, ranges = {}, [], [], []
        for name in ids:
            entries = [
                load_yaml(root / source / condition / f"run{run}" / name / "license_final.yaml")
                for run in range(1, 4)
            ]
            units = [atoms(entry) for entry in entries]
            per_run = [_micro_f1(unit, reference[name])["f1"] for unit in units]
            scores[name] = mean(per_run)
            ranges.append(max(per_run) - min(per_run))
            distances.append(mean(jaccard(a, b) for a, b in combinations(units, 2)))
            if name in profiles:
                facts.append(mean(osadl_fact_f1(entry, profiles[name])["f1"] for entry in entries))
        return {
            "Entry-F1": mean(scores.values()),
            "OSADL-F1": mean(facts),
            "Jaccard distance": mean(distances),
            "Entry-F1 range": mean(ranges),
            "entries": len(ids) * 3,
            "OSADL licenses": len(facts),
        }, scores

    def resources(condition, ids, *, use_cache):
        records = read(root / "resources" / f"{condition}.json")
        if "flash" in condition:
            cached_rate, uncached_rate, output_rate = 0.02, 1.0, 4.0
        elif "pro" in condition:
            cached_rate, uncached_rate, output_rate = 0.15, 4.5, 13.5
        else:
            cached_rate, uncached_rate, output_rate = 0.7, 0.7, 2.8
        if isinstance(records, dict):
            elapsed, tokens = records["sec"], records["total"]
            prompt, completion = records["prompt"], records["completion"]
            cached = 0
        else:
            subset = [row for row in records if row["license_id"] in ids]
            elapsed = mean(row["elapsed_s"] for row in subset)
            tokens = mean(row["total_tokens"] for row in subset)
            prompt = mean(row["prompt_tokens"] for row in subset)
            completion = mean(row["completion_tokens"] for row in subset)
            cached = mean(row["cached_input_tokens"] for row in subset) if use_cache else 0
        return {
            "time_s": elapsed,
            "tokens_k": tokens / 1000,
            "estimated_cost_CNY": (
                cached * cached_rate + (prompt - cached) * uncached_rate + completion * output_rate
            ) / 1_000_000,
        }

    full = {}
    for path in sorted((root / "generated").iterdir()):
        condition = path.name
        row, scores = summarize("generated", condition, all_ids)
        row["Entry-F1 by category"] = {
            group: mean(scores[name] for name in ids) for group, ids in groups.items()
        }
        row.update(resources(condition, all_ids, use_cache=True))
        full[condition] = row

    subset = read(root / "baseline_licenses.json")
    baseline = {}
    for source, conditions in (
        ("generated", ("flash_t01_off", "nano_t01")),
        ("baseline", ("legacy_flash_off", "legacy_flash_on", "legacy_nano")),
    ):
        for condition in conditions:
            row, _ = summarize(source, condition, subset)
            row.update(resources(condition, subset, use_cache=False))
            baseline[condition] = row
    return full, baseline


def assessment(payload):
    buckets = payload["engine_events"]
    if buckets["errors"]:
        return "deny"
    return "conditional" if buckets["warnings"] or buckets["restrictions"] else "allow"


def events(payload, needle=""):
    return {
        (event["type"].strip(), re.sub(r"\s+", " ", event["info"].strip()))
        for event in payload["events"]
        if needle in event["type"].lower()
    }


def plans():
    root = DATA / "plans"
    ids = read(root / "ids.json")
    counts = {name: [0, 0, 0] for name in ("errors", "warnings", "obligations")}
    fixed = {"plans": len(ids), "successful": 0, "three_state": 0, "denial_boundary": 0}
    programs = {"runs": len(ids) * 3, "successful": 0, "assessment": 0, "derived_license": 0}
    text_differences = []
    for name in ids:
        expert = read(root / "expert_entries" / f"{name}.json")
        generated = read(root / "generated_entries" / f"{name}.json")
        reference_assessment = assessment(expert)
        generated_assessment = assessment(generated)
        fixed["successful"] += bool(generated["analysis_success"])
        fixed["three_state"] += reference_assessment == generated_assessment
        fixed["denial_boundary"] += (reference_assessment == "deny") == (generated_assessment == "deny")
        for bucket, needle in (("errors", "error"), ("warnings", "warning"), ("obligations", "restriction")):
            predicted = events(generated, needle)
            reference = events(expert, needle)
            counts[bucket][0] += len(predicted & reference)
            counts[bucket][1] += len(predicted)
            counts[bucket][2] += len(reference)

        for run in range(1, 4):
            result = read(root / "program_results" / f"run{run}" / f"{name}.json")
            programs["successful"] += bool(result["analysis_success"])
            programs["assessment"] += assessment(result) == reference_assessment
            programs["derived_license"] += result["derived_license"] == expert["derived_license"]
            if events(result) != events(expert) and name not in text_differences:
                text_differences.append(name)

    fixed["events"] = {
        bucket: {
            "precision": tp / predicted,
            "recall": tp / reference,
            "F1": 2 * tp / (predicted + reference),
        }
        for bucket, (tp, predicted, reference) in counts.items()
    }
    programs["event_text_match_all_runs"] = len(ids) - len(text_differences)
    programs["event_text_differences"] = sorted(text_differences)
    return fixed, programs


def case_studies():
    root = DATA / "cases"
    return {
        group: {
            name: {
                run: {key: payload[key] for key in ("verdict", "derived_license")}
                for run in ("expert", "run1", "run2", "run3")
                for payload in [read(root / run / f"{name}.json")]
            }
            for name in names
        }
        for group, names in read(root / "groups.json").items()
    }


if __name__ == "__main__":
    full, baseline = modeling()
    downstream, programs = plans()
    print(json.dumps({
        "question_based_modeling": full,
        "construction_method_comparison": baseline,
        "effects_on_reuse_analysis": downstream,
        "natural_language_plan_interpretation": programs,
        "case_studies": case_studies(),
    }, indent=2))
