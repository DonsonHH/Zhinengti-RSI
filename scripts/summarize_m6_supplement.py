"""Descriptive M6 statistics from the delivered JSON/gzip, without PyTorch."""
import argparse
import json
import statistics
from pathlib import Path

from m6_evidence_io import read_json


def describe(values):
    return {"per_training_seed": values, "mean": statistics.mean(values),
            "sample_sd_across_training_seeds": statistics.stdev(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summaries = {
        variant: [read_json(args.results / f"{variant}-seed-{seed}/summary.json")
                  for seed in range(3)]
        for variant in ("joint_vae", "no_vae", "independent_targets")
    }
    result = {"descriptive_only": True, "independent_training_seeds_per_variant": 3,
              "held_out_conditions_per_seed": 48, "groups": {}}
    for variant, runs in summaries.items():
        groups = result["groups"][variant] = {}
        for method in runs[0]["test"]:
            method_rows = [run["test"][method]["cases"] for run in runs]
            stats = groups[method] = {
                metric: describe([statistics.mean(row[metric] for row in rows)
                                  for rows in method_rows])
                for metric in runs[0]["test"][method]["metrics"]
            }
            stats["worst_case_peak_age_steps"] = max(
                row["peak_age"] for rows in method_rows for row in rows)
            stats["first24_prediction_mse"] = describe([
                statistics.mean(statistics.mean(row["prediction_mse_series"][:24])
                                for row in rows) for rows in method_rows])
            noise_differences = []
            for rows in method_rows:
                clean = {(row["layout_seed"], row["profile"]): row
                         for row in rows if row["noise"] == 0}
                noise_differences.append(statistics.mean(
                    row["deadline_violation"] -
                    clean[row["layout_seed"], row["profile"]]["deadline_violation"]
                    for row in rows if row["noise"] > 0))
            stats["paired_noise_violation_difference"] = describe(noise_differences)
        costs = []
        for seed, run in enumerate(runs):
            lines = (args.results / f"{variant}-seed-{seed}/learning.jsonl").read_text().splitlines()
            training = json.loads(lines[-1])["seconds"]
            costs.append({"seed": seed, "selected_episode": run["selected_episode"],
                          "training_plus_validation_seconds": training,
                          "post_selection_evaluation_seconds": run["seconds"] - training})
        groups["costs"] = costs
        groups["mean_training_plus_validation_seconds"] = statistics.mean(
            row["training_plus_validation_seconds"] for row in costs)
        groups["mean_post_selection_evaluation_seconds"] = statistics.mean(
            row["post_selection_evaluation_seconds"] for row in costs)
    differences = [joint["test"]["learned"]["metrics"]["deadline_violation"] -
                   no_vae["test"]["learned"]["metrics"]["deadline_violation"]
                   for joint, no_vae in zip(summaries["joint_vae"], summaries["no_vae"])]
    result["paired_joint_vae_minus_no_vae_violation"] = describe(differences)
    result["vae_better_training_seeds"] = sum(value < 0 for value in differences)
    result["vae_worse_training_seeds"] = sum(value > 0 for value in differences)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2)
        output.write("\n")
    print("M6 descriptive supplement written; no training or checkpoint selection")


if __name__ == "__main__":
    main()
