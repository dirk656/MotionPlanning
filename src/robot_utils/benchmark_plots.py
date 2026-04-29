import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def pick_scenario(result_block, scenario_name=None):
    if not result_block:
        return None
    if scenario_name is None:
        return result_block[0]
    for entry in result_block:
        if entry.get("scenario") == scenario_name:
            return entry
    return result_block[0]


def plot_success_curves(results, out_dir, scenario_name=None, prefix="benchmark"):
    plt.figure()
    for planner, scenarios in results.items():
        entry = pick_scenario(scenarios, scenario_name)
        if entry is None:
            continue
        curve = entry.get("aggregate", {}).get("success_rate_curve", [])
        if not curve:
            continue
        xs = [c["time"] for c in curve]
        ys = [c["success_rate"] for c in curve]
        plt.plot(xs, ys, marker="o", label=planner)

    plt.xlabel("Time to first solution (s)")
    plt.ylabel("Success rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_success_curve.png", dpi=200)
    plt.close()


def plot_path_length_boxplot(results, out_dir, scenario_name=None, prefix="benchmark"):
    labels = []
    data = []
    for planner, scenarios in results.items():
        entry = pick_scenario(scenarios, scenario_name)
        if entry is None:
            continue
        dist = entry.get("aggregate", {}).get("path_length_distribution", [])
        if not dist:
            continue
        labels.append(planner)
        data.append(dist)

    if not data:
        return

    plt.figure()
    plt.boxplot(data, labels=labels, showfliers=False)
    plt.ylabel("Path length")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_path_length_boxplot.png", dpi=200)
    plt.close()


def plot_ablation_bar(ablation, out_dir, scenario_name=None, prefix="benchmark"):
    if not ablation:
        return

    labels = []
    values = []
    for group_name, results in ablation.items():
        if not results:
            continue
        planner_name = list(results.keys())[0]
        entry = pick_scenario(results.get(planner_name, []), scenario_name)
        if entry is None:
            continue
        success_rate = entry.get("aggregate", {}).get("success_rate")
        if success_rate is None:
            continue
        labels.append(group_name)
        values.append(float(success_rate))

    if not values:
        return

    plt.figure()
    plt.bar(labels, values)
    plt.ylabel("Success rate")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_ablation_bar.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark results")
    parser.add_argument("--result", type=str, required=True, help="Path to result.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--scenario", type=str, default=None, help="Scenario name to plot")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Output prefix")
    args = parser.parse_args()

    result_path = Path(args.result)
    out_dir = Path(args.out_dir) if args.out_dir else result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("benchmark", {})
    ablation = data.get("ablation") if isinstance(data.get("ablation"), dict) else None

    plot_success_curves(results, out_dir, scenario_name=args.scenario, prefix=args.prefix)
    plot_path_length_boxplot(results, out_dir, scenario_name=args.scenario, prefix=args.prefix)
    plot_ablation_bar(ablation, out_dir, scenario_name=args.scenario, prefix=args.prefix)


if __name__ == "__main__":
    main()
