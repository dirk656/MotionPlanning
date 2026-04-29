import argparse
import csv
import json
from pathlib import Path


def _format_float(val, digits=4):
    try:
        if val is None:
            return "nan"
        return f"{float(val):.{digits}f}"
    except Exception:
        return "nan"


def _flatten_results(results, group_name=None):
    rows = []
    for planner_name, scenarios in results.items():
        for scenario_entry in scenarios:
            agg = scenario_entry.get("aggregate", {})
            row = {
                "group": group_name or "benchmark",
                "planner": planner_name,
                "scenario": scenario_entry.get("scenario", "unknown"),
                "success_rate": agg.get("success_rate"),
                "path_length_mean": agg.get("path_length_mean"),
                "path_length_std": agg.get("path_length_std"),
                "planning_time_mean": agg.get("planning_time_mean"),
                "planning_time_std": agg.get("planning_time_std"),
                "collision_rate_mean": agg.get("collision_rate_mean"),
                "min_clearance_mean": agg.get("min_clearance_mean"),
                "replan_count_mean": agg.get("replan_count_mean"),
                "time_to_first_solution_mean": agg.get("time_to_first_solution_mean"),
                "replan_latency_mean": agg.get("replan_latency_mean"),
            }
            rows.append(row)
    return rows


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_latex_table(path, rows, caption="Benchmark Summary", label="tab:benchmark"):
    if not rows:
        return
    headers = [
        "Planner",
        "Scenario",
        "Success",
        "PathLen",
        "PlanTime",
        "ReplanLat",
    ]
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{l l r r r r}")
    lines.append("\\toprule")
    lines.append(" \\ ".join(headers) + " \\")
    lines.append("\\midrule")

    for row in rows:
        lines.append(
            "{} & {} & {} & {} & {} & {} \\").format(
                row.get("planner", ""),
                row.get("scenario", ""),
                _format_float(row.get("success_rate")),
                _format_float(row.get("path_length_mean")),
                _format_float(row.get("planning_time_mean")),
                _format_float(row.get("replan_latency_mean")),
            )

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Export benchmark results to CSV/LaTeX")
    parser.add_argument("--result", type=str, required=True, help="Path to result.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--prefix", type=str, default="benchmark", help="Output prefix")
    args = parser.parse_args()

    result_path = Path(args.result)
    out_dir = Path(args.out_dir) if args.out_dir else result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(result_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bench_results = data.get("benchmark", {})
    rows = _flatten_results(bench_results)
    if rows:
        _write_csv(out_dir / f"{args.prefix}_summary.csv", rows)
        _write_latex_table(out_dir / f"{args.prefix}_table.tex", rows)

    ablation = data.get("ablation")
    if isinstance(ablation, dict):
        ablation_rows = []
        for group_name, group_results in ablation.items():
            ablation_rows.extend(_flatten_results(group_results, group_name=group_name))
        if ablation_rows:
            _write_csv(out_dir / f"{args.prefix}_ablation_summary.csv", ablation_rows)
            _write_latex_table(
                out_dir / f"{args.prefix}_ablation_table.tex",
                ablation_rows,
                caption="Ablation Summary",
                label="tab:ablation",
            )


if __name__ == "__main__":
    main()
