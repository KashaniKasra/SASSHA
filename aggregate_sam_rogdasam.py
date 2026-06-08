import json
from pathlib import Path

import pandas as pd


RESULTS_ROOT = Path("results")

rows = []


def get_selection_metric(task):
    task = str(task).lower()

    if task == "mrpc":
        return "eval_f1"
    elif task == "rte":
        return "eval_accuracy"
    elif task == "stsb":
        return "eval_spearmanr"
    else:
        return "eval_accuracy"


def extract_from_path(path, prefix):
    for part in path.parts:
        if part.startswith(prefix):
            return part.replace(prefix, "")
    return None


def add_best_metric_info(row, result_dir):
    metrics_path = result_dir / "metrics_history.csv"

    if not metrics_path.exists():
        row["selection_metric"] = None
        row["best_epoch"] = None
        row["best_value"] = None
        row["final_value"] = None
        return row

    df_hist = pd.read_csv(metrics_path)

    if df_hist.empty:
        row["selection_metric"] = None
        row["best_epoch"] = None
        row["best_value"] = None
        row["final_value"] = None
        return row

    task = row.get(
        "task_name",
        df_hist["task_name"].iloc[0] if "task_name" in df_hist.columns else None,
    )

    metric_col = get_selection_metric(task)
    row["selection_metric"] = metric_col

    if metric_col not in df_hist.columns:
        row["best_epoch"] = None
        row["best_value"] = None
        row["final_value"] = None
        return row

    best_row = df_hist.loc[df_hist[metric_col].idxmax()]

    row["best_epoch"] = int(best_row["epoch"]) if "epoch" in df_hist.columns else None
    row["best_value"] = float(best_row[metric_col])
    row["final_value"] = float(df_hist[metric_col].iloc[-1])

    return row


def add_run(path, optimizer_family):
    with open(path) as f:
        row = json.load(f)

    result_dir = path.parent

    row["optimizer_family"] = optimizer_family
    row["result_dir"] = str(result_dir)

    if optimizer_family == "sam":
        row["optimizer"] = row.get("optimizer", "samsgd")
        row["lambda_val"] = None
        row["eta_eps"] = None

    elif optimizer_family == "rogdasam":
        row["optimizer"] = row.get("optimizer", "rogdasam")

        if row.get("lambda_val") is None:
            row["lambda_val"] = extract_from_path(path, "lambda_")

        if row.get("eta_eps") is None:
            row["eta_eps"] = extract_from_path(path, "eta_")

    elif optimizer_family == "rogdasam_then_sam":
        row["optimizer"] = "rogdasam_then_sam"

        if row.get("lambda_val") is None:
            row["lambda_val"] = extract_from_path(path, "lambda_")

        if row.get("eta_eps") is None:
            row["eta_eps"] = extract_from_path(path, "eta_")

        row["rogdasam_epochs"] = row.get("rogdasam_epochs", None)

    row["total_epochs"] = row.get("num_train_epochs", None)

    row = add_best_metric_info(row, result_dir)

    rows.append(row)


# ROGDASAM results
for path in Path("results/rogdasam").glob("**/final_summary.json"):
    add_run(path, "rogdasam")


# SAM results
for path in Path("results/sam").glob("**/final_summary.json"):
    add_run(path, "sam")


# ROGDASAMThenSAM
for path in Path("results/rogdasam_then_sam").glob("**/final_summary.json"):
    add_run(path, "rogdasam_then_sam")


df = pd.DataFrame(rows)

sort_cols = [
    c for c in [
        "task_name",
        "optimizer_family",
        "lambda_val",
        "eta_eps",
        "seed",
    ]
    if c in df.columns
]

if sort_cols:
    df = df.sort_values(sort_cols)

out = Path("results/optimizer_comparison_summary.csv")
df.to_csv(out, index=False)

print(f"Saved: {out}")