from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROGDASAM_EPOCHS = 10

RESULTS_ROOT = Path("results")
OUT_ROOT = RESULTS_ROOT / "comparison_plots"

ROGDASAM_ROOT = RESULTS_ROOT / "rogdasam"
SAM_ROOT = RESULTS_ROOT / "sam"
ROGDASAM_THEN_SAM_ROOT = RESULTS_ROOT / "rogdasam_then_sam"

OUT_ROOT.mkdir(parents=True, exist_ok=True)


def extract_from_path(path, prefix):
    for part in path.parts:
        if part.startswith(prefix):
            return part.replace(prefix, "")
    return None


def get_value_from_df_or_path(df, csv_path, col_name, path_prefix):
    if col_name in df.columns:
        value = df[col_name].iloc[0]
        if pd.notna(value):
            return value

    return extract_from_path(csv_path, path_prefix)


def load_rogdasam_runs():
    rows = []

    for csv_path in ROGDASAM_ROOT.glob("**/metrics_history.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] Could not read {csv_path}: {e}")
            continue

        if df.empty:
            print(f"[SKIP] Empty CSV: {csv_path}")
            continue

        task = str(df["task_name"].iloc[0])
        optimizer = str(df["optimizer"].iloc[0])
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0

        lambda_val = get_value_from_df_or_path(df, csv_path, "lambda_val", "lambda_")
        eta_eps = get_value_from_df_or_path(df, csv_path, "eta_eps", "eta_")

        if eta_eps is None:
            label = f"ROGDASAM λ={lambda_val}"
        else:
            label = f"ROGDASAM λ={lambda_val}, η={eta_eps}"

        rows.append({
            "task": task,
            "optimizer": optimizer,
            "label": label,
            "lambda_val": lambda_val,
            "eta_eps": eta_eps,
            "seed": seed,
            "csv_path": csv_path,
            "df": df,
        })

    return rows


def load_sam_runs():
    rows = []

    for csv_path in SAM_ROOT.glob("**/metrics_history.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] Could not read {csv_path}: {e}")
            continue

        if df.empty:
            print(f"[SKIP] Empty CSV: {csv_path}")
            continue

        task = str(df["task_name"].iloc[0])
        optimizer = str(df["optimizer"].iloc[0])
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0

        rows.append({
            "task": task,
            "optimizer": optimizer,
            "label": "SAM",
            "lambda_val": None,
            "eta_eps": None,
            "seed": seed,
            "csv_path": csv_path,
            "df": df,
        })

    return rows


def load_rogdasam_then_sam_runs():
    rows = []

    for csv_path in ROGDASAM_THEN_SAM_ROOT.glob("**/metrics_history.csv"):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[SKIP] Could not read {csv_path}: {e}")
            continue

        if df.empty:
            print(f"[SKIP] Empty CSV: {csv_path}")
            continue

        task = str(df["task_name"].iloc[0])
        seed = int(df["seed"].iloc[0]) if "seed" in df.columns else 0

        lambda_val = get_value_from_df_or_path(df, csv_path, "lambda_val", "lambda_")
        eta_eps = get_value_from_df_or_path(df, csv_path, "eta_eps", "eta_")

        label = f"ROGDASAMThenSAM λ={lambda_val}, η={eta_eps}"

        rows.append({
            "task": task,
            "optimizer": "rogdasam_then_sam",
            "label": label,
            "lambda_val": lambda_val,
            "eta_eps": eta_eps,
            "seed": seed,
            "csv_path": csv_path,
            "df": df,
        })

    return rows


def add_gap_loss_if_possible(df):
    if "gap_loss" not in df.columns:
        if "train_loss" in df.columns and "val_loss" in df.columns:
            df = df.copy()
            df["gap_loss"] = df["train_loss"] - df["val_loss"]

    return df


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


def build_task_ranking(task, runs):
    rows = []
    selection_metric = get_selection_metric(task)

    for run in runs:
        df = run["df"]

        if selection_metric not in df.columns:
            print(f"[SKIP ranking] {selection_metric} not found in {run['csv_path']}")
            continue

        if "epoch" not in df.columns:
            print(f"[SKIP ranking] No epoch column in {run['csv_path']}")
            continue

        best_idx = df[selection_metric].idxmax()
        best_row = df.loc[best_idx]

        rows.append({
            "task": task,
            "model": run["label"],
            "optimizer": run["optimizer"],
            "lambda_val": run["lambda_val"],
            "eta_eps": run["eta_eps"],
            "selection_metric": selection_metric,
            "best_epoch": int(best_row["epoch"]),
            "best_value": float(best_row[selection_metric]),
            "final_value": float(df[selection_metric].iloc[-1]),
            "csv_path": str(run["csv_path"]),
        })

    if not rows:
        return pd.DataFrame()

    ranking_df = pd.DataFrame(rows)
    ranking_df = ranking_df.sort_values("best_value", ascending=False).reset_index(drop=True)
    ranking_df.insert(0, "rank", range(1, len(ranking_df) + 1))

    return ranking_df


def plot_task_ranking(task, runs):
    outdir = OUT_ROOT / task
    outdir.mkdir(parents=True, exist_ok=True)

    ranking_df = build_task_ranking(task, runs)

    if ranking_df.empty:
        print(f"[SKIP ranking plot] No ranking data for task={task}")
        return

    ranking_csv = outdir / "ranking_summary.csv"
    ranking_df.to_csv(ranking_csv, index=False)

    plt.figure(figsize=(14, 7))

    labels = [
        f"{row['rank']}. {row['model']}\nepoch={row['best_epoch']}"
        for _, row in ranking_df.iterrows()
    ]

    bars = plt.bar(labels, ranking_df["best_value"])

    for bar, value in zip(bars, ranking_df["best_value"]):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.ylabel(ranking_df["selection_metric"].iloc[0])
    plt.xlabel("Model / Best epoch")
    plt.title(f"{task.upper()} | Ranking by {ranking_df['selection_metric'].iloc[0]}")

    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(outdir / "ranking_selected_metric.png", dpi=200)
    plt.close()

    print(f"[Ranking] Saved: {ranking_csv}")


def get_metric_columns(runs):
    preferred = [
        "train_loss",
        "val_loss",
        "gap_loss",
        "eval_accuracy",
        "eval_f1",
        "eval_pearson",
        "eval_spearmanr",
        "eval_matthews_correlation",
    ]

    available = set()

    for run in runs:
        df = add_gap_loss_if_possible(run["df"])
        available.update(df.columns)

    metrics = [m for m in preferred if m in available]

    extra_eval_metrics = sorted(
        col for col in available
        if col.startswith("eval_") and col not in metrics and col != "eval_loss"
    )

    return metrics + extra_eval_metrics


def sort_runs_for_plot(runs):
    def key(run):
        label = run["label"]

        try:
            lv = float(run["lambda_val"])
        except Exception:
            lv = -1.0

        try:
            eta = float(run["eta_eps"])
        except Exception:
            eta = -1.0

        if label == "SAM":
            return (0, 0.0, 0.0)

        if label.startswith("ROGDASAMThenSAM"):
            return (1, lv, eta)

        if label.startswith("ROGDASAM"):
            return (2, lv, eta)

        return (9, lv, eta)

    return sorted(runs, key=key)


def plot_task_metric(task, runs, metric):
    outdir = OUT_ROOT / task
    outdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plotted_any = False

    for run in sort_runs_for_plot(runs):
        df = add_gap_loss_if_possible(run["df"])

        if metric not in df.columns:
            continue

        if "epoch" not in df.columns:
            print(f"[SKIP] No epoch column in {run['csv_path']}")
            continue

        plt.plot(
            df["epoch"],
            df[metric],
            label=run["label"],
            linewidth=2,
        )
        plotted_any = True

    if not plotted_any:
        plt.close()
        return

    if metric.startswith("gap_"):
        plt.axhline(0, linestyle="--", linewidth=1)

    if any(run["label"].startswith("ROGDASAMThenSAM") for run in runs):
        plt.axvline(
            ROGDASAM_EPOCHS,
            linestyle="--",
            linewidth=1,
            label="switch to SAM",
        )

    families = []
    if any(run["label"] == "SAM" for run in runs):
        families.append("SAM")
    if any(run["label"].startswith("ROGDASAMThenSAM") for run in runs):
        families.append("ROGDASAMThenSAM")
    if any(run["label"].startswith("ROGDASAM λ") for run in runs):
        families.append("ROGDASAM")

    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.title(f"{task.upper()} | {metric} | {' vs '.join(families)}")
    plt.legend()
    plt.tight_layout()

    safe_metric = metric.replace("/", "_")
    plt.savefig(outdir / f"{safe_metric}_comparison.png", dpi=200)
    plt.close()


def plot_all():
    runs = load_sam_runs() + load_rogdasam_runs() + load_rogdasam_then_sam_runs()

    if not runs:
        print("No runs found.")
        return

    tasks = sorted(set(run["task"] for run in runs))

    print("Found tasks:", tasks)
    print("Total runs:", len(runs))

    for task in tasks:
        task_runs = [run for run in runs if run["task"] == task]

        print(f"\nTask: {task}")
        for run in sort_runs_for_plot(task_runs):
            print(f"  - {run['label']} | {run['csv_path']}")

        metrics = get_metric_columns(task_runs)

        for metric in metrics:
            plot_task_metric(task, task_runs, metric)

        plot_task_ranking(task, task_runs)

    print(f"\nDone. Comparison plots saved in: {OUT_ROOT}")


if __name__ == "__main__":
    plot_all()