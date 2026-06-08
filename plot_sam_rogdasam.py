from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


ROGDASAM_EPOCHS = 10

roots = [
    Path("results/rogdasam"),
    Path("results/sam"),
    Path("results/rogdasam_then_sam"),
]


def extract_from_path(path, prefix):
    for part in path.parts:
        if part.startswith(prefix):
            return part.replace(prefix, "")
    return None


for root in roots:
    if not root.exists():
        continue

    for csv_path in root.glob("**/metrics_history.csv"):
        df = pd.read_csv(csv_path)

        if df.empty:
            print(f"[SKIP] Empty CSV: {csv_path}")
            continue

        outdir = csv_path.parent

        task = df["task_name"].iloc[0]
        optimizer = df["optimizer"].iloc[0]
        lr = df["lr"].iloc[0]
        rho = df["rho"].iloc[0]

        lambda_val = df["lambda_val"].iloc[0] if "lambda_val" in df.columns else None
        eta_eps = df["eta_eps"].iloc[0] if "eta_eps" in df.columns else None

        if lambda_val is None or pd.isna(lambda_val):
            lambda_val = extract_from_path(csv_path, "lambda_")

        if eta_eps is None or pd.isna(eta_eps):
            eta_eps = extract_from_path(csv_path, "eta_")

        if root.name == "rogdasam_then_sam":
            title_suffix = (
                f"{task} | ROGDASAMThenSAM | "
                f"lr{lr}_rho{rho}_lambda{lambda_val}_eta{eta_eps}_switch{ROGDASAM_EPOCHS}"
            )

        elif optimizer == "rogdasam":
            title_suffix = (
                f"{task} | ROGDASAM | "
                f"lr{lr}_rho{rho}_lambda{lambda_val}_eta{eta_eps}"
            )

        else:
            title_suffix = f"{task} | SAM | lr{lr}_rho{rho}"

        # Loss curve
        if "train_loss" in df.columns and "val_loss" in df.columns:
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], df["train_loss"], label="train loss")
            plt.plot(df["epoch"], df["val_loss"], label="validation loss")

            if root.name == "rogdasam_then_sam":
                plt.axvline(
                    ROGDASAM_EPOCHS,
                    linestyle="--",
                    linewidth=1,
                    label="switch to SAM",
                )

            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"Loss | {title_suffix}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "loss_curve.png", dpi=200)
            plt.close()

        # Gap loss curve
        if "train_loss" in df.columns and "val_loss" in df.columns:
            df = df.copy()
            df["gap_loss"] = df["train_loss"] - df["val_loss"]

            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], df["gap_loss"], label="gap loss")
            plt.axhline(0, linestyle="--", linewidth=1)

            if root.name == "rogdasam_then_sam":
                plt.axvline(
                    ROGDASAM_EPOCHS,
                    linestyle="--",
                    linewidth=1,
                    label="switch to SAM",
                )

            plt.xlabel("epoch")
            plt.ylabel("train loss - validation loss")
            plt.title(f"Gap train-validation Loss | {title_suffix}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / "gap_loss_curve.png", dpi=200)
            plt.close()

        # Metric curves
        metric_cols = [
            c for c in df.columns
            if c.startswith("eval_") and c not in ["eval_loss"]
        ]

        for metric_col in metric_cols:
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], df[metric_col], label=metric_col)

            if root.name == "rogdasam_then_sam":
                plt.axvline(
                    ROGDASAM_EPOCHS,
                    linestyle="--",
                    linewidth=1,
                    label="switch to SAM",
                )

            plt.xlabel("epoch")
            plt.ylabel(metric_col)
            plt.title(f"{metric_col} | {title_suffix}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(outdir / f"{metric_col}_curve.png", dpi=200)
            plt.close()

print("Done.")