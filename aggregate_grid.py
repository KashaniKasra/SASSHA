import os
import json
import glob
import math
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RUNS_DIR = "runs"
BASE_TAG_PREFIX = "samRidge_grid_resnet32_cifar10"
OUT_AGG_DIR = os.path.join(RUNS_DIR, "aggregate_" + BASE_TAG_PREFIX)


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def load_run(run_dir):
    args_path = os.path.join(run_dir, "args.json")
    met_path = os.path.join(run_dir, "metrics_epoch.csv")
    if not (os.path.isfile(args_path) and os.path.isfile(met_path)):
        return None

    with open(args_path, "r") as f:
        args = json.load(f)

    run_tag = args.get("run_tag", "")
    if not run_tag.startswith(BASE_TAG_PREFIX):
        return None

    df = pd.read_csv(met_path)
    if df.empty:
        return None

    if "epoch" not in df.columns:
        df["epoch"] = np.arange(len(df))

    df = df.sort_values("epoch").reset_index(drop=True)

    return args, df


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def plot_curves_with_std(epochs, mean_train, std_train, mean_test, std_test, title, ylabel, out_path):
    plt.figure()

    plt.plot(epochs, mean_train, label="train (mean)")
    plt.plot(epochs, mean_test, label="test (mean)")

    if std_train is not None:
        plt.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2, label="train ±1 std")
    if std_test is not None:
        plt.fill_between(epochs, mean_test - std_test, mean_test + std_test, alpha=0.2, label="test ±1 std")

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_gap(epochs, mean_gap, std_gap, title, ylabel, out_path):
    plt.figure()

    plt.plot(epochs, mean_gap, label="gap (mean)")

    if std_gap is not None:
        plt.fill_between(epochs, mean_gap - std_gap, mean_gap + std_gap, alpha=0.2, label="gap ±1 std")

    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def aggregate_group(group_key, runs):
    lr, rho, hvp_every = group_key

    epoch_sets = [set(r["df"]["epoch"].tolist()) for r in runs]
    common_epochs = sorted(set.intersection(*epoch_sets)) if epoch_sets else []
    if len(common_epochs) == 0:
        return None

    def stack_metric(col):
        mats = []
        for r in runs:
            dfi = r["df"].set_index("epoch").loc[common_epochs]
            mats.append(dfi[col].to_numpy(dtype=float))
        return np.stack(mats, axis=0)

    tr_acc = stack_metric("train_acc")
    te_acc = stack_metric("test_acc")
    tr_loss = stack_metric("train_loss")
    te_loss = stack_metric("test_loss")

    mean = lambda x: x.mean(axis=0)
    std = lambda x: x.std(axis=0, ddof=1) if x.shape[0] > 1 else np.zeros_like(x[0])

    out = {
        "epochs": np.array(common_epochs),
        "train_acc_mean": mean(tr_acc),
        "train_acc_std": std(tr_acc),
        "test_acc_mean": mean(te_acc),
        "test_acc_std": std(te_acc),
        "train_loss_mean": mean(tr_loss),
        "train_loss_std": std(tr_loss),
        "test_loss_mean": mean(te_loss),
        "test_loss_std": std(te_loss)
    }

    # gap curves
    gap_acc = tr_acc - te_acc
    gap_loss = tr_loss - te_loss
    out["gap_acc_mean"] = mean(gap_acc)
    out["gap_acc_std"] = std(gap_acc)
    out["gap_loss_mean"] = mean(gap_loss)
    out["gap_loss_std"] = std(gap_loss)

    # scalar summaries per run then aggregate
    per_run = []
    for r in runs:
        df = r["df"]
        best_test_acc = df["test_acc"].max()
        final_test_acc = df.iloc[-1]["test_acc"]
        best_test_loss = df["test_loss"].min()
        final_test_loss = df.iloc[-1]["test_loss"]
        per_run.append({
            "run_dir": r["run_dir"],
            "seed": r["seed"],
            "best_test_acc": safe_float(best_test_acc),
            "final_test_acc": safe_float(final_test_acc),
            "best_test_loss": safe_float(best_test_loss),
            "final_test_loss": safe_float(final_test_loss)
        })
    per_run_df = pd.DataFrame(per_run)

    # group scalars
    def mstd(series):
        arr = series.to_numpy(dtype=float)
        return float(np.nanmean(arr)), float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else 0.0

    bacc_m, bacc_s = mstd(per_run_df["best_test_acc"])
    facc_m, facc_s = mstd(per_run_df["final_test_acc"])
    bloss_m, bloss_s = mstd(per_run_df["best_test_loss"])
    floss_m, floss_s = mstd(per_run_df["final_test_loss"])

    out["summary"] = {
        "lr": lr,
        "rho": rho,
        "hvp_every": hvp_every,
        "n_seeds": int(len(runs)),
        "best_test_acc_mean": bacc_m,
        "best_test_acc_std": bacc_s,
        "final_test_acc_mean": facc_m,
        "final_test_acc_std": facc_s,
        "best_test_loss_mean": bloss_m,
        "best_test_loss_std": bloss_s,
        "final_test_loss_mean": floss_m,
        "final_test_loss_std": floss_s
    }

    out["per_seed"] = per_run_df

    return out


def main():
    ensure_dir(OUT_AGG_DIR)

    # 1) collect runs
    runs = []
    for args_path in glob.glob(os.path.join(RUNS_DIR, "**", "args.json"), recursive=True):
        run_dir = os.path.dirname(args_path)
        loaded = load_run(run_dir)
        if loaded is None:
            continue

        args, df = loaded

        runs.append({
            "run_dir": run_dir,
            "args": args,
            "df": df,
            "seed": args.get("seed", None),
            "lr": args.get("lr", None),
            "rho": args.get("rho", None),
            "hvp_every": args.get("hvp_every", None)
        })

    if not runs:
        print("No matching runs found.")
        return

    # 2) group by grid key
    groups = defaultdict(list)
    for r in runs:
        key = (safe_float(r["lr"]), safe_float(r["rho"]), int(r["hvp_every"]))
        groups[key].append(r)

    # 3) aggregate each group
    leaderboard_rows = []
    for key, rs in sorted(groups.items()):
        lr, rho, hvp_every = key

        agg = aggregate_group(key, rs)
        if agg is None:
            continue

        group_name = f"lr{lr}_rho{rho}_hvpE{hvp_every}"
        out_dir = os.path.join(OUT_AGG_DIR, group_name)
        ensure_dir(out_dir)

        # save per-seed summary
        agg["per_seed"].to_csv(os.path.join(out_dir, "per_seed_summary.csv"), index=False)

        # save JSON summary
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(agg["summary"], f, indent=2)

        # save mean curves CSV
        curves_df = pd.DataFrame({
            "epoch": agg["epochs"],
            "train_acc_mean": agg["train_acc_mean"],
            "train_acc_std": agg["train_acc_std"],
            "test_acc_mean": agg["test_acc_mean"],
            "test_acc_std": agg["test_acc_std"],
            "train_loss_mean": agg["train_loss_mean"],
            "train_loss_std": agg["train_loss_std"],
            "test_loss_mean": agg["test_loss_mean"],
            "test_loss_std": agg["test_loss_std"],
            "gap_acc_mean": agg["gap_acc_mean"],
            "gap_acc_std": agg["gap_acc_std"],
            "gap_loss_mean": agg["gap_loss_mean"],
            "gap_loss_std": agg["gap_loss_std"]
        })
        curves_df.to_csv(os.path.join(out_dir, "curves_mean_std.csv"), index=False)

        # plots
        plot_curves_with_std(
            agg["epochs"], agg["train_acc_mean"], agg["train_acc_std"],
            agg["test_acc_mean"], agg["test_acc_std"],
            title=f"Accuracy (mean±std) | {group_name}",
            ylabel="Acc@1 (%)",
            out_path=os.path.join(out_dir, "acc_mean_std.png")
        )

        plot_curves_with_std(
            agg["epochs"], agg["train_loss_mean"], agg["train_loss_std"],
            agg["test_loss_mean"], agg["test_loss_std"],
            title=f"Loss (mean±std) | {group_name}",
            ylabel="Loss",
            out_path=os.path.join(out_dir, "loss_mean_std.png")
        )

        plot_gap(
            agg["epochs"], agg["gap_acc_mean"], agg["gap_acc_std"],
            title=f"Gap train-test Acc (mean±std) | {group_name}",
            ylabel="Acc gap (pp)",
            out_path=os.path.join(out_dir, "gap_acc_mean_std.png")
        )

        plot_gap(
            agg["epochs"], agg["gap_loss_mean"], agg["gap_loss_std"],
            title=f"Gap train-test Loss (mean±std) | {group_name}",
            ylabel="Loss gap",
            out_path=os.path.join(out_dir, "gap_loss_mean_std.png")
        )

        # leaderboard row
        s = agg["summary"]
        leaderboard_rows.append({
            "group": group_name,
            "lr": s["lr"],
            "rho": s["rho"],
            "hvp_every": s["hvp_every"],
            "n_seeds": s["n_seeds"],
            "best_test_acc_mean": s["best_test_acc_mean"],
            "best_test_acc_std": s["best_test_acc_std"],
            "final_test_acc_mean": s["final_test_acc_mean"],
            "final_test_acc_std": s["final_test_acc_std"],
            "best_test_loss_mean": s["best_test_loss_mean"],
            "best_test_loss_std": s["best_test_loss_std"]
        })

    leaderboard = pd.DataFrame(leaderboard_rows)
    leaderboard = leaderboard.sort_values(
        by=["best_test_acc_mean", "best_test_acc_std"],
        ascending=[False, True]
    ).reset_index(drop=True)

    leaderboard_path = os.path.join(OUT_AGG_DIR, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    plt.figure()
    plt.bar(np.arange(len(leaderboard)), leaderboard["best_test_acc_mean"].to_numpy())
    plt.xlabel("config rank")
    plt.ylabel("best test acc mean")
    plt.title("Configs ranked by best test acc (mean over seeds)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_AGG_DIR, "leaderboard_bar.png"), dpi=200)
    plt.close()

    print("DONE")
    print("Aggregate dir:", OUT_AGG_DIR)
    print("Leaderboard:", leaderboard_path)


if __name__ == "__main__":
    main()