"""Aggregate per-seed experiment CSVs into mean +/- std and publication-style plots.

Reads the per-(scenario, seed) CSVs written by run_seeded_experiment.py under
``outputs/seeds/`` and produces, under ``outputs/aggregated/``:

  * ``<scenario>_tabular.csv``   : Epoch, <algo>_mean, <algo>_std  (IQL/Hyst/VDN + baselines)
  * ``<scenario>_qmix.csv``      : Episode, qmix_mean, qmix_std       (1x4 scenarios only)
  * ``<scenario>_learning.png``  : tabular learning curves with +/-std shaded bands
  * ``<scenario>_qmix.png``      : QMIX learning curve with +/-std band (1x4 only)
  * ``cross_algorithm_bars.png`` : final-performance delay per scenario/algorithm w/ error bars
  * ``summary.md``               : final mean +/- std table (n seeds) per scenario/algorithm

"Delay" = |return| (lower is better) is used for the bar chart; learning curves plot the raw
return (higher / less-negative is better).
"""

import os
import glob
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED_DIR = os.path.join("outputs", "seeds")
AGG_DIR = os.path.join("outputs", "aggregated")

ONE_X_FOUR = ["baseline", "dense_wave", "cross_surge", "split_rush"]
RESCO = ["cologne3", "grid4x4"]
ALL_SCENARIOS = ONE_X_FOUR + RESCO

# Pretty labels for the algorithm CSV columns.
TAB_ALGO_LABELS = {
    "Tabular_IQL_Reward": "IQL",
    "Hysteretic_Reward": "Hysteretic",
    "VDN_Reward": "VDN",
    "Fixed_Baseline_Reward": "Fixed-Time",
    "Fixed_Native_Reward": "Fixed (native)",
    "Fixed_RoundRobin_Reward": "Fixed (round-robin)",
}
LEARNER_COLS = ["Tabular_IQL_Reward", "Hysteretic_Reward", "VDN_Reward"]
BASELINE_COLS = ["Fixed_Baseline_Reward", "Fixed_Native_Reward", "Fixed_RoundRobin_Reward"]


def _seed_of(path):
    m = re.search(r"_seed(\d+)\.csv$", path)
    return int(m.group(1)) if m else None


def _stack(frames, index_col, value_col):
    """Stack a value column across seed-frames into a (n_seeds, n_points) array aligned on index_col."""
    pivoted = []
    ref_index = None
    for df in frames:
        s = df.set_index(index_col)[value_col]
        pivoted.append(s)
        ref_index = s.index if ref_index is None else ref_index
    mat = np.vstack([s.reindex(ref_index).to_numpy(dtype=float) for s in pivoted])
    return ref_index.to_numpy(), mat  # x, (n_seeds, n_points)


def aggregate_tabular(scenario):
    files = sorted(glob.glob(os.path.join(SEED_DIR, f"training_evaluation_log_{scenario}_seed*.csv")))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    cols = [c for c in frames[0].columns if c != "Epoch"]
    out = pd.DataFrame({"Epoch": frames[0]["Epoch"].to_numpy()})
    mats = {}
    for col in cols:
        x, mat = _stack(frames, "Epoch", col)
        out[f"{col}_mean"] = np.nanmean(mat, axis=0)
        out[f"{col}_std"] = np.nanstd(mat, axis=0)
        mats[col] = mat  # (n_seeds, n_points)
    os.makedirs(AGG_DIR, exist_ok=True)
    out.to_csv(os.path.join(AGG_DIR, f"{scenario}_tabular.csv"), index=False)
    return {"df": out, "cols": cols, "mats": mats, "n_seeds": len(files)}


def aggregate_qmix(scenario):
    files = sorted(glob.glob(os.path.join(SEED_DIR, f"qmix_results_{scenario}_seed*.csv")))
    if not files:
        return None
    frames = [pd.read_csv(f) for f in files]
    x, mat = _stack(frames, "episode", "eval_reward")
    out = pd.DataFrame({"Episode": x,
                        "qmix_mean": np.nanmean(mat, axis=0),
                        "qmix_std": np.nanstd(mat, axis=0)})
    os.makedirs(AGG_DIR, exist_ok=True)
    out.to_csv(os.path.join(AGG_DIR, f"{scenario}_qmix.csv"), index=False)
    return {"df": out, "mat": mat, "n_seeds": len(files)}


def plot_tabular_learning(scenario, tab):
    df, cols, n = tab["df"], tab["cols"], tab["n_seeds"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = df["Epoch"].to_numpy()
    for col in cols:
        mean = df[f"{col}_mean"].to_numpy()
        std = df[f"{col}_std"].to_numpy()
        label = TAB_ALGO_LABELS.get(col, col)
        if col in BASELINE_COLS:
            ax.axhline(np.nanmean(mean), linestyle="--", alpha=0.7,
                       label=f"{label} (mean)")
        else:
            ax.plot(ep, mean, marker="o", label=label)
            ax.fill_between(ep, mean - std, mean + std, alpha=0.18)
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Evaluation return (higher = better)")
    ax.set_title(f"{scenario}: tabular learning curves (mean ± std over {n} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(AGG_DIR, f"{scenario}_learning.png"), dpi=140)
    plt.close(fig)


def plot_qmix_learning(scenario, q):
    df, n = q["df"], q["n_seeds"]
    fig, ax = plt.subplots(figsize=(8, 5))
    ep = df["Episode"].to_numpy()
    mean = df["qmix_mean"].to_numpy()
    std = df["qmix_std"].to_numpy()
    ax.plot(ep, mean, marker="o", color="C3", label="QMIX (CTDE)")
    ax.fill_between(ep, mean - std, mean + std, alpha=0.18, color="C3")
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Evaluation return (higher = better)")
    ax.set_title(f"{scenario}: QMIX learning curve (mean ± std over {n} seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(AGG_DIR, f"{scenario}_qmix.png"), dpi=140)
    plt.close(fig)


# Number of trailing eval points that define the "late-window" (converged-plateau)
# metric. With 50 eval points at every-10-epoch spacing this is the last ~10% of
# training (epochs ~460-500).
LATE_WINDOW = 5


def _seed_metrics(mat):
    """Per-seed final / late-window / peak return -> (mean, std) across seeds for each.

    mat is (n_seeds, n_points). Computing per seed first, then mean/std across seeds,
    is the statistically honest way to get a std for the late-window and peak metrics.
      * final : value at the last eval point.
      * late  : per-seed mean over the last LATE_WINDOW eval points (fair plateau metric).
      * peak  : per-seed best single eval point (optimistic upper bound).
    """
    w = min(LATE_WINDOW, mat.shape[1])
    final = mat[:, -1]
    late = np.nanmean(mat[:, -w:], axis=1)
    peak = np.nanmax(mat, axis=1)

    def ms(a):
        return float(np.nanmean(a)), float(np.nanstd(a))

    return {"final": ms(final), "late": ms(late), "peak": ms(peak)}


def _fmt(mean_std):
    mean, std = mean_std
    return f"{mean:,.0f} ± {std:,.0f}"


def _draw_bars(bar_data, scenarios, fname, ylabel, title):
    """Grouped bar chart of delay = |mean return| with std error bars."""
    algo_order = ["IQL", "Hysteretic", "VDN", "QMIX",
                  "Fixed-Time", "Fixed (native)", "Fixed (round-robin)"]
    present = [a for a in algo_order if any(a in bar_data[s] for s in scenarios)]
    if not (scenarios and present):
        return
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(scenarios)), 5.5))
    n_algos = len(present)
    width = 0.8 / n_algos
    x = np.arange(len(scenarios))
    for i, algo in enumerate(present):
        means = [bar_data[s].get(algo, (np.nan, np.nan))[0] for s in scenarios]
        stds = [bar_data[s].get(algo, (np.nan, 0.0))[1] for s in scenarios]
        ax.bar(x + i * width, means, width, yerr=stds, capsize=3, label=algo)
    ax.set_xticks(x + width * (n_algos - 1) / 2)
    ax.set_xticklabels(scenarios, rotation=15)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(AGG_DIR, fname), dpi=140)
    plt.close(fig)


def build_summary_and_bars(tab_results, qmix_results):
    """summary.md (final / late-window / peak) + final and late-window bar charts."""
    rows = []                # (scenario, label, n, {metric: (mean, std)})
    bar_final, bar_late = {}, {}

    for scenario in ALL_SCENARIOS:
        tab = tab_results.get(scenario)
        if tab is None:
            continue
        cols, n, mats = tab["cols"], tab["n_seeds"], tab["mats"]
        bar_final[scenario], bar_late[scenario] = {}, {}
        for col in cols:
            label = TAB_ALGO_LABELS.get(col, col)
            m = _seed_metrics(mats[col])
            rows.append((scenario, label, n, m))
            bar_final[scenario][label] = (abs(m["final"][0]), m["final"][1])
            bar_late[scenario][label] = (abs(m["late"][0]), m["late"][1])
        q = qmix_results.get(scenario)
        if q is not None:
            m = _seed_metrics(q["mat"])
            rows.append((scenario, "QMIX", q["n_seeds"], m))
            bar_final[scenario]["QMIX"] = (abs(m["final"][0]), m["final"][1])
            bar_late[scenario]["QMIX"] = (abs(m["late"][0]), m["late"][1])

    # summary.md (one row per scenario/algo, three metrics side by side)
    os.makedirs(AGG_DIR, exist_ok=True)
    with open(os.path.join(AGG_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("# Multi-seed aggregated results (evaluation return, mean ± std across seeds)\n\n")
        f.write("Return = sum of system negative waiting time over the eval episode "
                "(higher / less-negative is better).\n\n")
        f.write(f"- **Final** = last eval epoch.\n")
        f.write(f"- **Late-window** = mean over the last {LATE_WINDOW} evals "
                "(converged plateau; the fair head-to-head metric — robust to late ε-greedy noise).\n")
        f.write(f"- **Peak** = best single eval per seed (optimistic upper bound).\n\n")
        f.write("| Scenario | Algorithm | Final | Late-window | Peak | n seeds |\n")
        f.write("| :--- | :--- | ---: | ---: | ---: | ---: |\n")
        for scenario, label, n, m in rows:
            f.write(f"| {scenario} | {label} | {_fmt(m['final'])} | "
                    f"{_fmt(m['late'])} | {_fmt(m['peak'])} | {n} |\n")

    scenarios = [s for s in ALL_SCENARIOS if s in bar_final]
    _draw_bars(bar_final, scenarios, "cross_algorithm_bars.png",
               "Final delay = |return|  (lower is better)",
               "Final-epoch comparison across scenarios (mean ± std over seeds)")
    _draw_bars(bar_late, scenarios, "cross_algorithm_bars_late.png",
               f"Late-window delay = |return|  (lower is better)",
               f"Converged-plateau comparison (last {LATE_WINDOW} evals, mean ± std over seeds)")


def main():
    if not os.path.isdir(SEED_DIR):
        print(f"No seed directory at {SEED_DIR}. Run run_seeded_experiment.py first.")
        return
    os.makedirs(AGG_DIR, exist_ok=True)

    tab_results, qmix_results = {}, {}
    for scenario in ALL_SCENARIOS:
        tab = aggregate_tabular(scenario)
        if tab is not None:
            tab_results[scenario] = tab
            plot_tabular_learning(scenario, tab)
            print(f"[{scenario}] tabular: {tab['n_seeds']} seeds aggregated.")
        # QMIX now runs on RESCO scenarios too; aggregate wherever per-seed CSVs exist.
        q = aggregate_qmix(scenario)
        if q is not None:
            qmix_results[scenario] = q
            plot_qmix_learning(scenario, q)
            print(f"[{scenario}] qmix: {q['n_seeds']} seeds aggregated.")

    if tab_results:
        build_summary_and_bars(tab_results, qmix_results)
        print(f"Wrote aggregated outputs to {AGG_DIR}/ (summary.md, cross_algorithm_bars.png, per-scenario PNGs/CSVs).")
    else:
        print("No per-seed CSVs found to aggregate.")


if __name__ == "__main__":
    main()
