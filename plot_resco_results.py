import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_resco_tabular_learning_curves():
    RESCO_SCENARIOS = ["cologne3", "grid4x4"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {"Tabular_IQL_Reward": "#10b981", "Hysteretic_Reward": "#6366f1", "VDN_Reward": "#ec4899"}
    labels = {"Tabular_IQL_Reward": "Tabular IQL", "Hysteretic_Reward": "Hysteretic Q", "VDN_Reward": "VDN"}

    for ax, scenario in zip(axes, RESCO_SCENARIOS):
        csv_path = os.path.join("outputs", f"training_evaluation_log_{scenario}.csv")
        if not os.path.exists(csv_path):
            ax.set_title(f"{scenario} (no data)")
            continue
        df = pd.read_csv(csv_path)
        for col, color in colors.items():
            if col in df.columns:
                ax.plot(df["Epoch"], df[col].abs(), color=color, label=labels[col], linewidth=1.8)
        if "Fixed_Baseline_Reward" in df.columns:
            ax.axhline(df["Fixed_Baseline_Reward"].abs().iloc[0], color="#ef4444",
                       linestyle="--", linewidth=1.4, label="Fixed-Time")
        ax.set_yscale("log")
        ax.set_title(scenario.upper(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Delay (log, lower=better)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.2)

    plt.suptitle("RESCO Tabular Algorithms: Learning Curves per Scenario", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/resco_tabular_learning_curves_cologne.png", dpi=300)
    plt.close()
    print("Saved outputs/resco_tabular_learning_curves_cologne.png")

def plot_resco_cross_algorithm_bar():
    RESCO_SCENARIOS = ["cologne3", "grid4x4"]
    algo_cols = {
        "Tabular IQL": "Tabular_IQL_Reward",
        "Hysteretic Q": "Hysteretic_Reward",
        "VDN": "VDN_Reward",
    }
    algo_colors = {
        "Tabular IQL": "#10b981",
        "Hysteretic Q": "#6366f1",
        "VDN": "#ec4899",
        "Fixed-Time": "#ef4444",
    }

    # Collect best-performance rewards per algorithm per scenario
    data = {alg: [] for alg in list(algo_cols.keys()) + ["Fixed-Time"]}
    valid_scenarios = []

    for scenario in RESCO_SCENARIOS:
        csv_path = os.path.join("outputs", f"training_evaluation_log_{scenario}.csv")
        if not os.path.exists(csv_path):
            continue
        valid_scenarios.append(scenario.upper())
        df = pd.read_csv(csv_path)
        for alg, col in algo_cols.items():
            data[alg].append(abs(df[col].max()) if col in df.columns else 0)
        if "Fixed_Baseline_Reward" in df.columns:
            data["Fixed-Time"].append(abs(df["Fixed_Baseline_Reward"].max()))
        else:
            data["Fixed-Time"].append(0)

    if not valid_scenarios:
        print("No RESCO tabular log files found — skipping bar chart.")
        return

    x = np.arange(len(valid_scenarios))
    algos = list(algo_colors.keys())
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, alg in enumerate(algos):
        offset = (i - len(algos) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[alg], width, label=alg,
                      color=algo_colors[alg],
                      linestyle="--" if alg == "Fixed-Time" else "-",
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_scenarios, fontsize=11)
    ax.set_ylabel("Total Delay (lower is better)", fontsize=12)
    ax.set_title("RESCO Cross-Algorithm Comparison (Best Performance)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/resco_cross_algorithm_bar_cologne.png", dpi=300)
    plt.close()
    print("Saved outputs/resco_cross_algorithm_bar_cologne.png")

if __name__ == "__main__":
    try:
        plot_resco_tabular_learning_curves()
    except Exception as e:
        print(f"plot_resco_tabular_learning_curves skipped: {e}")

    try:
        plot_resco_cross_algorithm_bar()
    except Exception as e:
        print(f"plot_resco_cross_algorithm_bar skipped: {e}")
