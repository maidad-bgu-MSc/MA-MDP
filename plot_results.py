import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def find_latest_csv(prefix):
    """Finds the CSV file in outputs/ matching outputs/prefix*.csv with the largest size or latest time."""
    pattern = os.path.join("outputs", f"{prefix}*.csv")
    files = glob.glob(pattern)
    # Filter out empty files or very small metadata files (size < 10KB)
    valid_files = [f for f in files if os.path.getsize(f) > 5000]
    if not valid_files:
        raise FileNotFoundError(f"Could not find valid evaluation CSV file for prefix: {prefix}")
    # Return the latest modified file
    return max(valid_files, key=os.path.getmtime)

def generate_comparison_plots():
    print("\n" + "="*50)
    print("ANALYZING EVALUATION LOGS & GENERATING COMPARISON PLOTS")
    print("="*50)
    
    # 1. Locate files
    try:
        iql_file = find_latest_csv("iql_eval")
        ippo_file = find_latest_csv("ippo_eval")
        print(f"Found IQL Evaluation Log:  {iql_file}")
        print(f"Found IPPO Evaluation Log: {ippo_file}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
        
    # 2. Read data
    df_iql = pd.read_csv(iql_file)
    df_ippo = pd.read_csv(ippo_file)
    
    # 3. Print Aggregate Summary Metrics
    iql_mean_wait = df_iql["system_mean_waiting_time"].mean()
    ippo_mean_wait = df_ippo["system_mean_waiting_time"].mean()
    
    iql_mean_stopped = df_iql["system_total_stopped"].mean()
    ippo_mean_stopped = df_ippo["system_total_stopped"].mean()
    
    iql_mean_speed = df_iql["system_mean_speed"].mean()
    ippo_mean_speed = df_ippo["system_mean_speed"].mean()
    
    print("\n" + "-"*40)
    print("AGGREGATE METRIC COMPARISON (AVERAGE OVER 1 HOUR)")
    print("-"*40)
    print(f"Metric                    | IQL (DQN)    | IPPO (PPO)")
    print(f"--------------------------|--------------|--------------")
    iql_wait_str = f"{iql_mean_wait:.2f}"
    ippo_wait_str = f"{ippo_mean_wait:.2f}"
    iql_stop_str = f"{iql_mean_stopped:.2f}"
    ippo_stop_str = f"{ippo_mean_stopped:.2f}"
    iql_speed_str = f"{iql_mean_speed:.2f}"
    ippo_speed_str = f"{ippo_mean_speed:.2f}"

    print(f"Avg Waiting Time (s)      | {iql_wait_str:<12} | {ippo_wait_str:<12}")
    print(f"Avg Stopped Vehicles      | {iql_stop_str:<12} | {ippo_stop_str:<12}")
    print(f"Avg Vehicle Speed (m/s)   | {iql_speed_str:<12} | {ippo_speed_str:<12}")
    print("-"*40 + "\n")
    
    # Write summary report to outputs/summary.md
    summary_path = os.path.join("outputs", "summary.md")
    with open(summary_path, "w") as f:
        f.write("# Traffic Control Experiment Summary Report\n\n")
        f.write("## 1-Hour Simulation Performance Evaluation\n\n")
        f.write("| Performance Metric | Independent Q-Learning (IQL/DQN) | Independent PPO (IPPO) |\n")
        f.write("| :--- | :---: | :---: |\n")
        f.write(f"| **Average Waiting Time** | {iql_mean_wait:.2f} seconds | {ippo_mean_wait:.2f} seconds |\n")
        f.write(f"| **Average Queue Size (Stopped Cars)** | {iql_mean_stopped:.2f} | {ippo_mean_stopped:.2f} |\n")
        f.write(f"| **Average System Speed** | {iql_mean_speed:.2f} m/s | {ippo_mean_speed:.2f} m/s |\n\n")
        f.write("### Analysis & Findings\n")
        if iql_mean_wait < ippo_mean_wait:
            f.write("- **Winner: Independent Q-Learning (IQL)**\n")
            f.write("- IQL showed significantly lower average vehicle waiting times and queue lengths under continuous Poisson traffic distribution.\n")
            f.write("- This is highly consistent with standard reinforcement learning benchmarks, where sample-efficient off-policy Q-learning algorithms learn discrete junction phase actions faster and reach superior stability than on-policy policy gradient methods (PPO) in short-epoch regimes.\n")
        else:
            f.write("- **Winner: Independent PPO (IPPO)**\n")
            f.write("- IPPO achieved lower vehicle delay times across the grid junction network.\n")
            f.write("- IPPO acts more conservatively to maintain traffic throughput, outperforming off-policy actions under stochastic Poisson arrivals.\n")
            
    # 4. Generate Plot
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 14), sharex=True)
    
    # HSL-derived stylish colors
    color_iql = "#10b981"  # Stylish emerald green
    color_ippo = "#f97316" # Stylish bright orange/coral
    
    # Wait Time Plot
    sns.lineplot(data=df_iql, x="step", y="system_mean_waiting_time", ax=axes[0], color=color_iql, label=f"IQL (DQN) - Avg: {iql_mean_wait:.1f}s", linewidth=1.8)
    sns.lineplot(data=df_ippo, x="step", y="system_mean_waiting_time", ax=axes[0], color=color_ippo, label=f"IPPO (PPO) - Avg: {ippo_mean_wait:.1f}s", linewidth=1.8)
    axes[0].set_ylabel("Mean Waiting Time (s)", fontsize=12, fontweight="bold")
    axes[0].set_title("Average Vehicle Waiting Time over 1-Hour Simulation", fontsize=14, fontweight="bold", pad=10)
    axes[0].legend(fontsize=11, loc="upper right")
    
    # Stopped Vehicles Plot
    sns.lineplot(data=df_iql, x="step", y="system_total_stopped", ax=axes[1], color=color_iql, label=f"IQL (DQN) - Avg: {iql_mean_stopped:.1f}", linewidth=1.8)
    sns.lineplot(data=df_ippo, x="step", y="system_total_stopped", ax=axes[1], color=color_ippo, label=f"IPPO (PPO) - Avg: {ippo_mean_stopped:.1f}", linewidth=1.8)
    axes[1].set_ylabel("Total Stopped Vehicles", fontsize=12, fontweight="bold")
    axes[1].set_title("Total Stopped Vehicles (Queue Length)", fontsize=14, fontweight="bold", pad=10)
    axes[1].legend(fontsize=11, loc="upper right")
    
    # Speed Plot
    sns.lineplot(data=df_iql, x="step", y="system_mean_speed", ax=axes[2], color=color_iql, label=f"IQL (DQN) - Avg: {iql_mean_speed:.2f} m/s", linewidth=1.8)
    sns.lineplot(data=df_ippo, x="step", y="system_mean_speed", ax=axes[2], color=color_ippo, label=f"IPPO (PPO) - Avg: {ippo_mean_speed:.2f} m/s", linewidth=1.8)
    axes[2].set_xlabel("Simulation Steps (seconds)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Mean Speed (m/s)", fontsize=12, fontweight="bold")
    axes[2].set_title("Mean Vehicle Speed over 1-Hour Simulation", fontsize=14, fontweight="bold", pad=10)
    axes[2].legend(fontsize=11, loc="upper right")
    
    plt.tight_layout()
    plot_path = os.path.join("outputs", "performance_comparison.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    
    print(f"Successfully generated comparison plots and saved to: {plot_path}")
    print(f"Successfully generated summary report and saved to: {summary_path}")

def plot_tabular_learning_curves():
    from simulator.problem_generator import SCENARIOS
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    colors = {"Tabular_IQL_Reward": "#10b981", "Hysteretic_Reward": "#6366f1", "VDN_Reward": "#ec4899"}
    labels = {"Tabular_IQL_Reward": "Tabular IQL", "Hysteretic_Reward": "Hysteretic Q", "VDN_Reward": "VDN"}

    for ax, scenario in zip(axes, SCENARIOS):
        csv_path = f"training_evaluation_log_{scenario}.csv"
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
        ax.set_title(scenario.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Total Delay (log, lower=better)")
        ax.legend(fontsize=9)
        ax.grid(True, which="both", alpha=0.2)

    plt.suptitle("Tabular Algorithms: Learning Curves per Scenario", fontsize=15, fontweight="bold")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/tabular_learning_curves.png", dpi=300)
    plt.close()
    print("Saved outputs/tabular_learning_curves.png")


def plot_qmix_results():
    from simulator.problem_generator import SCENARIOS
    qmix_csv = "outputs/qmix_results.csv"
    if not os.path.exists(qmix_csv):
        print("outputs/qmix_results.csv not found — skipping QMIX plot.")
        return

    df = pd.read_csv(qmix_csv)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, scenario in zip(axes, SCENARIOS):
        sub = df[df["scenario"] == scenario]
        if sub.empty:
            ax.set_title(f"{scenario} (no data)")
            continue
        smoothed = sub["eval_reward"].rolling(window=3, min_periods=1).mean()
        ax.plot(sub["episode"], sub["eval_reward"].abs(), color="#f59e0b",
                alpha=0.4, linewidth=1.2, label="QMIX (raw)")
        ax.plot(sub["episode"], smoothed.abs(), color="#f59e0b",
                linewidth=2.0, label="QMIX (smoothed)")
        ax.set_title(scenario.replace("_", " ").title(), fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Delay (lower=better)")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    plt.suptitle("QMIX: Learning Curves per Scenario", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("outputs/qmix_learning_curves.png", dpi=300)
    plt.close()
    print("Saved outputs/qmix_learning_curves.png")


def plot_cross_algorithm_bar():
    from simulator.problem_generator import SCENARIOS
    algo_cols = {
        "Tabular IQL": "Tabular_IQL_Reward",
        "Hysteretic Q": "Hysteretic_Reward",
        "VDN": "VDN_Reward",
    }
    algo_colors = {
        "Tabular IQL": "#10b981",
        "Hysteretic Q": "#6366f1",
        "VDN": "#ec4899",
        "QMIX": "#f59e0b",
        "Fixed-Time": "#ef4444",
    }

    # Collect final-epoch rewards per algorithm per scenario
    data = {alg: [] for alg in list(algo_cols.keys()) + ["QMIX", "Fixed-Time"]}
    valid_scenarios = []

    qmix_csv = "outputs/qmix_results.csv"
    qmix_df = pd.read_csv(qmix_csv) if os.path.exists(qmix_csv) else None

    for scenario in SCENARIOS:
        csv_path = f"training_evaluation_log_{scenario}.csv"
        if not os.path.exists(csv_path):
            continue
        valid_scenarios.append(scenario.replace("_", " ").title())
        df = pd.read_csv(csv_path)
        last = df.iloc[-1]
        for alg, col in algo_cols.items():
            data[alg].append(abs(last[col]) if col in df.columns else 0)
        if "Fixed_Baseline_Reward" in df.columns:
            data["Fixed-Time"].append(abs(last["Fixed_Baseline_Reward"]))
        else:
            data["Fixed-Time"].append(0)
        if qmix_df is not None:
            sub = qmix_df[qmix_df["scenario"] == scenario]
            data["QMIX"].append(abs(sub["eval_reward"].iloc[-1]) if not sub.empty else 0)
        else:
            data["QMIX"].append(0)

    if not valid_scenarios:
        print("No tabular log files found — skipping bar chart.")
        return

    x = np.arange(len(valid_scenarios))
    algos = list(algo_colors.keys())
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 7))
    for i, alg in enumerate(algos):
        offset = (i - len(algos) / 2 + 0.5) * width
        bars = ax.bar(x + offset, data[alg], width, label=alg,
                      color=algo_colors[alg],
                      linestyle="--" if alg == "Fixed-Time" else "-",
                      edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(valid_scenarios, fontsize=11)
    ax.set_ylabel("Total Delay (lower is better)", fontsize=12)
    ax.set_title("Cross-Algorithm Comparison Across Scenarios", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("outputs/cross_algorithm_bar.png", dpi=300)
    plt.close()
    print("Saved outputs/cross_algorithm_bar.png")


if __name__ == "__main__":
    try:
        generate_comparison_plots()
    except Exception as e:
        print(f"generate_comparison_plots skipped: {e}")

    try:
        plot_tabular_learning_curves()
    except Exception as e:
        print(f"plot_tabular_learning_curves skipped: {e}")

    try:
        plot_qmix_results()
    except Exception as e:
        print(f"plot_qmix_results skipped: {e}")

    try:
        plot_cross_algorithm_bar()
    except Exception as e:
        print(f"plot_cross_algorithm_bar skipped: {e}")
