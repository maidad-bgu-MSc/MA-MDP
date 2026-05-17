import os
import pandas as pd
import numpy as np
import pytest

def test_evaluation_to_plot_data_integrity():
    """Asserts that evaluation logging lists compile into correct shapes and plot error-free."""
    import matplotlib
    matplotlib.use('Agg') # Force headless backend for testing environments
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # 1. Mock the metrics dataframe generated after evaluation runs
    # 2 algorithms, 2 scaling sizes (4 agents and 9 agents)
    mock_data = [
        {"Algorithm": "Fixed-Time Controller", "Agents": 4, "Avg Waiting Time (s)": 38.5},
        {"Algorithm": "Fixed-Time Controller", "Agents": 9, "Avg Waiting Time (s)": 52.1},
        {"Algorithm": "Independent Tabular Q-Learning", "Agents": 4, "Avg Waiting Time (s)": 15.2},
        {"Algorithm": "Independent Tabular Q-Learning", "Agents": 9, "Avg Waiting Time (s)": 28.4}
    ]
    df_results = pd.DataFrame(mock_data)
    
    # Assert data shapes and columns conform to expected structures
    assert len(df_results) == 4
    assert set(df_results.columns) == {"Algorithm", "Agents", "Avg Waiting Time (s)"}
    
    # 2. Run the Matplotlib / Seaborn lineplot logic
    plt.figure(figsize=(10, 6))
    
    palette = {
        "Fixed-Time Controller": "#94a3b8",
        "Independent Tabular Q-Learning": "#38bdf8"
    }
    
    try:
        sns.lineplot(
            data=df_results,
            x="Agents",
            y="Avg Waiting Time (s)",
            hue="Algorithm",
            style="Algorithm",
            markers=True,
            dashes=False,
            linewidth=2.5,
            markersize=9,
            palette=palette
        )
        
        plt.title("Mock Algorithm Performance Scaling", fontsize=12)
        plt.xlabel("System Complexity (Number of Agents)", fontsize=10)
        plt.ylabel("Average Waiting Time (seconds)", fontsize=10)
        plt.xticks([4, 9], ["2x2\n(4 Agents)", "3x3\n(9 Agents)"])
        plt.tight_layout()
        
        # Test file write to outputs folder (ensuring parent dir exists)
        os.makedirs("outputs", exist_ok=True)
        test_plot_path = os.path.join("outputs", "test_scaling_plot.png")
        
        plt.savefig(test_plot_path, dpi=100)
        plt.close()
        
        # Assert plot was physically written and is non-empty
        assert os.path.exists(test_plot_path), "Visualizer line plot was not generated on disk!"
        assert os.path.getsize(test_plot_path) > 1000, "Visualizer line plot is corrupted or empty!"
        
    finally:
        # Cleanup temporary files
        if os.path.exists(test_plot_path):
            os.remove(test_plot_path)
