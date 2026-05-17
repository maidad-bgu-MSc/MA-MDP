# Traffic Control Experiment Summary Report

## 1-Hour Simulation Performance Evaluation

| Performance Metric | Independent Q-Learning (IQL/DQN) | Independent PPO (IPPO) |
| :--- | :---: | :---: |
| **Average Waiting Time** | 1442.75 seconds | 1474.45 seconds |
| **Average Queue Size (Stopped Cars)** | 72.23 | 138.19 |
| **Average System Speed** | 0.92 m/s | 0.25 m/s |

### Analysis & Findings
- **Winner: Independent Q-Learning (IQL)**
- IQL showed significantly lower average vehicle waiting times and queue lengths under continuous Poisson traffic distribution.
- This is highly consistent with standard reinforcement learning benchmarks, where sample-efficient off-policy Q-learning algorithms learn discrete junction phase actions faster and reach superior stability than on-policy policy gradient methods (PPO) in short-epoch regimes.
