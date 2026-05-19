import os
from watch_agents import FixedTimeController
from run_tabular_experiment import evaluate_agents

def run_baseline_evaluation():
    print("Evaluating 5 Different Fixed-Time Baselines (600 seconds each)\n")
    
    agent_ids = ["A0", "B0", "C0", "D0"]
    results = {}
    
    baselines = [
        {
            "name": "Policy 1: 50/50 Split, No Offsets (Naive)",
            "ew": 10, "ns": 10,
            "offsets": {"A0": 0, "B0": 0, "C0": 0, "D0": 0}
        },
        {
            "name": "Policy 2: 80/20 Proportional Split, No Offsets",
            "ew": 24, "ns": 6,
            "offsets": {"A0": 0, "B0": 0, "C0": 0, "D0": 0}
        },
        {
            "name": "Policy 3: 80/20 Split, Green Wave (Staggered Offsets)",
            "ew": 24, "ns": 6,
            "offsets": {"A0": 0, "B0": 3, "C0": 6, "D0": 9}
        },
        {
            "name": "Policy 4: Short Cycle (60s EW / 15s NS), No Offsets",
            "ew": 12, "ns": 3,
            "offsets": {"A0": 0, "B0": 0, "C0": 0, "D0": 0}
        },
        {
            "name": "Policy 5: Short Cycle (60s EW / 15s NS), Green Wave",
            "ew": 12, "ns": 3,
            "offsets": {"A0": 0, "B0": 3, "C0": 6, "D0": 9}
        }
    ]
    
    for b in baselines:
        print(f"Running {b['name']}...")
        agents = {
            agent: FixedTimeController(agent, ew_steps=b["ew"], ns_steps=b["ns"], offset_steps=b["offsets"][agent])
            for agent in agent_ids
        }
        reward = evaluate_agents(agents, algo_name=b["name"], sim_seconds=600)
        results[b["name"]] = reward
        
    print("\n" + "="*50)
    print("BASELINE COMPARISON RESULTS")
    print("="*50)
    
    # Sort results from best (least negative) to worst
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    
    for i, (name, reward) in enumerate(sorted_results):
        print(f"{i+1}. {name}: {reward:.2f}")

if __name__ == "__main__":
    run_baseline_evaluation()
