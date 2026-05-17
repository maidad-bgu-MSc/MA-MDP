import os
import torch
import torch.nn as nn
import numpy as np

# Setup SUMO paths
from env_setup import setup_sumo_env, make_env
setup_sumo_env()

# Tianshou imports
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.data import Collector
from tianshou.algorithm import DQN, PPO
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.ppo import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm, MultiAgentOnPolicyAlgorithm

# Define DQN Network
class QNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape)
        )
        
    def forward(self, obs, state=None, info={}):
        if hasattr(obs, "obs"):
            obs = obs.obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        q = self.model(obs)
        return q, state

# Define PPO Networks
class ActorNet(nn.Module):
    def __init__(self, state_shape, action_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape)
        )
        
    def forward(self, obs, state=None, info={}):
        if hasattr(obs, "obs"):
            obs = obs.obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        logits = self.model(obs)
        return logits, state

class CriticNet(nn.Module):
    def __init__(self, state_shape):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, obs, state=None, info={}):
        if hasattr(obs, "obs"):
            obs = obs.obs
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        val = self.model(obs)
        return val

def evaluate_iql(num_seconds=3600):
    print("\n" + "="*50)
    print("STARTING INDEPENDENT Q-LEARNING (IQL) DETERMINISTIC EVALUATION")
    print("="*50)
    
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize evaluation environment with output CSV name
    raw_env = make_env(num_seconds=num_seconds, out_csv_name="outputs/iql_eval")
    env = PettingZooEnv(raw_env)
    
    eval_envs = DummyVectorEnv([lambda: PettingZooEnv(make_env(num_seconds=num_seconds, out_csv_name="outputs/iql_eval"))])
    
    # Create DQN agents
    dqn_agents = []
    for agent in env.agents:
        net = QNet(state_shape=2, action_shape=2)
        policy = DiscreteQLearningPolicy(
            model=net,
            action_space=env.action_space,
            eps_training=0.0,
            eps_inference=0.0
        )
        optim_factory = AdamOptimizerFactory(lr=1e-3)
        dqn = DQN(policy=policy, optim=optim_factory)
        dqn_agents.append(dqn)
        
    # Multi-Agent Manager
    manager = MultiAgentOffPolicyAlgorithm(algorithms=dqn_agents, env=env)
    
    # Load trained weights
    print("Loading IQL weights from models/iql_policy.pth...")
    manager.load_state_dict(torch.load("models/iql_policy.pth"), strict=False)
    
    # Collect 1 deterministic episode
    collector = Collector(manager, eval_envs)
    stats = collector.collect(n_episode=1, reset_before_collect=True)
    
    print("IQL Evaluation Stats:")
    print(stats)
    raw_env.close()

def evaluate_ippo(num_seconds=3600):
    print("\n" + "="*50)
    print("STARTING INDEPENDENT PPO (IPPO) DETERMINISTIC EVALUATION")
    print("="*50)
    
    os.makedirs("outputs", exist_ok=True)
    
    # Initialize evaluation environment with output CSV name
    raw_env = make_env(num_seconds=num_seconds, out_csv_name="outputs/ippo_eval")
    env = PettingZooEnv(raw_env)
    
    eval_envs = DummyVectorEnv([lambda: PettingZooEnv(make_env(num_seconds=num_seconds, out_csv_name="outputs/ippo_eval"))])
    
    # Create PPO agents
    ppo_agents = []
    for agent in env.agents:
        actor = ActorNet(state_shape=2, action_shape=2)
        critic = CriticNet(state_shape=2)
        actor_policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
            action_space=env.action_space,
            action_scaling=False,
            deterministic_eval=True # argmax actions
        )
        optim_factory = AdamOptimizerFactory(lr=5e-4)
        ppo = PPO(policy=actor_policy, critic=critic, optim=optim_factory)
        ppo_agents.append(ppo)
        
    # Multi-Agent Manager
    manager = MultiAgentOnPolicyAlgorithm(algorithms=ppo_agents, env=env)
    
    # Load trained weights
    print("Loading IPPO weights from models/ippo_policy.pth...")
    manager.load_state_dict(torch.load("models/ippo_policy.pth"), strict=False)
    
    # Collect 1 deterministic episode
    collector = Collector(manager, eval_envs)
    stats = collector.collect(n_episode=1, reset_before_collect=True)
    
    print("IPPO Evaluation Stats:")
    print(stats)
    raw_env.close()

if __name__ == "__main__":
    evaluate_iql(num_seconds=3600)
    evaluate_ippo(num_seconds=3600)
