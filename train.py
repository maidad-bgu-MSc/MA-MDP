import os
import sys
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

# Setup SUMO paths
from simulator.env_setup import make_wave_env
import os

def make_env(num_seconds=1000):
    net_file = "wave_1x4.net.xml"
    rou_file = "wave_1x4.rou.xml"
    if not os.path.exists(net_file) or not os.path.exists(rou_file):
        from simulator.generate_1x4_wave import build_1x4_scenario
        build_1x4_scenario()
    return make_wave_env(net_file=net_file, route_file=rou_file, num_seconds=num_seconds)
# Tianshou imports
from tianshou.env import PettingZooEnv, DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.algorithm import DQN, PPO
from tianshou.algorithm.modelfree.dqn import DiscreteQLearningPolicy
from tianshou.algorithm.modelfree.ppo import ProbabilisticActorPolicy
from tianshou.algorithm.optim import AdamOptimizerFactory
from tianshou.algorithm.multiagent.marl import MultiAgentOffPolicyAlgorithm, MultiAgentOnPolicyAlgorithm
from tianshou.trainer import OffPolicyTrainer, OffPolicyTrainerParams, OnPolicyTrainer, OnPolicyTrainerParams

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

def train_iql(num_seconds=1000, epochs=5):
    """Trains the baseline Independent Q-Learning (IQL) policy using DQN in Tianshou v2."""
    print("\n" + "="*50)
    print("STARTING INDEPENDENT Q-LEARNING (IQL) TRAINING")
    print("="*50)
    
    # Initialize environment
    raw_env = make_env(num_seconds=num_seconds)
    env = PettingZooEnv(raw_env)
    
    train_envs = DummyVectorEnv([lambda: PettingZooEnv(make_env(num_seconds=num_seconds))])
    test_envs = DummyVectorEnv([lambda: PettingZooEnv(make_env(num_seconds=num_seconds))])
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create independent DQN algorithms for each agent
    dqn_agents = []
    agents = env.agents
    for agent in agents:
        net = QNet(state_shape=2, action_shape=2)
        # Tianshou v2: First create a policy class DiscreteQLearningPolicy
        policy = DiscreteQLearningPolicy(
            model=net,
            action_space=env.action_space,
            eps_training=0.1,
            eps_inference=0.0
        )
        # Create Optimizer Factory
        optim_factory = AdamOptimizerFactory(lr=1e-3)
        # Create DQN algorithm wrapping policy and optimizer
        dqn = DQN(
            policy=policy,
            optim=optim_factory,
            gamma=0.95,
            target_update_freq=50
        )
        dqn_agents.append(dqn)
        
    # Combine into Multi-Agent Off-Policy algorithm manager
    manager = MultiAgentOffPolicyAlgorithm(algorithms=dqn_agents, env=env)
    
    # Setup Collectors
    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = Collector(manager, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(manager, test_envs)
    
    # Pre-fill buffer with some random steps
    train_collector.collect(n_step=200, reset_before_collect=True)
    
    # Trainer Parameters
    os.makedirs("models", exist_ok=True)
    params = OffPolicyTrainerParams(
        max_epochs=epochs,
        epoch_num_steps=1000,
        training_collector=train_collector,
        test_collector=test_collector,
        collection_step_num_env_steps=10,
        batch_size=64,
        update_step_num_gradient_steps_per_sample=0.1,
        test_step_num_episodes=1,
        verbose=True
    )
    
    # Instantiate and run trainer
    trainer = OffPolicyTrainer(algorithm=manager, params=params)
    result = trainer.run()
    
    # Save IQL policies
    torch.save(manager.state_dict(), "models/iql_policy.pth")
    print("Successfully trained and saved IQL policy to models/iql_policy.pth!")
    print(result)
    raw_env.close()

def train_ippo(num_seconds=1000, epochs=5):
    """Trains the advanced Independent PPO (IPPO) policy using PPO in Tianshou v2."""
    print("\n" + "="*50)
    print("STARTING INDEPENDENT PPO (IPPO) TRAINING")
    print("="*50)
    
    # Initialize environment
    raw_env = make_env(num_seconds=num_seconds)
    env = PettingZooEnv(raw_env)
    
    train_envs = DummyVectorEnv([lambda: PettingZooEnv(make_env(num_seconds=num_seconds))])
    test_envs = DummyVectorEnv([lambda: PettingZooEnv(make_env(num_seconds=num_seconds))])
    
    # Set seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create independent PPO algorithms for each agent
    ppo_agents = []
    agents = env.agents
    for agent in agents:
        actor = ActorNet(state_shape=2, action_shape=2)
        critic = CriticNet(state_shape=2)
        
        # Tianshou v2: First create a policy class ProbabilisticActorPolicy
        actor_policy = ProbabilisticActorPolicy(
            actor=actor,
            dist_fn=lambda logits: torch.distributions.Categorical(logits=logits),
            action_space=env.action_space,
            action_scaling=False
        )
        
        # Create Optimizer Factory
        optim_factory = AdamOptimizerFactory(lr=5e-4)
        
        # Create PPO algorithm wrapping policy and critic
        ppo = PPO(
            policy=actor_policy,
            critic=critic,
            optim=optim_factory,
            gamma=0.95,
            advantage_normalization=True,
            recompute_advantage=True,
            eps_clip=0.2,
            value_clip=True
        )
        ppo_agents.append(ppo)
        
    # Combine into Multi-Agent On-Policy algorithm manager
    manager = MultiAgentOnPolicyAlgorithm(algorithms=ppo_agents, env=env)
    
    # Setup Collectors
    buffer = VectorReplayBuffer(total_size=10000, buffer_num=len(train_envs))
    train_collector = Collector(manager, train_envs, buffer)
    test_collector = Collector(manager, test_envs)
    
    # Trainer Parameters
    os.makedirs("models", exist_ok=True)
    params = OnPolicyTrainerParams(
        max_epochs=epochs,
        epoch_num_steps=1000,
        training_collector=train_collector,
        test_collector=test_collector,
        collection_step_num_env_steps=200, # Collect 200 transitions per iteration
        batch_size=64,
        update_step_num_repetitions=4,
        test_step_num_episodes=1,
        verbose=True
    )
    
    # Instantiate and run trainer
    trainer = OnPolicyTrainer(algorithm=manager, params=params)
    result = trainer.run()
    
    # Save IPPO policies
    torch.save(manager.state_dict(), "models/ippo_policy.pth")
    print("Successfully trained and saved IPPO policy to models/ippo_policy.pth!")
    print(result)
    raw_env.close()

if __name__ == "__main__":
    # Run training for 5 epochs on shorter simulation length (1000 seconds) for efficiency
    train_iql(num_seconds=1000, epochs=100)
    train_ippo(num_seconds=1000, epochs=100)
