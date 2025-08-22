# Lesson 4: Training with Reinforcement Learning

## RL Fundamentals

In RL, an agent learns by trial and error:
1. Agent observes state `s`
2. Agent takes action `a` based on policy π(a|s)
3. Environment returns reward `r` and next state `s'`
4. Agent updates policy to maximize cumulative reward

## Proximal Policy Optimization (PPO)

PPO is the most popular algorithm for robotics because:
- Stable training
- Good sample efficiency
- Works well with continuous actions

### The Math Behind PPO

PPO optimizes the policy by maximizing:

**L = E[min(r(θ)Â, clip(r(θ), 1-ε, 1+ε)Â)]**

Where:
- r(θ) = π(a|s) / π_old(a|s) (probability ratio)
- Â = advantage estimate (how much better than average)
- ε = clipping parameter (usually 0.2)

The clipping prevents large policy updates that could destabilize training.

## Setting Up Training

First, install Stable Baselines3:
```bash
pip install stable-baselines3
```

## Basic Training Script

```python
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch

# Import your custom environment
from go1_env import Go1Env

def make_env(xml_path, rank=0):
    """
    Factory function for parallel environments
    """
    def _init():
        env = Go1Env(xml_path)
        env.reset(seed=rank)
        return env
    return _init

def train():
    # Path to your robot model
    xml_path = "/path/to/go1/scene.xml"
    
    # Create parallel environments (faster training)
    num_envs = 4
    env = SubprocVecEnv([make_env(xml_path, i) for i in range(num_envs)])
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        learning_rate=3e-4,
        n_steps=2048,  # Steps per update
        batch_size=64,
        n_epochs=10,  # PPO epochs
        gamma=0.99,  # Discount factor
        gae_lambda=0.95,  # GAE parameter
        clip_range=0.2,  # PPO clipping
        verbose=1,
        tensorboard_log="./tensorboard_logs/",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Training
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=CheckpointCallback(
            save_freq=10000,
            save_path="./models/",
            name_prefix="go1_walk"
        )
    )
    
    # Save final model
    model.save("go1_final")
    
    return model
```

## Understanding the Policy Network

The policy network is a neural network that maps observations to actions:

```python
# Simplified view of what's inside "MlpPolicy"
class PolicyNetwork(torch.nn.Module):
    def __init__(self, obs_dim=46, action_dim=12):
        super().__init__()
        
        # Actor network (outputs actions)
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, action_dim)
        )
        
        # Critic network (estimates value)
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 256),
            torch.nn.Tanh(),
            torch.nn.Linear(256, 1)
        )
        
    def forward(self, obs):
        return self.actor(obs), self.critic(obs)
```

## Monitoring Training

Use TensorBoard to monitor progress:

```python
# In terminal:
# tensorboard --logdir ./tensorboard_logs/

# Key metrics to watch:
# - ep_rew_mean: Average episode reward (should increase)
# - ep_len_mean: Average episode length
# - loss: Policy loss (should decrease)
# - explained_variance: How well value function predicts returns
```

## Testing Your Trained Policy

```python
import mujoco
from mujoco import viewer
import numpy as np
from stable_baselines3 import PPO
from go1_env import Go1Env

def test_policy(model_path):
    # Load environment and model
    env = Go1Env("/path/to/go1/scene.xml")
    model = PPO.load(model_path)
    
    # Run episodes
    for episode in range(5):
        obs, _ = env.reset()
        total_reward = 0
        
        for step in range(1000):
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                print(f"Episode {episode}: Total reward = {total_reward:.2f}")
                break

def visualize_policy(model_path):
    """
    Visualize trained policy in MuJoCo viewer
    """
    model = PPO.load(model_path)
    
    mj_model = mujoco.MjModel.from_xml_path("/path/to/go1/scene.xml")
    mj_data = mujoco.MjData(mj_model)
    
    # Create environment for observations
    env = Go1Env("/path/to/go1/scene.xml")
    obs, _ = env.reset()
    
    with viewer.launch_passive(mj_model, mj_data) as v:
        while v.is_running():
            # Get action from policy
            action, _ = model.predict(obs, deterministic=True)
            
            # Apply action
            mj_data.ctrl[:] = action * 20.0
            
            # Step simulation
            mujoco.mj_step(mj_model, mj_data)
            
            # Update observation
            obs = env._get_observation()
            
            v.sync()
```

## Common Training Issues and Solutions

### 1. Robot Learns to Fall Forward
**Problem**: Robot gets reward for forward velocity even when falling
**Solution**: Add alive bonus, penalize low height more

```python
def _get_reward(self):
    # Alive bonus
    alive_bonus = 1.0
    
    # Forward velocity (only if upright)
    if self.data.qpos[2] > 0.25:
        forward_reward = self.data.qvel[0]
    else:
        forward_reward = 0
    
    return alive_bonus + forward_reward
```

### 2. Shaky/Jittery Motion
**Problem**: Actions change too rapidly
**Solution**: Add action smoothness penalty

```python
def _get_reward(self):
    # ... other rewards ...
    
    # Penalize action changes
    if hasattr(self, 'last_action'):
        smoothness_penalty = -0.1 * np.sum((action - self.last_action)**2)
    else:
        smoothness_penalty = 0
    
    self.last_action = action.copy()
```

### 3. Not Learning
**Problem**: Reward not increasing
**Solutions**:
- Simplify reward function
- Reduce action/observation space
- Curriculum learning (start with easier task)
- Check observation normalization

## Hyperparameter Tuning

Key parameters to experiment with:

```python
# Learning rate: Too high = unstable, Too low = slow learning
learning_rate = 3e-4  # Try: 1e-4 to 1e-3

# Batch size: Larger = more stable, needs more memory
batch_size = 64  # Try: 32, 64, 128

# Discount factor: How much to value future rewards
gamma = 0.99  # Try: 0.95 to 0.99

# GAE lambda: Bias-variance tradeoff in advantage estimation
gae_lambda = 0.95  # Try: 0.9 to 0.98

# PPO epochs: How many times to reuse data
n_epochs = 10  # Try: 5 to 20

# Clip range: How much policy can change
clip_range = 0.2  # Try: 0.1 to 0.3
```

## Exercises
1. Train a basic walking policy
2. Monitor training with TensorBoard
3. Experiment with different reward functions
4. Try different network architectures (add layers, change sizes)
5. Implement curriculum learning (gradually increase difficulty)

## Key Takeaways
- PPO is the go-to algorithm for robot learning
- Good reward design is crucial
- Monitor training metrics to diagnose issues
- Start simple and gradually add complexity
- Training takes time and experimentation
