# Lesson 5: Advanced RL Algorithms for Robotics

## Beyond PPO: Algorithm Zoo

While PPO is popular for robotics, understanding other algorithms helps you choose the right tool for your task.

## Soft Actor-Critic (SAC)

SAC is an off-policy algorithm that maximizes both reward and entropy (exploration).

### Mathematical Foundation

SAC optimizes:
**J(π) = E[Σ(r_t + αH(π(·|s_t)))]**

Where H is entropy, encouraging exploration.

### Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

class SACNetwork(nn.Module):
    """Actor-Critic networks for SAC"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Output mean and log_std
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        
        # Twin Q-networks (critics)
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward_actor(self, state):
        features = self.actor(state)
        mean = self.mean(features)
        log_std = self.log_std(features)
        log_std = torch.clamp(log_std, -20, 2)  # Stability
        
        return mean, log_std
    
    def sample_action(self, state, deterministic=False):
        mean, log_std = self.forward_actor(state)
        
        if deterministic:
            return torch.tanh(mean)
        
        # Reparameterization trick
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        
        # Squash to [-1, 1] with tanh
        action = torch.tanh(x)
        
        # Compute log probability with correction for tanh
        log_prob = normal.log_prob(x)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob
    
    def forward_critics(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

class SAC:
    def __init__(
        self,
        state_dim,
        action_dim,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        buffer_size=1000000
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.network = SACNetwork(state_dim, action_dim).to(self.device)
        self.target_network = SACNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(
            list(self.network.actor.parameters()) + 
            list(self.network.mean.parameters()) + 
            list(self.network.log_std.parameters()),
            lr=lr
        )
        self.q1_optimizer = torch.optim.Adam(self.network.q1.parameters(), lr=lr)
        self.q2_optimizer = torch.optim.Adam(self.network.q2.parameters(), lr=lr)
        
        # Hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Replay buffer
        self.buffer = deque(maxlen=buffer_size)
        
    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample_batch(self, batch_size=256):
        batch = random.sample(self.buffer, batch_size)
        
        states = torch.FloatTensor([t[0] for t in batch]).to(self.device)
        actions = torch.FloatTensor([t[1] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t[2] for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([t[3] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t[4] for t in batch]).unsqueeze(1).to(self.device)
        
        return states, actions, rewards, next_states, dones
    
    def update(self, batch_size=256):
        if len(self.buffer) < batch_size:
            return {}
            
        states, actions, rewards, next_states, dones = self.sample_batch(batch_size)
        
        # Update critics
        with torch.no_grad():
            next_actions, next_log_probs = self.network.sample_action(next_states)
            next_q1, next_q2 = self.target_network.forward_critics(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            
            # Include entropy in target
            target_q = rewards + self.gamma * (1 - dones) * (next_q - self.alpha * next_log_probs)
            
        current_q1, current_q2 = self.network.forward_critics(states, actions)
        
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # Update actor
        new_actions, log_probs = self.network.sample_action(states)
        q1_new, q2_new = self.network.forward_critics(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Soft update target network
        for param, target_param in zip(
            self.network.parameters(),
            self.target_network.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
            
        return {
            'q1_loss': q1_loss.item(),
            'q2_loss': q2_loss.item(),
            'actor_loss': actor_loss.item(),
            'mean_q': q_new.mean().item()
        }

# Training loop for SAC
def train_sac_quadruped(env, episodes=1000):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    agent = SAC(state_dim, action_dim)
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        
        while True:
            # Get action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
                action, _ = agent.network.sample_action(state_tensor)
                action = action.cpu().numpy()[0]
                
            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Store transition
            agent.store_transition(state, action, reward, next_state, done)
            
            # Update agent
            if len(agent.buffer) > 1000:
                losses = agent.update()
                
            episode_reward += reward
            state = next_state
            
            if done:
                print(f"Episode {episode}: Reward = {episode_reward:.2f}")
                break
                
    return agent
```

## Twin Delayed DDPG (TD3)

TD3 improves on DDPG with three key tricks:
1. Twin Q-networks (like SAC)
2. Delayed policy updates
3. Target policy smoothing

```python
class TD3:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action=1.0,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Actor network
        self.actor = self._build_actor(state_dim, action_dim).to(self.device)
        self.actor_target = self._build_actor(state_dim, action_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Twin Q-networks
        self.critic1 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic1_target = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        
        self.critic2 = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2_target = self._build_critic(state_dim, action_dim).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=lr)
        
        # Hyperparameters
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        
        self.total_iterations = 0
        
    def _build_actor(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        
    def _build_critic(self, state_dim, action_dim):
        return nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def select_action(self, state, add_noise=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy()[0]
        
        if add_noise:
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action = (action + noise).clip(-self.max_action, self.max_action)
            
        return action
        
    def update(self, replay_buffer, batch_size=256):
        self.total_iterations += 1
        
        # Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Compute target Q-values
            target_q1 = self.critic1_target(torch.cat([next_states, next_actions], 1))
            target_q2 = self.critic2_target(torch.cat([next_states, next_actions], 1))
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q
            
        # Update critics
        current_q1 = self.critic1(torch.cat([states, actions], 1))
        current_q2 = self.critic2(torch.cat([states, actions], 1))
        
        critic1_loss = F.mse_loss(current_q1, target_q)
        critic2_loss = F.mse_loss(current_q2, target_q)
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Delayed policy update
        if self.total_iterations % self.policy_delay == 0:
            # Compute actor loss
            actor_loss = -self.critic1(
                torch.cat([states, self.actor(states)], 1)
            ).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Soft update targets
            for param, target_param in zip(
                self.actor.parameters(), self.actor_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
                
            for param, target_param in zip(
                self.critic1.parameters(), self.critic1_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
                
            for param, target_param in zip(
                self.critic2.parameters(), self.critic2_target.parameters()
            ):
                target_param.data.copy_(
                    self.tau * param.data + (1 - self.tau) * target_param.data
                )
```

## Model-Based RL: Dreamer

Model-based RL learns a dynamics model and plans using it.

```python
class WorldModel(nn.Module):
    """Learns environment dynamics"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Encoder: observation -> latent
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log_std
        )
        
        # Dynamics: predict next latent
        self.dynamics = nn.GRU(
            action_dim + latent_dim,
            hidden_dim,
            batch_first=True
        )
        self.dynamics_out = nn.Linear(hidden_dim, latent_dim * 2)
        
        # Decoder: latent -> observation
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        
        # Reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def encode(self, obs):
        """Encode observation to latent distribution"""
        out = self.encoder(obs)
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-4
        
        return torch.distributions.Normal(mean, std)
        
    def decode(self, latent):
        """Decode latent to observation"""
        return self.decoder(latent)
        
    def predict_next(self, latent, action, hidden=None):
        """Predict next latent state"""
        input = torch.cat([latent, action], dim=-1)
        
        if len(input.shape) == 2:
            input = input.unsqueeze(1)
            
        out, hidden = self.dynamics(input, hidden)
        out = self.dynamics_out(out.squeeze(1))
        
        mean, log_std = out.chunk(2, dim=-1)
        std = F.softplus(log_std) + 1e-4
        
        return torch.distributions.Normal(mean, std), hidden
        
    def predict_reward(self, latent):
        """Predict reward from latent"""
        return self.reward_predictor(latent)
        
    def imagine_trajectory(self, initial_latent, policy, horizon=15):
        """Imagine future trajectory using learned model"""
        latent = initial_latent
        hidden = None
        
        trajectory = []
        
        for t in range(horizon):
            # Get action from policy
            action = policy(latent)
            
            # Predict next latent
            next_dist, hidden = self.predict_next(latent, action, hidden)
            latent = next_dist.rsample()
            
            # Predict reward
            reward = self.predict_reward(latent)
            
            trajectory.append({
                'latent': latent,
                'action': action,
                'reward': reward
            })
            
        return trajectory

class DreamerAgent:
    def __init__(self, state_dim, action_dim, lr=3e-4):
        self.world_model = WorldModel(state_dim, action_dim)
        self.actor = nn.Sequential(
            nn.Linear(32, 256),  # latent_dim = 32
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
    def update_world_model(self, observations, actions, rewards, next_observations):
        """Train world model on real data"""
        # Encode observations
        latent_dist = self.world_model.encode(observations)
        latent = latent_dist.rsample()
        
        # Predict next latent
        pred_next_dist, _ = self.world_model.predict_next(latent, actions)
        
        # Encode actual next observation
        next_latent_dist = self.world_model.encode(next_observations)
        
        # Reconstruction loss
        recon = self.world_model.decode(latent)
        recon_loss = F.mse_loss(recon, observations)
        
        # Dynamics loss (KL divergence)
        dynamics_loss = torch.distributions.kl_divergence(
            pred_next_dist, next_latent_dist
        ).mean()
        
        # Reward loss
        pred_reward = self.world_model.predict_reward(latent)
        reward_loss = F.mse_loss(pred_reward, rewards)
        
        # Total loss
        loss = recon_loss + dynamics_loss + reward_loss
        
        self.world_optimizer.zero_grad()
        loss.backward()
        self.world_optimizer.step()
        
        return loss.item()
        
    def update_actor_critic(self, batch_size=32, horizon=15):
        """Update policy using imagined trajectories"""
        # Sample initial latents
        # In practice, sample from replay buffer
        initial_latents = torch.randn(batch_size, 32)
        
        # Imagine trajectories
        trajectories = []
        for i in range(batch_size):
            traj = self.world_model.imagine_trajectory(
                initial_latents[i],
                self.actor,
                horizon
            )
            trajectories.append(traj)
            
        # Compute returns
        returns = []
        for traj in trajectories:
            ret = 0
            gamma = 0.99
            for t in reversed(range(len(traj))):
                ret = traj[t]['reward'] + gamma * ret
                traj[t]['return'] = ret
                
        # Update critic (TD learning)
        critic_loss = 0
        for traj in trajectories:
            for t in range(len(traj) - 1):
                value = self.critic(traj[t]['latent'])
                next_value = self.critic(traj[t+1]['latent'])
                target = traj[t]['reward'] + gamma * next_value.detach()
                critic_loss += F.mse_loss(value, target)
                
        critic_loss /= (batch_size * horizon)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor (policy gradient)
        actor_loss = 0
        for traj in trajectories:
            for t in range(len(traj)):
                value = self.critic(traj[t]['latent'])
                advantage = traj[t]['return'] - value.detach()
                
                # Reinforce-style update
                log_prob = -((traj[t]['action'] - self.actor(traj[t]['latent']))**2).sum()
                actor_loss -= log_prob * advantage
                
        actor_loss /= (batch_size * horizon)
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
```

## Choosing the Right Algorithm

### Algorithm Comparison

| Algorithm | Type | Sample Efficiency | Stability | Best For |
|-----------|------|------------------|-----------|----------|
| PPO | On-policy | Medium | High | General robotics, sim-to-real |
| SAC | Off-policy | High | High | Continuous control, exploration |
| TD3 | Off-policy | High | Medium | Precise control |
| Dreamer | Model-based | Very High | Medium | Limited real data |

### Decision Tree for Algorithm Selection

1. **Do you have a accurate simulator?**
   - No → Use model-based (Dreamer) to minimize real interactions
   - Yes → Continue to 2

2. **Is exploration critical?**
   - Yes → Use SAC (maximum entropy)
   - No → Continue to 3

3. **Need stable, predictable training?**
   - Yes → Use PPO
   - No → Use TD3 for sample efficiency

## Advanced Techniques

### 1. Hindsight Experience Replay (HER)
Learn from failures by relabeling goals:

```python
class HER:
    def __init__(self, replay_buffer, goal_selection_strategy='future'):
        self.buffer = replay_buffer
        self.strategy = goal_selection_strategy
        
    def store_episode(self, episode):
        """Store episode with hindsight goals"""
        T = len(episode)
        
        for t in range(T):
            # Original transition
            self.buffer.add(episode[t])
            
            # Hindsight transitions
            if self.strategy == 'future':
                # Sample future states as goals
                future_t = np.random.randint(t + 1, T)
                hindsight_goal = episode[future_t]['achieved_goal']
                
                # Recompute reward with new goal
                hindsight_reward = self.compute_reward(
                    episode[t]['achieved_goal'],
                    hindsight_goal
                )
                
                # Store hindsight transition
                hindsight_transition = episode[t].copy()
                hindsight_transition['goal'] = hindsight_goal
                hindsight_transition['reward'] = hindsight_reward
                
                self.buffer.add(hindsight_transition)
    
    def compute_reward(self, achieved_goal, desired_goal, threshold=0.05):
        """Sparse reward: -1 if not at goal, 0 if at goal"""
        distance = np.linalg.norm(achieved_goal - desired_goal)
        return 0.0 if distance < threshold else -1.0
```

### 2. Curriculum Learning
Gradually increase task difficulty:

```python
class CurriculumManager:
    def __init__(self, stages):
        self.stages = stages
        self.current_stage = 0
        self.success_buffer = deque(maxlen=100)
        
    def get_current_task(self):
        """Return current curriculum stage"""
        return self.stages[self.current_stage]
        
    def update(self, success):
        """Update curriculum based on performance"""
        self.success_buffer.append(success)
        
        # Graduate to next stage if success rate > 80%
        if len(self.success_buffer) == 100:
            success_rate = np.mean(self.success_buffer)
            if success_rate > 0.8 and self.current_stage < len(self.stages) - 1:
                self.current_stage += 1
                self.success_buffer.clear()
                print(f"Graduated to stage {self.current_stage}")
                
# Example curriculum for quadruped
curriculum_stages = [
    {'target_speed': 0.5, 'terrain': 'flat'},
    {'target_speed': 1.0, 'terrain': 'flat'},
    {'target_speed': 1.0, 'terrain': 'rough'},
    {'target_speed': 1.5, 'terrain': 'rough'},
]
```

### 3. Domain Randomization
Improve sim-to-real transfer:

```python
class DomainRandomizer:
    def __init__(self, params_range):
        self.params_range = params_range
        
    def randomize_physics(self, model):
        """Randomize physics parameters"""
        # Mass randomization
        for i in range(model.nbody):
            nominal_mass = model.body_mass[i]
            low, high = self.params_range['mass']
            model.body_mass[i] = nominal_mass * np.random.uniform(low, high)
            
        # Friction randomization
        for i in range(model.ngeom):
            low, high = self.params_range['friction']
            model.geom_friction[i] = np.random.uniform(low, high, 3)
            
        # Damping randomization
        for i in range(model.njnt):
            low, high = self.params_range['damping']
            model.dof_damping[i] = np.random.uniform(low, high)
            
    def randomize_observations(self, obs, noise_scale=0.01):
        """Add noise to observations"""
        noise = np.random.normal(0, noise_scale, obs.shape)
        return obs + noise
        
    def randomize_delays(self, action, delay_buffer, max_delay=3):
        """Simulate action delays"""
        delay = np.random.randint(0, max_delay + 1)
        
        if delay == 0:
            return action
            
        delay_buffer.append(action)
        if len(delay_buffer) > delay:
            return delay_buffer.popleft()
        else:
            return np.zeros_like(action)

# Usage
randomizer = DomainRandomizer({
    'mass': [0.8, 1.2],
    'friction': [[0.5, 0.5, 0.005], [2.0, 1.0, 0.01]],
    'damping': [0.1, 2.0]
})
```

### 4. Adversarial Training
Train robust policies with adversaries:

```python
class AdversarialTraining:
    def __init__(self, protagonist, adversary):
        self.protagonist = protagonist
        self.adversary = adversary
        
    def train_step(self, env):
        """One step of adversarial training"""
        state, _ = env.reset()
        
        while not done:
            # Protagonist acts
            protagonist_action = self.protagonist.get_action(state)
            
            # Adversary applies disturbance
            disturbance = self.adversary.get_action(state)
            env.apply_disturbance(disturbance)
            
            # Step environment
            next_state, reward, done, _ = env.step(protagonist_action)
            
            # Update both agents
            self.protagonist.update(state, protagonist_action, reward, next_state)
            
            # Adversary gets negative reward (zero-sum game)
            self.adversary.update(state, disturbance, -reward, next_state)
            
            state = next_state
```

## Real-World Deployment Considerations

### 1. Safety Constraints
Ensure safe exploration:

```python
class SafetyWrapper:
    def __init__(self, env, safety_threshold):
        self.env = env
        self.threshold = safety_threshold
        
    def step(self, action):
        # Predict next state
        predicted_state = self.predict_next_state(action)
        
        # Check safety
        if self.is_unsafe(predicted_state):
            # Project to safe action
            action = self.project_to_safe_action(action)
            
        return self.env.step(action)
        
    def is_unsafe(self, state):
        # Check joint limits
        if np.any(np.abs(state['joint_angles']) > self.threshold['joint_limits']):
            return True
            
        # Check velocity limits
        if np.linalg.norm(state['velocity']) > self.threshold['max_velocity']:
            return True
            
        # Check stability (ZMP criterion)
        if self.compute_zmp_margin(state) < self.threshold['zmp_margin']:
            return True
            
        return False
```

### 2. Sim-to-Real Transfer
Bridge the reality gap:

```python
class SimToRealAdapter:
    def __init__(self, sim_model, real_robot):
        self.sim_model = sim_model
        self.real_robot = real_robot
        self.calibration_data = []
        
    def calibrate(self, num_samples=100):
        """Calibrate sim to match real robot"""
        for _ in range(num_samples):
            # Random action
            action = np.random.uniform(-1, 1, self.real_robot.action_dim)
            
            # Execute on real robot
            real_next_state = self.real_robot.step(action)
            
            # Execute in sim
            sim_next_state = self.sim_model.step(action)
            
            # Store difference
            self.calibration_data.append({
                'action': action,
                'real': real_next_state,
                'sim': sim_next_state
            })
            
        # Learn correction model
        self.learn_dynamics_correction()
        
    def learn_dynamics_correction(self):
        """Learn residual dynamics model"""
        # Train neural network to predict: real_state - sim_state
        pass
        
    def transfer_policy(self, sim_policy):
        """Adapt policy trained in sim for real robot"""
        def real_policy(obs):
            # Apply observation adaptation
            adapted_obs = self.adapt_observation(obs)
            
            # Get sim policy action
            action = sim_policy(adapted_obs)
            
            # Apply action adaptation
            adapted_action = self.adapt_action(action)
            
            return adapted_action
            
        return real_policy
```

## Debugging and Visualization

### Training Diagnostics
Monitor these metrics:

```python
class TrainingMonitor:
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def log(self, **kwargs):
        for key, value in kwargs.items():
            self.metrics[key].append(value)
            
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Rewards
        axes[0, 0].plot(self.metrics['episode_reward'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Success rate
        axes[0, 1].plot(self.metrics['success_rate'])
        axes[0, 1].set_title('Success Rate')
        
        # Actor loss
        axes[0, 2].plot(self.metrics['actor_loss'])
        axes[0, 2].set_title('Actor Loss')
        
        # Critic loss
        axes[1, 0].plot(self.metrics['critic_loss'])
        axes[1, 0].set_title('Critic Loss')
        
        # Entropy
        axes[1, 1].plot(self.metrics['entropy'])
        axes[1, 1].set_title('Policy Entropy')
        
        # KL divergence
        axes[1, 2].plot(self.metrics['kl_divergence'])
        axes[1, 2].set_title('KL Divergence')
        
        plt.tight_layout()
        plt.show()
```

## Exercises
1. Implement SAC for the Go1 robot
2. Compare PPO vs SAC vs TD3 on walking task
3. Add curriculum learning for different speeds
4. Implement domain randomization for sim-to-real
5. Add safety constraints for joint limits
6. Visualize learned value functions
7. Implement HER for reaching tasks
8. Train adversarial disturbance policy

## Key Takeaways
- SAC maximizes entropy for better exploration
- TD3 uses twin critics and delayed updates for stability
- Model-based methods are sample-efficient but require good models
- Domain randomization helps sim-to-real transfer
- Curriculum learning enables learning complex behaviors
- Safety constraints are crucial for real deployment
- Always monitor multiple metrics during training
- Choose algorithm based on your specific requirements
