# Lesson 3: Building a Gymnasium Environment

## What is a Gym Environment?

Gymnasium (formerly OpenAI Gym) provides a standard interface for RL:
- `reset()`: Start a new episode
- `step(action)`: Take an action, get observation, reward, done
- `observation_space`: What the agent observes
- `action_space`: What actions are available

## Basic Structure

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class Go1Env(gym.Env):
    def __init__(self, xml_path):
        super().__init__()
        
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=-1.0, 
            high=1.0, 
            shape=(12,),  # 12 joint torques
            dtype=np.float32
        )
        
        # We'll define observations later
        obs_dim = self._get_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial position
        self.data.qpos[2] = 0.3  # Height
        
        # Get observation
        observation = self._get_observation()
        info = {}
        
        return observation, info
    
    def step(self, action):
        # Apply action (torques)
        self.data.ctrl[:] = action * 10  # Scale actions
        
        # Step simulation (4 substeps for stability)
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._get_reward()
        
        # Check if episode is done
        terminated = self._is_terminated()
        truncated = False
        
        info = {}
        
        return observation, reward, terminated, truncated, info
```

## Designing Observations

What should the robot "see"? Common choices:

```python
def _get_observation(self):
    """
    Build observation vector
    """
    obs = []
    
    # 1. Base orientation (quaternion or rotation matrix)
    base_quat = self.data.qpos[3:7]
    obs.append(base_quat)
    
    # 2. Joint positions (12 joints)
    joint_pos = self.data.qpos[7:19]
    obs.append(joint_pos)
    
    # 3. Base linear velocity (in body frame)
    base_vel = self.data.qvel[0:3]
    # Transform to body frame using rotation matrix
    rot_mat = self._quat_to_rot_matrix(base_quat)
    base_vel_body = rot_mat.T @ base_vel
    obs.append(base_vel_body)
    
    # 4. Base angular velocity
    base_angvel = self.data.qvel[3:6]
    obs.append(base_angvel)
    
    # 5. Joint velocities
    joint_vel = self.data.qvel[6:18]
    obs.append(joint_vel)
    
    # 6. Previous action (helps with smooth motion)
    if hasattr(self, 'prev_action'):
        obs.append(self.prev_action)
    else:
        obs.append(np.zeros(12))
    
    # Concatenate all observations
    return np.concatenate(obs).astype(np.float32)

def _get_obs_dim(self):
    """
    Calculate observation dimension
    """
    return (
        4 +   # quaternion
        12 +  # joint positions
        3 +   # base velocity
        3 +   # angular velocity
        12 +  # joint velocities
        12    # previous action
    )  # Total: 46
```

## Designing Rewards

The reward function teaches the robot what we want:

```python
def _get_reward(self):
    """
    Calculate reward for current state
    """
    # Forward velocity reward
    forward_vel = self.data.qvel[0]  # x-velocity
    forward_reward = forward_vel
    
    # Penalize lateral movement
    lateral_vel = abs(self.data.qvel[1])  # y-velocity
    lateral_penalty = -0.5 * lateral_vel
    
    # Penalize high angular velocity (wobbly motion)
    ang_vel = np.linalg.norm(self.data.qvel[3:6])
    stability_penalty = -0.1 * ang_vel
    
    # Energy penalty (penalize high torques)
    energy_penalty = -0.001 * np.sum(self.data.ctrl**2)
    
    # Height reward (stay at target height)
    target_height = 0.3
    height = self.data.qpos[2]
    height_reward = -2.0 * abs(height - target_height)
    
    # Foot contact reward (all feet should touch ground periodically)
    # We'll implement this later
    
    total_reward = (
        forward_reward + 
        lateral_penalty + 
        stability_penalty + 
        energy_penalty + 
        height_reward
    )
    
    return total_reward
```

## Termination Conditions

When should an episode end?

```python
def _is_terminated(self):
    """
    Check if episode should end
    """
    # Robot fell over
    height = self.data.qpos[2]
    if height < 0.15:  # Too low
        return True
    
    # Robot is tilted too much
    quat = self.data.qpos[3:7]
    rot_mat = self._quat_to_rot_matrix(quat)
    z_axis = rot_mat[:, 2]  # Robot's up direction
    if z_axis[2] < 0.5:  # More than 60 degrees tilt
        return True
    
    return False
```

## Complete Minimal Environment

```python
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco

class Go1Env(gym.Env):
    def __init__(self, xml_path, render_mode=None):
        super().__init__()
        
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        
        # Action space: normalized joint torques
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32
        )
        
        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(46,), dtype=np.float32
        )
        
        self.prev_action = np.zeros(12)
        self.timestep = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.3
        
        # Small random perturbation for variety
        self.data.qpos[7:19] += np.random.uniform(-0.1, 0.1, 12)
        
        self.prev_action = np.zeros(12)
        self.timestep = 0
        
        return self._get_observation(), {}
    
    def step(self, action):
        # Scale and apply action
        self.data.ctrl[:] = action * 20.0
        self.prev_action = action
        
        # Simulate
        for _ in range(4):
            mujoco.mj_step(self.model, self.data)
        
        self.timestep += 1
        
        obs = self._get_observation()
        reward = self._get_reward()
        terminated = self._is_terminated()
        truncated = self.timestep >= 1000  # Max episode length
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        return np.concatenate([
            self.data.qpos[3:7],      # orientation
            self.data.qpos[7:19],     # joint angles
            self.data.qvel[0:3],      # base velocity
            self.data.qvel[3:6],      # angular velocity
            self.data.qvel[6:18],     # joint velocities
            self.prev_action          # previous action
        ])
    
    def _get_reward(self):
        forward_reward = self.data.qvel[0]
        energy_penalty = -0.001 * np.sum(self.data.ctrl**2)
        height_penalty = -2.0 * abs(self.data.qpos[2] - 0.3)
        
        return forward_reward + energy_penalty + height_penalty
    
    def _is_terminated(self):
        return self.data.qpos[2] < 0.15
    
    def _quat_to_rot_matrix(self, quat):
        w, x, y, z = quat
        return np.array([
            [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
        ])
```

## Testing Your Environment

```python
# Test the environment
env = Go1Env("/path/to/go1/scene.xml")

# Check spaces
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Run random actions
obs, info = env.reset()
for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Reward: {reward:.3f}")
    
    if terminated or truncated:
        obs, info = env.reset()
```

## Exercises
1. Implement the environment and test with random actions
2. Add more observations (foot contacts, IMU data)
3. Experiment with different reward weights
4. Add domain randomization (varying mass, friction)

## Key Takeaways
- Gym environments provide a standard RL interface
- Observations should contain all info needed for decision-making
- Rewards shape the behavior you want to learn
- Start simple and gradually add complexity
