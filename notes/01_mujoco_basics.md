# Lesson 1: MuJoCo Basics

## What is MuJoCo?
MuJoCo (Multi-Joint dynamics with Contact) is a physics engine designed for robotics simulation. It simulates:
- Rigid body dynamics (how things move)
- Contacts and collisions
- Actuators (motors)
- Sensors

## Core Concepts

### 1. Model vs Data
- **Model** (`MjModel`): The static description of your robot (like a blueprint)
- **Data** (`MjData`): The current state (positions, velocities, forces)

### 2. Generalized Coordinates
MuJoCo uses "generalized coordinates" (qpos, qvel):
- **qpos**: Position coordinates (x,y,z position + quaternion orientation + joint angles)
- **qvel**: Velocity coordinates (linear velocity + angular velocity + joint velocities)

For the Go1 robot:
- qpos[0:3] = robot base position (x,y,z)
- qpos[3:7] = robot base orientation (quaternion)
- qpos[7:19] = 12 joint angles (3 per leg × 4 legs)

### 3. The Simulation Loop
```python
import mujoco
import numpy as np

# Load model
model = mujoco.MjModel.from_xml_path("path/to/go1/scene.xml")
data = mujoco.MjData(model)

# Simulation loop
for _ in range(1000):
    # Apply control (torques to joints)
    data.ctrl[:] = np.zeros(model.nu)  # nu = number of actuators
    
    # Step physics forward
    mujoco.mj_step(model, data)
    
    # Time advances by model.opt.timestep (usually 0.002 seconds)
```

## Physics Behind mj_step()

The `mj_step()` function solves the equations of motion:

**M(q)q̈ + C(q,q̇) = τ + J^T f**

Where:
- M = mass matrix (inertia)
- q = generalized positions
- C = Coriolis and gravity forces
- τ = applied torques
- J^T f = contact forces

## Your First Interactive Simulation

```python
import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("path/to/go1/scene.xml")
data = mujoco.MjData(model)

# Reset to initial position
mujoco.mj_resetData(model, data)
data.qpos[2] = 0.3  # Set robot height to 30cm

# Launch interactive viewer
with viewer.launch_passive(model, data) as v:
    while v.is_running():
        # Simple PD controller to keep joints at zero
        data.ctrl[:] = -10 * data.qpos[7:19] - 1 * data.qvel[6:18]
        
        mujoco.mj_step(model, data)
        v.sync()
```

## Exercises
1. Run the simulation and observe the robot
2. Try changing `data.qpos[2]` to different heights
3. Experiment with the PD controller gains (the -10 and -1 values)
4. Print out `data.qpos` and `data.qvel` shapes to understand the state

## Key Takeaways
- MuJoCo simulates physics by integrating equations of motion
- The state is split into positions (qpos) and velocities (qvel)
- We control the robot by setting torques (data.ctrl)
- The simulator steps forward in discrete time steps
