# Lesson 1: MuJoCo Basics (Comprehensive)

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

Important note: qvel has dimension 18 (not 19) because quaternion derivative has 3 DOF, not 4.

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

## Deep Dive: Physics Engine Internals

### The Forward Dynamics Problem
MuJoCo solves the equations of motion:

**M(q)q̈ + C(q,q̇) = τ + J^T f**

Where:
- M = mass matrix (inertia)
- q = generalized positions
- C = Coriolis and gravity forces
- τ = applied torques
- J^T f = contact forces

The solver proceeds in stages:
1. **Forward kinematics**: Compute cartesian positions from joint angles
2. **Compute forces**: Gravity, Coriolis, applied torques
3. **Detect contacts**: Find colliding geometries
4. **Solve constraints**: Compute contact forces that prevent penetration
5. **Integrate**: Update velocities and positions

### Contact Dynamics
MuJoCo uses a unique "soft contact" model:
```python
# Contact force components
# Normal force: F_n = -kp * penetration - kd * velocity
# Friction: Coulomb friction cone constraint

# Access contact information
for i in range(data.ncon):  # ncon = number of contacts
    contact = data.contact[i]
    
    # Contact position
    pos = contact.pos
    
    # Contact force (6D: force + torque)
    force = data.efc_force[i]
    
    # Geometries in contact
    geom1 = contact.geom1
    geom2 = contact.geom2
```

### Actuator Models
MuJoCo supports various actuator types:
```python
# Position servo (PD controller built-in)
# <actuator>
#   <position joint="joint_name" kp="100" kv="10"/>
# </actuator>

# Motor (torque control)
# <actuator>
#   <motor joint="joint_name" gear="1"/>
# </actuator>

# Access actuator info
for i in range(model.nu):
    # Actuator force/torque
    force = data.actuator_force[i]
    
    # Actuator length and velocity
    length = data.actuator_length[i]
    velocity = data.actuator_velocity[i]
```

## Advanced MuJoCo Features

### 1. Sensors
MuJoCo can simulate various sensors:
```python
# Define sensors in XML
# <sensor>
#   <accelerometer site="imu_site"/>
#   <gyro site="imu_site"/>
#   <force site="foot_site"/>
#   <torque joint="knee_joint"/>
# </sensor>

# Read sensor data
accel = data.sensordata[0:3]  # Accelerometer
gyro = data.sensordata[3:6]   # Gyroscope
```

### 2. Sites and Geoms
- **Geoms**: Collision geometry and visualization
- **Sites**: Reference frames for sensors, no collision

```python
# Get site position and orientation
site_id = model.site('foot_site').id
site_pos = data.site_xpos[site_id]
site_mat = data.site_xmat[site_id].reshape(3, 3)

# Get geom properties
geom_id = model.geom('leg_geom').id
geom_pos = data.geom_xpos[geom_id]
```

### 3. Jacobians
Essential for control and planning:
```python
# Get end-effector Jacobian
jacp = np.zeros((3, model.nv))  # Position Jacobian
jacr = np.zeros((3, model.nv))  # Rotation Jacobian

# Compute Jacobian for a body
body_id = model.body('foot').id
mujoco.mj_jac(model, data, jacp, jacr, data.xpos[body_id], body_id)

# Use for inverse dynamics
# tau = J^T * F (map end-effector force to joint torques)
```

## Performance Optimization

### 1. Timestep Selection
```python
# Smaller timestep = more stable but slower
model.opt.timestep = 0.001  # 1ms (very stable)
model.opt.timestep = 0.002  # 2ms (good balance)
model.opt.timestep = 0.005  # 5ms (faster but less stable)

# For quadrupeds, 1-2ms is typical
```

### 2. Solver Parameters
```python
# Iterations for constraint solver
model.opt.iterations = 50  # More iterations = better accuracy

# Tolerance
model.opt.tolerance = 1e-8

# Contact parameters
model.opt.o_margin = 0.001  # Contact detection margin
model.opt.o_solimp = np.array([0.9, 0.95, 0.001, 0.5, 2])
# [dmin, dmax, width, midpoint, power]
```

### 3. Parallel Simulation
```python
# Run multiple simulations in parallel
import multiprocessing as mp

def simulate_episode(seed):
    model = mujoco.MjModel.from_xml_path("robot.xml")
    data = mujoco.MjData(model)
    # ... run simulation ...
    return results

# Parallel execution
with mp.Pool(4) as pool:
    results = pool.map(simulate_episode, range(100))
```

## Debugging Tips

### 1. Visualization Helpers
```python
# Enable contact point visualization
model.vis.flag[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True

# Show contact forces
model.vis.flag[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True

# Transparent rendering
model.vis.flag[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
```

### 2. Checking Model Validity
```python
# Check for unstable configurations
def check_stability(data):
    # Check for NaN values
    if np.any(np.isnan(data.qpos)) or np.any(np.isnan(data.qvel)):
        print("WARNING: NaN detected in state!")
        return False
    
    # Check for excessive velocities
    if np.any(np.abs(data.qvel) > 100):
        print("WARNING: Excessive velocities!")
        return False
    
    return True

# Monitor energy for conservation
def compute_energy(model, data):
    # Kinetic energy
    kinetic = 0.5 * data.qvel @ mujoco.mj_fullM(model, data.qM) @ data.qvel
    
    # Potential energy
    potential = -np.sum(data.xipos[:, 2] * model.body_mass)
    
    return kinetic + potential
```

### 3. Common Issues and Solutions

**Issue: Robot explodes/flies away**
- Reduce timestep
- Check actuator limits
- Verify mass and inertia values
- Add damping to joints

**Issue: Contacts not detected**
- Increase contact margin
- Check geom sizes
- Verify collision pairs

**Issue: Slow simulation**
- Increase timestep (carefully)
- Reduce solver iterations
- Simplify collision geometry
- Use convex hulls instead of meshes

## Your First Interactive Simulation (Enhanced)

```python
import mujoco
from mujoco import viewer
import numpy as np

class RobotController:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Controller state
        self.target_pos = np.zeros(12)
        self.kp = 30.0
        self.kd = 2.0
        
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.3  # Set height
        
        # Set standing position
        stand_angles = np.array([
            0.0, 0.7, -1.4,  # FR
            0.0, 0.7, -1.4,  # FL
            0.0, 0.7, -1.4,  # RR
            0.0, 0.7, -1.4,  # RL
        ])
        self.data.qpos[7:19] = stand_angles
        
    def compute_torques(self):
        # PD control
        pos_error = self.target_pos - self.data.qpos[7:19]
        vel_error = -self.data.qvel[6:18]
        
        torques = self.kp * pos_error + self.kd * vel_error
        
        # Gravity compensation (simplified)
        # In practice, compute using inverse dynamics
        gravity_comp = np.zeros(12)
        gravity_comp[1::3] = 2.0  # Hip joints
        gravity_comp[2::3] = 1.0  # Knee joints
        
        return torques + gravity_comp
    
    def step(self):
        self.data.ctrl[:] = self.compute_torques()
        mujoco.mj_step(self.model, self.data)
        
    def run_interactive(self):
        self.reset()
        
        with viewer.launch_passive(self.model, self.data) as v:
            while v.is_running():
                self.step()
                v.sync()

# Usage
controller = RobotController("path/to/go1/scene.xml")
controller.run_interactive()
```

## Exercises
1. Run the simulation and observe the robot
2. Try changing `data.qpos[2]` to different heights
3. Experiment with the PD controller gains
4. Print out `data.qpos` and `data.qvel` shapes
5. Visualize contact forces
6. Implement gravity compensation
7. Add IMU sensor readings
8. Create a simple gait pattern

## Key Takeaways
- MuJoCo simulates physics by integrating equations of motion
- The state is split into positions (qpos) and velocities (qvel)
- We control the robot by setting torques (data.ctrl)
- The simulator steps forward in discrete time steps
- Contact dynamics are crucial for legged robots
- Proper timestep and solver settings are essential for stability
- Debugging tools help identify issues early
