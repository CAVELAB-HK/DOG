# Lesson 2: Robot Kinematics & Coordinates

## The Go1 Robot Structure

The Unitree Go1 has:
- 1 floating base (body)
- 4 legs (FR, FL, RR, RL - Front/Rear, Right/Left)
- 3 joints per leg (hip_abduction, hip_flexion, knee)
- Total: 12 actuated joints

```
      Front
    FL  │  FR
    ○───┼───○
    │ Body  │
    ○───┼───○
    RL  │  RR
      Rear
```

## Joint Naming Convention
Each leg has 3 joints:
1. **Hip Abduction/Adduction**: Moves leg sideways
2. **Hip Flexion/Extension**: Moves leg forward/backward
3. **Knee Flexion/Extension**: Bends the knee

## Understanding Coordinates

### World Frame
- Origin: Usually ground level
- X: Forward
- Y: Left  
- Z: Up

### Body Frame
- Origin: Robot's center of mass
- Moves and rotates with the robot

### Joint Angles
- Each joint has limits (e.g., -1.0 to 1.0 radians)
- Positive/negative depends on joint axis

## Forward Kinematics
Given joint angles, where is the foot?

```python
import numpy as np

def forward_kinematics_leg(hip_abd, hip_flex, knee):
    """
    Simplified FK for one leg (relative to hip)
    Real FK involves rotation matrices!
    """
    # Leg segment lengths (approximate for Go1)
    L_hip = 0.08  # Hip offset
    L_thigh = 0.213  # Upper leg
    L_shin = 0.213  # Lower leg
    
    # Simplified calculation (ignoring hip abduction for now)
    x = L_thigh * np.sin(hip_flex) + L_shin * np.sin(hip_flex + knee)
    z = -L_thigh * np.cos(hip_flex) - L_shin * np.cos(hip_flex + knee)
    y = L_hip * np.sin(hip_abd)
    
    return np.array([x, y, z])
```

## Inverse Kinematics
Given desired foot position, what joint angles?

```python
def inverse_kinematics_leg(x, y, z):
    """
    Simplified IK for one leg
    This is a 2D simplification!
    """
    L_thigh = 0.213
    L_shin = 0.213
    
    # Hip abduction from y position
    hip_abd = np.arcsin(y / 0.08)
    
    # 2D IK in x-z plane
    distance = np.sqrt(x**2 + z**2)
    
    # Law of cosines for knee angle
    cos_knee = (distance**2 - L_thigh**2 - L_shin**2) / (2 * L_thigh * L_shin)
    knee = np.arccos(np.clip(cos_knee, -1, 1))
    
    # Hip angle using trigonometry
    alpha = np.arctan2(x, -z)
    beta = np.arcsin(L_shin * np.sin(knee) / distance)
    hip_flex = alpha - beta
    
    return hip_abd, hip_flex, knee
```

## Reading Robot State in MuJoCo

```python
import mujoco

model = mujoco.MjModel.from_xml_path("path/to/go1/scene.xml")
data = mujoco.MjData(model)

# Get base position and orientation
base_pos = data.qpos[0:3]  # [x, y, z]
base_quat = data.qpos[3:7]  # [w, x, y, z] quaternion

# Get joint angles (12 joints)
joint_angles = data.qpos[7:19]

# Map to legs
FR_joints = joint_angles[0:3]  # [hip_abd, hip_flex, knee]
FL_joints = joint_angles[3:6]
RR_joints = joint_angles[6:9]
RL_joints = joint_angles[9:12]

# Get velocities
base_vel = data.qvel[0:3]  # Linear velocity
base_angvel = data.qvel[3:6]  # Angular velocity
joint_velocities = data.qvel[6:18]

# Get foot positions (MuJoCo computes these!)
# Find site IDs for feet
FR_foot_id = model.site('FR_foot').id
FL_foot_id = model.site('FL_foot').id
RR_foot_id = model.site('RR_foot').id
RL_foot_id = model.site('RL_foot').id

# Get foot positions in world frame
FR_foot_pos = data.site_xpos[FR_foot_id]
FL_foot_pos = data.site_xpos[FL_foot_id]
# etc...
```

## Quaternions - Representing Orientation

Quaternions avoid "gimbal lock" and are used in MuJoCo:
- q = [w, x, y, z] where w² + x² + y² + z² = 1
- Identity quaternion: [1, 0, 0, 0] (no rotation)

```python
# Convert quaternion to rotation matrix
def quat_to_rot_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])

# Get robot's orientation matrix
orientation = quat_to_rot_matrix(data.qpos[3:7])
```

## Exercises
1. Print all joint angles and identify which joint is which
2. Move individual joints and observe the effect
3. Calculate foot position using FK and compare with MuJoCo's computed position
4. Try to make the robot stand at different heights by adjusting joint angles

## Key Takeaways
- The Go1 has 12 actuated joints plus a floating base
- Forward kinematics: joints → foot position
- Inverse kinematics: foot position → joints
- MuJoCo handles complex FK/IK internally
- Quaternions represent orientation without singularities
