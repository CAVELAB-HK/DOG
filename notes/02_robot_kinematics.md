# Lesson 2: Robot Kinematics & Coordinates (Comprehensive)

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

## Mathematical Foundations

### Homogeneous Transformations
Robot kinematics uses 4x4 transformation matrices:

```python
import numpy as np

def create_transform_matrix(rotation, translation):
    """
    Create 4x4 homogeneous transformation matrix
    T = [R  t]
        [0  1]
    """
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T

def rotation_x(angle):
    """Rotation around X axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])

def rotation_y(angle):
    """Rotation around Y axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

def rotation_z(angle):
    """Rotation around Z axis"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1]
    ])
```

### Denavit-Hartenberg (DH) Parameters
Standard method for describing robot kinematics:

```python
# DH parameters for one leg of Go1 (approximate)
# [a, alpha, d, theta]
# a: link length, alpha: link twist, d: link offset, theta: joint angle

class DHParameters:
    def __init__(self):
        # Hip to thigh
        self.dh1 = {'a': 0.08, 'alpha': np.pi/2, 'd': 0, 'theta': 0}
        # Thigh
        self.dh2 = {'a': 0.213, 'alpha': 0, 'd': 0, 'theta': 0}
        # Shin
        self.dh3 = {'a': 0.213, 'alpha': 0, 'd': 0, 'theta': 0}
    
    def dh_matrix(self, a, alpha, d, theta):
        """Compute transformation matrix from DH parameters"""
        ct = np.cos(theta)
        st = np.sin(theta)
        ca = np.cos(alpha)
        sa = np.sin(alpha)
        
        return np.array([
            [ct, -st*ca, st*sa, a*ct],
            [st, ct*ca, -ct*sa, a*st],
            [0, sa, ca, d],
            [0, 0, 0, 1]
        ])
```

## Complete Forward Kinematics

```python
class Go1LegKinematics:
    def __init__(self, leg_name='FR'):
        # Leg configuration
        self.leg_name = leg_name
        
        # Link lengths (meters)
        self.l_hip = 0.08505  # Hip offset (abduction joint to hip joint)
        self.l_thigh = 0.213  # Upper leg length
        self.l_calf = 0.213   # Lower leg length
        
        # Hip offset depends on leg position
        if 'R' in leg_name:  # Right legs
            self.hip_offset_y = -self.l_hip
        else:  # Left legs
            self.hip_offset_y = self.l_hip
            
        if 'F' in leg_name:  # Front legs
            self.hip_offset_x = 0.1881
        else:  # Rear legs
            self.hip_offset_x = -0.1881
            
    def forward_kinematics(self, q):
        """
        Compute foot position given joint angles
        q = [hip_abduction, hip_flexion, knee_flexion]
        Returns position in hip frame
        """
        q_abd, q_hip, q_knee = q
        
        # Transform from base to hip abduction joint
        T_base_abd = create_transform_matrix(
            np.eye(3),
            [self.hip_offset_x, self.hip_offset_y, 0]
        )
        
        # Hip abduction rotation
        T_abd = create_transform_matrix(
            rotation_x(q_abd),
            [0, 0, 0]
        )
        
        # Hip flexion rotation with offset
        T_hip = create_transform_matrix(
            rotation_y(q_hip),
            [0, self.l_hip if 'R' in self.leg_name else -self.l_hip, 0]
        )
        
        # Thigh link
        T_thigh = create_transform_matrix(
            rotation_y(0),
            [0, 0, -self.l_thigh]
        )
        
        # Knee rotation
        T_knee = create_transform_matrix(
            rotation_y(q_knee),
            [0, 0, 0]
        )
        
        # Calf link to foot
        T_foot = create_transform_matrix(
            np.eye(3),
            [0, 0, -self.l_calf]
        )
        
        # Complete transformation
        T_total = T_base_abd @ T_abd @ T_hip @ T_thigh @ T_knee @ T_foot
        
        # Extract foot position
        foot_pos = T_total[:3, 3]
        
        return foot_pos
    
    def jacobian(self, q):
        """
        Compute leg Jacobian matrix
        J = [dx/dq1, dx/dq2, dx/dq3]
        """
        eps = 1e-6
        J = np.zeros((3, 3))
        
        # Numerical differentiation
        pos0 = self.forward_kinematics(q)
        
        for i in range(3):
            q_plus = q.copy()
            q_plus[i] += eps
            pos_plus = self.forward_kinematics(q_plus)
            
            J[:, i] = (pos_plus - pos0) / eps
            
        return J
```

## Advanced Inverse Kinematics

```python
class Go1InverseKinematics:
    def __init__(self, leg_kinematics):
        self.fk = leg_kinematics
        
    def analytical_ik(self, target_pos):
        """
        Analytical solution for 3-DOF leg
        target_pos: desired foot position [x, y, z] in hip frame
        """
        x, y, z = target_pos
        
        # Hip abduction angle from y position
        q_abd = np.arctan2(y, -z)
        
        # Distance in x-z plane after abduction rotation
        z_eff = -np.sqrt(y**2 + z**2)
        
        # 2-link IK in x-z plane
        l1 = self.fk.l_thigh
        l2 = self.fk.l_calf
        
        # Distance from hip to foot
        d = np.sqrt(x**2 + z_eff**2)
        
        # Check reachability
        if d > l1 + l2:
            print("Target unreachable: too far")
            d = l1 + l2 - 0.001
        elif d < abs(l1 - l2):
            print("Target unreachable: too close")
            d = abs(l1 - l2) + 0.001
            
        # Law of cosines for knee angle
        cos_knee = (l1**2 + l2**2 - d**2) / (2 * l1 * l2)
        q_knee = np.pi - np.arccos(np.clip(cos_knee, -1, 1))
        
        # Hip angle
        alpha = np.arctan2(x, -z_eff)
        cos_beta = (l1**2 + d**2 - l2**2) / (2 * l1 * d)
        beta = np.arccos(np.clip(cos_beta, -1, 1))
        q_hip = alpha + beta
        
        return np.array([q_abd, q_hip, q_knee])
    
    def numerical_ik(self, target_pos, q_init=None, max_iter=100):
        """
        Numerical IK using Newton-Raphson method
        More robust for complex kinematics
        """
        if q_init is None:
            q_init = np.zeros(3)
            
        q = q_init.copy()
        
        for i in range(max_iter):
            # Current position
            pos = self.fk.forward_kinematics(q)
            
            # Error
            error = target_pos - pos
            
            # Check convergence
            if np.linalg.norm(error) < 1e-4:
                return q
                
            # Jacobian
            J = self.fk.jacobian(q)
            
            # Pseudo-inverse with damping (more stable)
            lambda_damp = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damp * np.eye(3))
            
            # Update
            dq = J_pinv @ error
            q += 0.5 * dq  # Step size for stability
            
            # Joint limits
            q = np.clip(q, [-1.047, -0.663, -2.721], [1.047, 2.966, -0.837])
            
        print(f"IK did not converge, error: {np.linalg.norm(error)}")
        return q
```

## Whole-Body Kinematics

```python
class Go1WholeBodyKinematics:
    def __init__(self):
        self.legs = {
            'FR': Go1LegKinematics('FR'),
            'FL': Go1LegKinematics('FL'),
            'RR': Go1LegKinematics('RR'),
            'RL': Go1LegKinematics('RL')
        }
        
        # Base dimensions
        self.body_length = 0.3762
        self.body_width = 0.1701
        
    def get_foot_positions_world(self, base_pos, base_quat, joint_angles):
        """
        Get all foot positions in world frame
        base_pos: [x, y, z]
        base_quat: [w, x, y, z]
        joint_angles: [12] array
        """
        # Base rotation matrix
        R_base = quat_to_rot_matrix(base_quat)
        
        foot_positions = {}
        
        for i, leg_name in enumerate(['FR', 'FL', 'RR', 'RL']):
            # Get joint angles for this leg
            q = joint_angles[i*3:(i+1)*3]
            
            # Forward kinematics in body frame
            foot_body = self.legs[leg_name].forward_kinematics(q)
            
            # Transform to world frame
            foot_world = base_pos + R_base @ foot_body
            
            foot_positions[leg_name] = foot_world
            
        return foot_positions
    
    def compute_com(self, joint_angles):
        """
        Compute center of mass position
        Simplified: assumes uniform mass distribution
        """
        # Mass distribution (approximate)
        mass_body = 12.0  # kg
        mass_leg = 1.0    # kg per leg
        
        # Body COM at origin (body frame)
        com = np.zeros(3) * mass_body
        
        # Add leg contributions
        for i, leg_name in enumerate(['FR', 'FL', 'RR', 'RL']):
            q = joint_angles[i*3:(i+1)*3]
            
            # Approximate leg COM at mid-point to foot
            foot_pos = self.legs[leg_name].forward_kinematics(q)
            leg_com = foot_pos * 0.5
            
            com += leg_com * mass_leg
            
        # Normalize by total mass
        total_mass = mass_body + 4 * mass_leg
        com /= total_mass
        
        return com
    
    def stability_margin(self, foot_positions, com_xy):
        """
        Compute static stability margin
        Distance from COM to support polygon edge
        """
        # Get support polygon (convex hull of foot contacts)
        from scipy.spatial import ConvexHull
        
        # Assume all feet in contact
        points = np.array([pos[:2] for pos in foot_positions.values()])
        
        if len(points) < 3:
            return 0  # Unstable
            
        hull = ConvexHull(points)
        
        # Check if COM is inside polygon
        # Simplified: compute minimum distance to edges
        min_distance = float('inf')
        
        for simplex in hull.simplices:
            p1 = points[simplex[0]]
            p2 = points[simplex[1]]
            
            # Distance from point to line segment
            edge = p2 - p1
            t = np.clip(np.dot(com_xy - p1, edge) / np.dot(edge, edge), 0, 1)
            closest = p1 + t * edge
            distance = np.linalg.norm(com_xy - closest)
            
            min_distance = min(min_distance, distance)
            
        return min_distance
```

## Dynamics and Forces

```python
class Go1Dynamics:
    def __init__(self):
        self.mass = 12.0  # kg
        self.gravity = 9.81
        
        # Inertia tensor (approximate)
        self.inertia = np.diag([0.5, 0.8, 0.3])
        
    def compute_grf(self, foot_positions, com_pos, com_acc=None):
        """
        Compute ground reaction forces (simplified)
        Using static equilibrium or dynamics
        """
        if com_acc is None:
            # Static case: forces balance weight
            total_force = np.array([0, 0, self.mass * self.gravity])
        else:
            # Dynamic case: F = ma
            total_force = self.mass * (com_acc + np.array([0, 0, self.gravity]))
            
        # Distribute forces based on distance from COM
        # This is simplified; real distribution depends on optimization
        forces = {}
        weights = {}
        
        for leg, pos in foot_positions.items():
            # Weight inversely proportional to distance
            dist = np.linalg.norm(pos[:2] - com_pos[:2])
            weights[leg] = 1.0 / (dist + 0.1)
            
        # Normalize weights
        total_weight = sum(weights.values())
        
        for leg in foot_positions:
            forces[leg] = (weights[leg] / total_weight) * total_force
            
        return forces
    
    def joint_torques_from_forces(self, leg_kinematics, q, foot_force):
        """
        Compute joint torques from foot force
        tau = J^T * F
        """
        J = leg_kinematics.jacobian(q)
        tau = J.T @ foot_force
        
        # Add gravity compensation
        tau += self.gravity_compensation(leg_kinematics, q)
        
        return tau
    
    def gravity_compensation(self, leg_kinematics, q):
        """
        Compute torques to compensate for gravity
        Simplified version
        """
        # This would normally use the full dynamics model
        # Here's a simple approximation
        
        # Masses
        m_thigh = 0.5  # kg
        m_calf = 0.3   # kg
        
        # Simplified: assume COM at link centers
        tau = np.zeros(3)
        
        # Hip torque from thigh and calf weight
        tau[1] = (m_thigh * 0.5 + m_calf) * leg_kinematics.l_thigh * self.gravity * np.cos(q[1])
        
        # Knee torque from calf weight
        tau[2] = m_calf * 0.5 * leg_kinematics.l_calf * self.gravity * np.cos(q[1] + q[2])
        
        return tau
```

## Quaternion Operations (Essential for 3D Robotics)

```python
class QuaternionOps:
    @staticmethod
    def quat_multiply(q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    @staticmethod
    def quat_inverse(q):
        """Inverse of quaternion"""
        w, x, y, z = q
        return np.array([w, -x, -y, -z])
    
    @staticmethod
    def quat_to_euler(q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        w, x, y, z = q
        
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(np.clip(sinp, -1, 1))
        
        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        return np.array([roll, pitch, yaw])
    
    @staticmethod
    def euler_to_quat(euler):
        """Convert Euler angles to quaternion"""
        roll, pitch, yaw = euler
        
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)
        
        w = cr * cp * cy + sr * sp * sy
        x = sr * cp * cy - cr * sp * sy
        y = cr * sp * cy + sr * cp * sy
        z = cr * cp * sy - sr * sp * cy
        
        return np.array([w, x, y, z])
    
    @staticmethod
    def quat_slerp(q1, q2, t):
        """Spherical linear interpolation between quaternions"""
        # Normalize
        q1 = q1 / np.linalg.norm(q1)
        q2 = q2 / np.linalg.norm(q2)
        
        # Compute angle between quaternions
        dot = np.dot(q1, q2)
        
        # If negative dot, negate one quaternion
        if dot < 0:
            q2 = -q2
            dot = -dot
            
        # If very close, use linear interpolation
        if dot > 0.9995:
            result = q1 + t * (q2 - q1)
            return result / np.linalg.norm(result)
            
        # Spherical interpolation
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        
        w1 = np.sin((1 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        
        return w1 * q1 + w2 * q2
```

## Practical Implementation with MuJoCo

```python
import mujoco

class MuJoCoKinematicsInterface:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Cache body and site IDs
        self.foot_sites = {
            'FR': model.site('FR_foot').id,
            'FL': model.site('FL_foot').id,
            'RR': model.site('RR_foot').id,
            'RL': model.site('RL_foot').id
        }
        
    def get_foot_positions(self):
        """Get foot positions from MuJoCo"""
        positions = {}
        for leg, site_id in self.foot_sites.items():
            positions[leg] = self.data.site_xpos[site_id].copy()
        return positions
    
    def get_foot_velocities(self):
        """Get foot velocities using Jacobian"""
        velocities = {}
        
        for leg, site_id in self.foot_sites.items():
            # Get Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            
            # Compute velocity: v = J * qvel
            vel = jacp @ self.data.qvel
            velocities[leg] = vel
            
        return velocities
    
    def set_foot_target(self, leg, target_pos):
        """
        Use IK to set foot to target position
        This is a simplified version - real implementation would be more robust
        """
        # Get current joint angles for the leg
        leg_idx = ['FR', 'FL', 'RR', 'RL'].index(leg)
        q_current = self.data.qpos[7 + leg_idx*3: 7 + (leg_idx+1)*3]
        
        # Create kinematics object
        leg_kin = Go1LegKinematics(leg)
        ik_solver = Go1InverseKinematics(leg_kin)
        
        # Transform target from world to body frame
        base_pos = self.data.qpos[:3]
        base_quat = self.data.qpos[3:7]
        R_base = quat_to_rot_matrix(base_quat)
        
        target_body = R_base.T @ (target_pos - base_pos)
        
        # Solve IK
        q_target = ik_solver.numerical_ik(target_body, q_current)
        
        # Set joint positions
        self.data.qpos[7 + leg_idx*3: 7 + (leg_idx+1)*3] = q_target
        
        return q_target
```

## Exercises
1. Implement forward kinematics for all legs and verify against MuJoCo
2. Test analytical vs numerical IK solutions
3. Compute and visualize the support polygon
4. Implement gravity compensation and test in simulation
5. Create a foot trajectory generator for walking
6. Compute workspace of each leg (reachable positions)
7. Implement whole-body inverse kinematics
8. Add joint velocity and acceleration limits to IK

## Key Takeaways
- Forward kinematics uses transformation matrices to find end-effector position
- Inverse kinematics can be solved analytically (fast) or numerically (flexible)
- Jacobians relate joint velocities to end-effector velocities
- Quaternions avoid singularities in 3D rotations
- Dynamics relates forces to motion through mass and inertia
- MuJoCo provides many kinematic computations built-in
- Always consider joint limits and singularities in real implementations
