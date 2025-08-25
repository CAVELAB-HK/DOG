# Lesson 7: Phase-Based Trajectory Generation for Quadruped Locomotion

## Overview

Phase-based trajectory generation is the foundation of rhythmic quadruped locomotion. Instead of planning explicit paths, we use phase variables that cycle from 0 to 2π to coordinate leg movements. This approach creates natural, periodic gaits that can adapt to different speeds and terrains.

## Core Concepts

### What is Phase?
Phase (φ) represents where we are in a cyclic motion:
- φ = 0: Start of cycle
- φ = π: Halfway through
- φ = 2π: Complete cycle (same as 0)

For walking, phase determines:
1. Which legs are in stance (supporting) vs swing (moving)
2. Where each foot should be in its trajectory
3. How legs coordinate with each other

## Mathematical Foundations

### 1. Phase Dynamics

The phase evolves over time with frequency ω:

**φ(t) = ωt mod 2π**

Or as a differential equation:
**dφ/dt = ω**

Where:
- ω = 2π/T (angular frequency)
- T = gait period (time for one complete cycle)

```python
import numpy as np
import matplotlib.pyplot as plt

class PhaseGenerator:
    def __init__(self, frequency=1.0):
        """
        frequency: Gait frequency in Hz (cycles per second)
        """
        self.frequency = frequency
        self.omega = 2 * np.pi * frequency
        self.phase = 0.0
        
    def update(self, dt):
        """Update phase based on timestep"""
        self.phase += self.omega * dt
        self.phase = self.phase % (2 * np.pi)  # Wrap to [0, 2π]
        return self.phase
    
    def reset(self, initial_phase=0.0):
        """Reset phase to initial value"""
        self.phase = initial_phase
        
    def get_phase(self):
        """Get current phase"""
        return self.phase
```

### 2. Gait Patterns and Phase Offsets

Different gaits are characterized by phase relationships between legs:

**Trot**: Diagonal pairs move together
- FR: φ
- FL: φ + π
- RR: φ + π  
- RL: φ

**Walk**: One leg at a time
- FR: φ
- FL: φ + π/2
- RR: φ + π
- RL: φ + 3π/2

**Bound**: Front/back pairs together
- FR: φ
- FL: φ
- RR: φ + π
- RL: φ + π

**Gallop**: Asymmetric pattern
- FR: φ
- FL: φ + π/4
- RR: φ + π
- RL: φ + 5π/4

```python
class GaitPattern:
    """Define phase offsets for different gaits"""
    
    TROT = {
        'FR': 0.0,
        'FL': np.pi,
        'RR': np.pi,
        'RL': 0.0
    }
    
    WALK = {
        'FR': 0.0,
        'FL': np.pi/2,
        'RR': np.pi,
        'RL': 3*np.pi/2
    }
    
    BOUND = {
        'FR': 0.0,
        'FL': 0.0,
        'RR': np.pi,
        'RL': np.pi
    }
    
    GALLOP = {
        'FR': 0.0,
        'FL': np.pi/4,
        'RR': np.pi,
        'RL': 5*np.pi/4
    }
    
    @staticmethod
    def get_phase_offsets(gait_type='TROT'):
        """Get phase offsets for specified gait"""
        gaits = {
            'TROT': GaitPattern.TROT,
            'WALK': GaitPattern.WALK,
            'BOUND': GaitPattern.BOUND,
            'GALLOP': GaitPattern.GALLOP
        }
        return gaits.get(gait_type, GaitPattern.TROT)
```

### 3. Duty Factor and Stance/Swing Phases

**Duty Factor (β)**: Fraction of cycle spent in stance phase
- β = 0.5: Equal stance and swing (trot)
- β = 0.75: Longer stance (walk)
- β = 0.3: Short stance (running)

**Phase decomposition**:
- Stance phase: φ ∈ [0, 2πβ]
- Swing phase: φ ∈ [2πβ, 2π]

```python
def is_stance_phase(phase, duty_factor=0.5):
    """
    Determine if leg is in stance or swing phase
    Returns: True if stance, False if swing
    """
    stance_duration = 2 * np.pi * duty_factor
    return phase <= stance_duration

def get_phase_in_swing(phase, duty_factor=0.5):
    """
    Get normalized phase within swing period [0, 1]
    """
    stance_duration = 2 * np.pi * duty_factor
    if phase <= stance_duration:
        return 0.0  # Not in swing
    
    swing_start = stance_duration
    swing_duration = 2 * np.pi * (1 - duty_factor)
    swing_phase = (phase - swing_start) / swing_duration
    
    return np.clip(swing_phase, 0, 1)

def get_phase_in_stance(phase, duty_factor=0.5):
    """
    Get normalized phase within stance period [0, 1]
    """
    stance_duration = 2 * np.pi * duty_factor
    if phase > stance_duration:
        return 1.0  # Not in stance
    
    stance_phase = phase / stance_duration
    return np.clip(stance_phase, 0, 1)
```

## Trajectory Generation

### 1. Swing Foot Trajectory

The swing trajectory lifts the foot and moves it forward. We use a combination of sinusoidal and polynomial curves:

**Vertical (Z) trajectory**:
```
z(s) = z_clearance * sin(πs)
```

**Horizontal (X) trajectory**:
```
x(s) = x_start + (x_end - x_start) * s
```

Where s ∈ [0, 1] is normalized swing phase.

```python
class SwingTrajectoryGenerator:
    def __init__(self, step_height=0.08, step_length=0.1):
        """
        step_height: Maximum foot clearance during swing
        step_length: Forward step distance
        """
        self.step_height = step_height
        self.step_length = step_length
        
    def compute_swing_trajectory(self, phase_in_swing, v_des=0.0):
        """
        Compute foot position during swing phase
        phase_in_swing: Normalized phase [0, 1]
        v_des: Desired forward velocity
        """
        s = phase_in_swing
        
        # Bezier curve for smooth trajectory
        # Control points for x trajectory
        x_start = -self.step_length / 2
        x_end = self.step_length / 2
        
        # Adjust for desired velocity
        x_end += v_des * 0.1  # Velocity feedforward
        
        # Horizontal position (5th order polynomial for smooth acceleration)
        x = x_start + (x_end - x_start) * (
            10 * s**3 - 15 * s**4 + 6 * s**5
        )
        
        # Vertical position (sinusoidal for smooth lift)
        z = self.step_height * np.sin(np.pi * s)
        
        # Lateral position (usually zero for straight walking)
        y = 0.0
        
        return np.array([x, y, z])
    
    def compute_swing_velocity(self, phase_in_swing, omega, v_des=0.0):
        """
        Compute foot velocity during swing phase
        """
        s = phase_in_swing
        ds_dt = omega / (2 * np.pi * (1 - 0.5))  # Assuming duty_factor = 0.5
        
        # Derivatives of position trajectories
        x_start = -self.step_length / 2
        x_end = self.step_length / 2 + v_des * 0.1
        
        # dx/ds * ds/dt
        dx_ds = (x_end - x_start) * (
            30 * s**2 - 60 * s**3 + 30 * s**4
        )
        vx = dx_ds * ds_dt
        
        # dz/ds * ds/dt
        dz_ds = self.step_height * np.pi * np.cos(np.pi * s)
        vz = dz_ds * ds_dt
        
        return np.array([vx, 0.0, vz])
```

### 2. Stance Foot Trajectory

During stance, the foot maintains ground contact and moves backward relative to the body:

**Raibert Heuristic**: Foot placement for balance
```
x_foot = v_des * T_stance/2 + k_p * (v_actual - v_des)
```

```python
class StanceTrajectoryGenerator:
    def __init__(self):
        self.k_p = 0.03  # Position feedback gain
        self.k_d = 0.01  # Velocity feedback gain
        
    def compute_stance_trajectory(self, phase_in_stance, v_des=0.0, v_actual=0.0):
        """
        Compute foot position during stance phase
        phase_in_stance: Normalized phase [0, 1]
        """
        s = phase_in_stance
        
        # Linear motion backward relative to body
        x = v_des * 0.5 * (1 - 2*s)
        
        # Add feedback for stability (Raibert heuristic)
        x += self.k_p * (v_actual - v_des)
        
        # Maintain ground contact (z = 0)
        z = 0.0
        y = 0.0
        
        return np.array([x, y, z])
    
    def compute_ground_reaction_force(self, phase_in_stance, body_weight):
        """
        Compute desired ground reaction force profile
        """
        # Simple model: constant force during stance
        # More sophisticated: use optimal force distribution
        
        # Vertical force (support body weight)
        fz = body_weight / 2  # Assuming 2 legs in stance (trot)
        
        # Horizontal force for propulsion
        # Peak at mid-stance
        fx = 0.1 * body_weight * np.sin(np.pi * phase_in_stance)
        
        return np.array([fx, 0.0, fz])
```

### 3. Complete Foot Trajectory

Combining stance and swing phases:

```python
class FootTrajectoryGenerator:
    def __init__(self, gait_params):
        """
        gait_params: Dictionary with gait parameters
        """
        self.swing_gen = SwingTrajectoryGenerator(
            step_height=gait_params.get('step_height', 0.08),
            step_length=gait_params.get('step_length', 0.1)
        )
        self.stance_gen = StanceTrajectoryGenerator()
        self.duty_factor = gait_params.get('duty_factor', 0.5)
        
        # Trajectory blending parameters
        self.blend_duration = 0.05  # 5% of phase for smooth transition
        
    def compute_foot_trajectory(self, phase, v_des=0.0, v_actual=0.0):
        """
        Compute complete foot trajectory based on phase
        """
        if is_stance_phase(phase, self.duty_factor):
            s = get_phase_in_stance(phase, self.duty_factor)
            pos = self.stance_gen.compute_stance_trajectory(s, v_des, v_actual)
            
            # Add small vertical offset for ground clearance
            pos[2] = -0.35  # Default standing height
            
        else:
            s = get_phase_in_swing(phase, self.duty_factor)
            swing_pos = self.swing_gen.compute_swing_trajectory(s, v_des)
            
            # Add to default standing position
            pos = swing_pos + np.array([0, 0, -0.35])
            
        return pos
    
    def compute_smooth_transition(self, phase, v_des=0.0):
        """
        Smooth transition between stance and swing using sigmoid blending
        """
        # Compute both trajectories
        s_stance = get_phase_in_stance(phase, self.duty_factor)
        s_swing = get_phase_in_swing(phase, self.duty_factor)
        
        stance_pos = self.stance_gen.compute_stance_trajectory(s_stance, v_des)
        swing_pos = self.swing_gen.compute_swing_trajectory(s_swing, v_des)
        
        # Compute blending factor
        transition_phase = 2 * np.pi * self.duty_factor
        blend_start = transition_phase - self.blend_duration
        blend_end = transition_phase + self.blend_duration
        
        if blend_start <= phase <= blend_end:
            # Sigmoid blending
            t = (phase - blend_start) / (2 * self.blend_duration)
            alpha = 1 / (1 + np.exp(-10 * (t - 0.5)))
            
            # Blend positions
            pos = (1 - alpha) * stance_pos + alpha * swing_pos
        elif phase < blend_start:
            pos = stance_pos
        else:
            pos = swing_pos
            
        return pos
```

## Advanced Trajectory Modulation

### 1. Adaptive Frequency Oscillator (AFO)

Adapt phase frequency based on sensory feedback:

```python
class AdaptiveFrequencyOscillator:
    """
    Adaptive oscillator that synchronizes with external signals
    Based on Righetti & Ijspeert (2006)
    """
    def __init__(self, natural_freq=1.0, learning_rate=1.0):
        self.omega_0 = 2 * np.pi * natural_freq  # Natural frequency
        self.omega = self.omega_0  # Current frequency
        self.phase = 0.0
        self.learning_rate = learning_rate
        
    def update(self, dt, feedback_signal=0.0):
        """
        Update phase with frequency adaptation
        feedback_signal: External signal to synchronize with (e.g., foot contact)
        """
        # Frequency adaptation rule
        phase_error = feedback_signal * np.sin(self.phase)
        self.omega += self.learning_rate * phase_error * dt
        
        # Limit frequency range
        self.omega = np.clip(self.omega, 0.5 * self.omega_0, 2.0 * self.omega_0)
        
        # Update phase
        self.phase += self.omega * dt
        self.phase = self.phase % (2 * np.pi)
        
        return self.phase
    
    def reset(self):
        """Reset to natural frequency"""
        self.omega = self.omega_0
        self.phase = 0.0
```

### 2. Central Pattern Generator (CPG) Network

Network of coupled oscillators for robust gait generation:

```python
class CPGNetwork:
    """
    Central Pattern Generator using coupled oscillators
    Each leg has its own oscillator coupled to others
    """
    def __init__(self, gait_pattern='TROT'):
        self.oscillators = {
            'FR': AdaptiveFrequencyOscillator(),
            'FL': AdaptiveFrequencyOscillator(),
            'RR': AdaptiveFrequencyOscillator(),
            'RL': AdaptiveFrequencyOscillator()
        }
        
        # Coupling weights between oscillators
        self.set_coupling_weights(gait_pattern)
        
        # Phase offsets
        self.phase_offsets = GaitPattern.get_phase_offsets(gait_pattern)
        
    def set_coupling_weights(self, gait_pattern):
        """
        Set coupling weights based on gait pattern
        Strong coupling maintains phase relationships
        """
        if gait_pattern == 'TROT':
            # Strong diagonal coupling
            self.coupling = {
                ('FR', 'RL'): 2.0,  # Same phase
                ('FL', 'RR'): 2.0,  # Same phase
                ('FR', 'FL'): -2.0,  # Opposite phase
                ('FR', 'RR'): -2.0,  # Opposite phase
            }
        elif gait_pattern == 'WALK':
            # Sequential coupling
            self.coupling = {
                ('FR', 'FL'): 1.0,
                ('FL', 'RR'): 1.0,
                ('RR', 'RL'): 1.0,
                ('RL', 'FR'): 1.0,
            }
        else:
            self.coupling = {}
            
    def update(self, dt, foot_contacts=None):
        """
        Update all oscillators with coupling
        foot_contacts: Dictionary of boolean contact states
        """
        if foot_contacts is None:
            foot_contacts = {leg: False for leg in self.oscillators}
            
        # Compute coupling terms
        coupling_terms = {leg: 0.0 for leg in self.oscillators}
        
        for (leg1, leg2), weight in self.coupling.items():
            phase_diff = self.oscillators[leg2].phase - self.oscillators[leg1].phase
            coupling_terms[leg1] += weight * np.sin(phase_diff)
            
        # Update each oscillator
        phases = {}
        for leg, osc in self.oscillators.items():
            # Add sensory feedback (foot contact)
            feedback = 1.0 if foot_contacts[leg] else 0.0
            
            # Update with coupling
            osc.omega += coupling_terms[leg] * dt
            phases[leg] = osc.update(dt, feedback)
            
        return phases
    
    def get_leg_phases(self):
        """Get current phase for each leg with offsets"""
        phases = {}
        for leg, osc in self.oscillators.items():
            phases[leg] = (osc.phase + self.phase_offsets[leg]) % (2 * np.pi)
        return phases
```

### 3. Terrain Adaptation

Modify trajectories based on terrain:

```python
class TerrainAdaptiveTrajectory:
    """
    Adapt foot trajectories based on terrain feedback
    """
    def __init__(self, base_trajectory_gen):
        self.base_gen = base_trajectory_gen
        self.terrain_memory = {}  # Store terrain height at different positions
        
    def update_terrain_map(self, foot_position, measured_height):
        """
        Update internal terrain map based on foot contacts
        """
        grid_key = (
            int(foot_position[0] * 10),  # 10cm grid resolution
            int(foot_position[1] * 10)
        )
        self.terrain_memory[grid_key] = measured_height
        
    def predict_terrain_height(self, position):
        """
        Predict terrain height at given position
        """
        grid_key = (
            int(position[0] * 10),
            int(position[1] * 10)
        )
        
        if grid_key in self.terrain_memory:
            return self.terrain_memory[grid_key]
        
        # Default to flat ground
        return 0.0
        
    def compute_adaptive_trajectory(self, phase, next_foot_position):
        """
        Adapt trajectory based on predicted terrain
        """
        # Get base trajectory
        base_pos = self.base_gen.compute_foot_trajectory(phase)
        
        # Predict terrain at next foot position
        terrain_height = self.predict_terrain_height(next_foot_position)
        
        # Adapt swing height based on terrain
        if not is_stance_phase(phase):
            # Increase clearance for rough terrain
            terrain_roughness = self.estimate_terrain_roughness()
            clearance_factor = 1.0 + 0.5 * terrain_roughness
            
            base_pos[2] *= clearance_factor
            
        # Adapt foot placement height
        base_pos[2] += terrain_height
        
        return base_pos
    
    def estimate_terrain_roughness(self):
        """
        Estimate terrain roughness from memory
        """
        if len(self.terrain_memory) < 2:
            return 0.0
            
        heights = list(self.terrain_memory.values())
        return np.std(heights)
```

## Implementation Example

### Complete Walking Controller

```python
class PhaseBasedWalkingController:
    """
    Complete phase-based walking controller for quadruped
    """
    def __init__(self, gait_type='TROT', frequency=2.0):
        # Gait parameters
        self.gait_params = {
            'step_height': 0.08,
            'step_length': 0.1,
            'duty_factor': 0.5 if gait_type == 'TROT' else 0.75
        }
        
        # Initialize components
        self.cpg = CPGNetwork(gait_type)
        self.foot_traj_generators = {
            leg: FootTrajectoryGenerator(self.gait_params)
            for leg in ['FR', 'FL', 'RR', 'RL']
        }
        
        # Controller state
        self.time = 0.0
        self.desired_velocity = np.array([0.5, 0.0, 0.0])  # m/s
        self.actual_velocity = np.array([0.0, 0.0, 0.0])
        
        # Leg offsets in body frame
        self.leg_offsets = {
            'FR': np.array([0.1881, -0.08505, 0]),
            'FL': np.array([0.1881, 0.08505, 0]),
            'RR': np.array([-0.1881, -0.08505, 0]),
            'RL': np.array([-0.1881, 0.08505, 0])
        }
        
    def update(self, dt, foot_contacts, body_velocity):
        """
        Main control update
        """
        self.time += dt
        self.actual_velocity = body_velocity
        
        # Update CPG network
        leg_phases = self.cpg.update(dt, foot_contacts)
        
        # Compute foot trajectories for each leg
        foot_positions = {}
        for leg, phase in leg_phases.items():
            # Get trajectory in leg frame
            traj_leg = self.foot_traj_generators[leg].compute_foot_trajectory(
                phase,
                v_des=self.desired_velocity[0],
                v_actual=self.actual_velocity[0]
            )
            
            # Transform to body frame
            foot_positions[leg] = self.leg_offsets[leg] + traj_leg
            
        return foot_positions
    
    def set_command(self, velocity_command):
        """Set desired velocity"""
        self.desired_velocity = velocity_command
        
    def get_gait_info(self):
        """Get current gait information for debugging"""
        phases = self.cpg.get_leg_phases()
        
        info = {
            'time': self.time,
            'phases': phases,
            'stance_legs': [
                leg for leg, phase in phases.items()
                if is_stance_phase(phase, self.gait_params['duty_factor'])
            ],
            'desired_velocity': self.desired_velocity,
            'actual_velocity': self.actual_velocity
        }
        
        return info
```

## Integration with Your walk.py

Here's how to integrate phase-based trajectories into your existing code:

```python
# Enhanced version of your Controller class
class EnhancedController(Controller):
    def __init__(self, path):
        super().__init__(path)
        
        # Initialize phase-based controller
        self.phase_controller = PhaseBasedWalkingController(
            gait_type='TROT',
            frequency=2.0  # 2 Hz gait frequency
        )
        
        # Track foot contacts
        self.foot_contacts = {
            'FR': False, 'FL': False,
            'RR': False, 'RL': False
        }
        
    def detect_foot_contacts(self):
        """
        Detect foot contacts from force sensors or geometry
        """
        # Simple height-based detection
        # In practice, use force sensors
        threshold_height = 0.02  # 2cm above ground
        
        # This is pseudocode - adapt to your MuJoCo model
        # foot_heights = self.get_foot_heights()
        # for leg in ['FR', 'FL', 'RR', 'RL']:
        #     self.foot_contacts[leg] = foot_heights[leg] < threshold_height
        
        return self.foot_contacts
    
    def compute_leg_ik_phase_based(self):
        """
        Compute IK using phase-based trajectories
        """
        # Get body velocity (simplified)
        body_velocity = self.data.qvel[0:3]
        
        # Detect contacts
        foot_contacts = self.detect_foot_contacts()
        
        # Update phase controller
        dt = 0.002  # MuJoCo timestep
        foot_positions = self.phase_controller.update(
            dt, foot_contacts, body_velocity
        )
        
        # Convert foot positions to joint angles using IK
        joint_angles = []
        for leg in ['FR', 'FL', 'RR', 'RL']:
            # Your existing IK code
            pos = foot_positions[leg]
            # ... IK computation ...
            # joint_angles.extend([hip_abd, hip_flex, knee])
            
        return joint_angles
```

## Visualization and Analysis

```python
def visualize_phase_trajectories():
    """
    Visualize foot trajectories for different gaits
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Parameters
    gait_params = {
        'step_height': 0.08,
        'step_length': 0.1,
        'duty_factor': 0.5
    }
    
    generator = FootTrajectoryGenerator(gait_params)
    
    # Generate trajectory over one cycle
    phases = np.linspace(0, 2*np.pi, 100)
    positions = []
    
    for phase in phases:
        pos = generator.compute_foot_trajectory(phase, v_des=0.5)
        positions.append(pos)
        
    positions = np.array(positions)
    
    # Plot X-Z trajectory (side view)
    axes[0, 0].plot(positions[:, 0], positions[:, 2])
    axes[0, 0].set_xlabel('X (m)')
    axes[0, 0].set_ylabel('Z (m)')
    axes[0, 0].set_title('Foot Trajectory (Side View)')
    axes[0, 0].grid(True)
    
    # Plot phase vs position
    axes[0, 1].plot(phases, positions[:, 0], label='X')
    axes[0, 1].plot(phases, positions[:, 2], label='Z')
    axes[0, 1].set_xlabel('Phase (rad)')
    axes[0, 1].set_ylabel('Position (m)')
    axes[0, 1].set_title('Position vs Phase')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot gait diagram
    legs = ['FR', 'FL', 'RR', 'RL']
    offsets = GaitPattern.TROT
    
    for i, leg in enumerate(legs):
        leg_phases = (phases + offsets[leg]) % (2*np.pi)
        stance = [is_stance_phase(p, gait_params['duty_factor']) for p in leg_phases]
        
        # Create binary plot
        axes[0, 2].fill_between(
            phases, i, i+1,
            where=stance,
            color='blue',
            alpha=0.5,
            label=leg if i == 0 else ""
        )
        
    axes[0, 2].set_xlabel('Phase (rad)')
    axes[0, 2].set_ylabel('Leg')
    axes[0, 2].set_yticks(range(len(legs)))
    axes[0, 2].set_yticklabels(legs)
    axes[0, 2].set_title('Gait Diagram (Trot)')
    axes[0, 2].grid(True)
    
    # Plot CPG coupling
    cpg = CPGNetwork('TROT')
    time = np.linspace(0, 5, 500)
    dt = time[1] - time[0]
    
    phase_history = {leg: [] for leg in legs}
    
    for t in time:
        phases = cpg.update(dt)
        for leg in legs:
            phase_history[leg].append(phases[leg])
            
    for leg in legs:
        axes[1, 0].plot(time, phase_history[leg], label=leg)
        
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Phase (rad)')
    axes[1, 0].set_title('CPG Phase Evolution')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Plot phase differences
    axes[1, 1].plot(time, np.array(phase_history['FR']) - np.array(phase_history['FL']), label='FR-FL')
    axes[1, 1].plot(time, np.array(phase_history['FR']) - np.array(phase_history['RL']), label='FR-RL')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Phase Difference (rad)')
    axes[1, 1].set_title('Phase Synchronization')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot velocity modulation
    velocities = [0.0, 0.5, 1.0, 1.5]
    colors = plt.cm.viridis(np.linspace(0, 1, len(velocities)))
    
    for v, color in zip(velocities, colors):
        positions = []
        for phase in phases:
            pos = generator.compute_foot_trajectory(phase, v_des=v)
            positions.append(pos)
        positions = np.array(positions)
        
        axes[1, 2].plot(positions[:, 0], positions[:, 2], 
                       color=color, label=f'v={v} m/s')
        
    axes[1, 2].set_xlabel('X (m)')
    axes[1, 2].set_ylabel('Z (m)')
    axes[1, 2].set_title('Velocity Adaptation')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    plt.show()

# Run visualization
if __name__ == "__main__":
    visualize_phase_trajectories()
```

## Key Equations Summary

1. **Phase Evolution**: φ(t) = ωt mod 2π

2. **Gait Phase Offset**: φ_leg = φ_base + φ_offset

3. **Duty Factor**: β = T_stance / T_cycle

4. **Swing Trajectory**:
   - x(s) = x_start + (x_end - x_start) * polynomial(s)
   - z(s) = h * sin(πs)

5. **Stance Trajectory**: x(s) = v * T_stance * (0.5 - s)

6. **Raibert Heuristic**: x_foot = v*T/2 + k(v_actual - v_desired)

7. **CPG Coupling**: dφ_i/dt = ω_i + Σ_j K_ij * sin(φ_j - φ_i)

8. **Adaptive Frequency**: dω/dt = α * F(t) * sin(φ)

## Exercises

1. **Basic Phase Control**: Modify your walk.py to use phase-based trajectories instead of simple circular paths

2. **Gait Transitions**: Implement smooth transitions between trot and walk gaits

3. **Speed Control**: Add velocity command tracking using the Raibert heuristic

4. **Terrain Adaptation**: Implement step height adjustment based on terrain feedback

5. **CPG Network**: Build a full CPG network and test synchronization

6. **Frequency Adaptation**: Implement adaptive frequency based on desired speed

7. **Stability Analysis**: Analyze phase stability using Poincaré maps

8. **Energy Optimization**: Find optimal duty factors for different speeds

## Key Takeaways

- Phase-based control creates natural, periodic gaits
- Different gaits are just different phase relationships between legs
- Duty factor controls stance/swing timing
- Smooth trajectories prevent jerky motion
- CPG networks provide robust coordination
- Sensory feedback enables adaptation
- The Raibert heuristic provides simple velocity control
- Phase synchronization emerges from coupling

This approach is used in most modern quadruped controllers, from MIT Cheetah to Boston Dynamics Spot!
