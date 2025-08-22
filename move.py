import mujoco
from mujoco import viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
data = mujoco.MjData(model)

def simple_stand_controller():
    """
    Simple PD controller to make the robot stand
    This is what you'll replace with RL!
    """
    
    # Reset and set initial height
    mujoco.mj_resetData(model, data)
    data.qpos[2] = 0.3  # 30cm height
    
    # Target joint angles for standing (roughly)
    target_angles = np.array([
        0.0, 0.7, -1.4,  # FR leg
        0.0, 0.7, -1.4,  # FL leg
        0.0, 0.7, -1.4,  # RR leg
        0.0, 0.7, -1.4,  # RL leg
    ])
    
    with viewer.launch_passive(model, data) as v:
        while v.is_running():
            # PD controller
            kp = 20.0  # Proportional gain
            kd = 1.0   # Derivative gain
            
            joint_pos = data.qpos[7:19]
            joint_vel = data.qvel[6:18]
            
            # Calculate torques
            torques = kp * (target_angles - joint_pos) - kd * joint_vel
            
            # Apply torques
            data.ctrl[:] = torques
            
            # Step simulation
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    viewer.launch(model, data)