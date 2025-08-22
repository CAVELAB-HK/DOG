import mujoco
from mujoco import viewer
import numpy as np
import time

# mjpython /Users/zoe/CAVELAB/DOG/basics.py to run this script

class Controller: 
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.target_pos = np.zeros(12)
        self.kp = 10.0
        self.kd = 1.0 

    def reset(self): 
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = 0.3
        # Standing position
        stand_angles = np.array([
            # one leg (hip_abduct, hip_flex, knee_flex)
            0.0, 0.7, -1.4, # FR
            0.0, 0.7, -1.4, # FL
            0.0, 0.7, -1.4, # RR
            0.0, 0.7, -1.4  # RL
        ])
        self.data.qpos[7:19] = stand_angles
        self.target_pos = stand_angles

    def compute_torques(self): 
        # PD control to reach target positions
        # u(t) = Kp * e(t) + Kd * e'(t)
        pos_error = self.target_pos - self.data.qpos[7:19]
        vel_error = -self.data.qvel[6:18]
        # 12 leg joint angles start at index 7 in qpos and their velocities start at index 6 in qvel
        torques = self.kp * pos_error + self.kd * vel_error
        return torques
    
    def step(self): 
        self.data.ctrl[:] = self.compute_torques()
        mujoco.mj_step(self.model, self.data)

    def run_interactive(self): 
        self.reset()
        with viewer.launch_passive(self.model, self.data) as sim:
            while sim.is_running(): 
                self.step()
                time.sleep(0.01)
                sim.sync()

controller = Controller("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
controller.run_interactive()





