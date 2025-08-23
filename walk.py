import mujoco
from mujoco import viewer
import numpy as np
import time

# mjpython /Users/zoe/CAVELAB/DOG/walk.py to run this script

class Controller: 
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)
        self.model.opt.gravity[0:3] = [0, 0, -9.81]
        self.model.geom_friction[0] = [1.5, 0.005, 0.0001] 
        self.model.geom_friction[1] = [1.5, 0.005, 0.0001]
        self.l_thigh = 0.213  
        self.l_calf = 0.213 
        self.l_hip = -0.05
        self.time = 0
    
    def trajectory_generator(self, t): 
        r = 0.05
        x = -r * np.cos(t) 
        y = 0
        z = -0.35 + r * np.sin(t) 
        target = np.array([x, y, z])
        return target
    
    def compute_leg_ik(self, base_target):
        FR_target = self.trajectory_generator(self.time)       
        FL_target = self.trajectory_generator(self.time + np.pi) 
        BR_target = self.trajectory_generator(self.time + np.pi)    
        BL_target = self.trajectory_generator(self.time) 

        l1, l2 = self.l_thigh, self.l_calf
        
        # FR leg (Front Right)
        x1, y1, z1 = FR_target[0], FR_target[1] - self.l_hip, FR_target[2]
        hip_abd1 = np.arctan2(-y1, -z1)
        z_eff1 = -np.sqrt(y1**2 + z1**2)
        d1 = np.sqrt(x1**2 + z_eff1**2)
        d1 = np.clip(d1, 0.1, l1 + l2 - 0.01)
        theta1 = np.arccos(np.clip((l1**2 + l2**2 - d1**2) / (2 * l1 * l2), -1, 1))
        phi1 = np.arctan2(x1, -z_eff1) + np.arccos(np.clip((l1**2 + d1**2 - l2**2) / (2 * l1 * d1), -1, 1))
        
        # FL leg (Front Left)
        x2, y2, z2 = FL_target[0], FL_target[1] + self.l_hip, FL_target[2]
        hip_abd2 = np.arctan2(-y2, -z2)
        z_eff2 = -np.sqrt(y2**2 + z2**2)
        d2 = np.sqrt(x2**2 + z_eff2**2)
        d2 = np.clip(d2, 0.1, l1 + l2 - 0.01)
        theta2 = np.arccos(np.clip((l1**2 + l2**2 - d2**2) / (2 * l1 * l2), -1, 1))
        phi2 = np.arctan2(x2, -z_eff2) + np.arccos(np.clip((l1**2 + d2**2 - l2**2) / (2 * l1 * d2), -1, 1))

        # BR leg (Back Right)
        x3, y3, z3 = BR_target[0], BR_target[1] - self.l_hip, BR_target[2]
        hip_abd3 = np.arctan2(-y3, -z3)
        z_eff3 = -np.sqrt(y3**2 + z3**2)
        d3 = np.sqrt(x3**2 + z_eff3**2)
        d3 = np.clip(d3, 0.1, l1 + l2 - 0.01)
        theta3 = np.arccos(np.clip((l1**2 + l2**2 - d3**2) / (2 * l1 * l2), -1, 1))
        phi3 = np.arctan2(x3, -z_eff3) + np.arccos(np.clip((l1**2 + d3**2 - l2**2) / (2 * l1 * d3), -1, 1))

        # BL leg (Back Left)
        x4, y4, z4 = BL_target[0], BL_target[1] + self.l_hip, BL_target[2]
        hip_abd4 = np.arctan2(-y4, -z4)
        z_eff4 = -np.sqrt(y4**2 + z4**2)
        d4 = np.sqrt(x4**2 + z_eff4**2)
        d4 = np.clip(d4, 0.1, l1 + l2 - 0.01)
        theta4 = np.arccos(np.clip((l1**2 + l2**2 - d4**2) / (2 * l1 * l2), -1, 1))
        phi4 = np.arctan2(x4, -z_eff4) + np.arccos(np.clip((l1**2 + d4**2 - l2**2) / (2 * l1 * d4), -1, 1))
        
        return hip_abd1, phi1, theta1, hip_abd2, phi2, theta2, hip_abd3, phi3, theta3, hip_abd4, phi4, theta4

    def compute_torques(self):
        hip_abd1, phi1, theta1, hip_abd2, phi2, theta2, hip_abd3, phi3, theta3, hip_abd4, phi4, theta4 = self.compute_leg_ik(None)
        # Each leg has 3 joints: hip_abduction, hip_flexion, knee_flexion
        control_values = [
            hip_abd1, phi1, -(np.pi - theta1),  # FR leg
            hip_abd2, phi2, -(np.pi - theta2),  # FL leg  
            hip_abd3, phi3, -(np.pi - theta3),  # BR leg
            hip_abd4, phi4, -(np.pi - theta4)   # BL leg
        ]
        available_controls = min(len(control_values), self.data.ctrl.shape[0])
        self.data.ctrl[0:available_controls] = control_values[:available_controls]  

    def run_interactive(self):
        mujoco.mj_resetData(self.model, self.data)
        with viewer.launch_passive(self.model, self.data) as sim:
            while sim.is_running():
                self.compute_torques()
                mujoco.mj_step(self.model, self.data)
                sim.sync()
                time.sleep(0.01)
                self.time += 0.05  


controller = Controller("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
controller.run_interactive()