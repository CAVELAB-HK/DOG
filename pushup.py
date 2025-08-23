import mujoco
from mujoco import viewer
import numpy as np
import time

# mjpython /Users/zoe/CAVELAB/DOG/pushup.py to run this script

class Controller: 
    def __init__(self, path):
        self.model = mujoco.MjModel.from_xml_path(path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.gravity[0:3] = [0, 0, -9.81] 
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

    def kneel(self): 
        kneel_angles = np.array([
            0.0, 2, -4, 
            0.0, 2, -4,
            0.0, 2, -4, 
            0.0, 2, -4 
        ])
        self.target_pos = kneel_angles

    def stand(self):
        stand_angles = np.array([
            0.0, 0.7, -1.4,
            0.0, 0.7, -1.4, 
            0.0, 0.7, -1.4, 
            0.0, 0.7, -1.4  
        ])
        self.target_pos = stand_angles

    def compute_torques(self): 
        # PD control to reach target positions
        # u(t) = Kp * e(t) + Kd * e'(t)
        pos_error = self.target_pos - self.data.qpos[7:19]
        vel_error = -self.data.qvel[6:18]

        in_contact = self.data.ncon > 0 

        if in_contact: 
            kp_con = self.kp * 0.8
            kd_con = self.kd * 1.5
            gravity_scale = 0.4
        else: 
            kp_con = self.kp
            kd_con = self.kd
            gravity_scale = 0.6

        # 12 leg joint angles start at index 7 in qpos and their velocities start at index 6 in qvel
        torques = kp_con * pos_error + kd_con * vel_error
        '''
        Gravity compensation using inverse dynamics
        Inverse dynamics: tau = M(q)*qdd + C(q,qd)*qd + G(q)
        τ = joint torques
        M(q) = inertia matrix
        C(q,q̇) = Coriolis and centrifugal forces
        G(q) = gravitational forces
        q, q̇, q̈ = positions, velocities, accelerations
        Counteracts the gravitational forces acting on your robot's joints
        We will use mj_rne (Recursive Newton-Euler), `mujoco.mj_rne(model, data, flg_acc, result)`
        For flg_acc, 0 for only gravity compensation, 1 for full inverse dynamics
        '''
        # This is how to compute full inverse dynamics
        '''
        self.data.qacc[:] = 0
        self.data.qacc[6:18] = desired_acc
        inv_dynamics = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, self.data, 1, inv_dynamics)
        '''

        # This is how to compute just gravity compensation
        gravity_comp = np.zeros(self.model.nv) # nv = number of degrees of freedom
        mujoco.mj_rne(self.model, self.data, 0, gravity_comp)

        total_torques = torques + 0.5 * gravity_comp[6:18] 
        return total_torques
    
    def step(self): 
        self.data.ctrl[:] = self.compute_torques()
        mujoco.mj_step(self.model, self.data)

    def run_interactive(self): 
        self.reset()
        step_count = 0
        with viewer.launch_passive(self.model, self.data) as sim:
            while sim.is_running(): 
                step_count += 1
                if step_count == 200:
                    self.kneel()
                if step_count == 400:
                    self.stand()
                    step_count = 0
                self.step()
                time.sleep(0.01)
                sim.sync()

controller = Controller("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
controller.run_interactive()





