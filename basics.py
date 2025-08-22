import mujoco
from mujoco import viewer
import numpy as np
import time

# mjpython /Users/zoe/CAVELAB/DOG/basics.py to run this script

model = mujoco.MjModel.from_xml_path("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
data = mujoco.MjData(model)

with viewer.launch_passive(model, data) as v:
    while v.is_running():
        data.ctrl[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # Apply control and step the simulation 
        accel = data.sensordata[0:3]
        gyro = data.sensordata[3:6]
        print("accel:", accel, "gyro:", gyro)
        mujoco.mj_step(model, data)
        time.sleep(0.01)
        v.sync()


