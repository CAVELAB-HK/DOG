import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
data = mujoco.MjData(model)

print(f"Body pos: {data.xpos}, Body quat: {data.xquat}")
print(f"Joint pos: {data.qpos}, Joint vel: {data.qvel}")
print(f"Controls: {data.ctrl}, Sensors: {data.sensordata}")

viewer.launch(model, data)