import mujoco
from mujoco import viewer

model = mujoco.MjModel.from_xml_path("/Users/zoe/CAVELAB/DOG/mujoco_menagerie/unitree_go1/scene.xml")
data = mujoco.MjData(model)


viewer.launch(model, data)