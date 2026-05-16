import mujoco
import mujoco.viewer
import numpy as np

model = mujoco.MjModel.from_xml_path("unitree_h1\\scene.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()