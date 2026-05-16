import mujoco
import mujoco.viewer
import numpy as np

# scene_lower_body scene_upper_body
model = mujoco.MjModel.from_xml_path("unitree_h1\\scene_upper_body.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():

        mujoco.mj_step(model, data)
        viewer.sync()