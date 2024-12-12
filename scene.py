import mujoco
import time
import mujoco.viewer
import numpy as np

# from robot_descriptions.loaders.mujoco import load_robot_description

# model = load_robot_description("xarm7_mj_description")
model = mujoco.MjModel.from_xml_path("sim.xml")


data = mujoco.MjData(model)

# print(data.geom("name1"))

N = 500
q_start = data.qpos[0]
q_end = 4
q = np.linspace(q_start, q_end, num=N)
i = 0
mujoco.viewer.launch(model, data)
print(data.qpos)

# with mujoco.viewer.launch_passive(model, data) as viewer:
#     print(data.body("name1").cvel)
#     while viewer.is_running() and i < N:
#         data.qpos[0] = q[i]
#         mujoco.mj_step(model, data)
#         viewer.sync()
#         time.sleep(0.05)
#         i+=1
#    # data.body("name1").cvel = [0, 0, 3, 0, 0, 0]
