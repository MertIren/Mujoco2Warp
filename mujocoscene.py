import mujoco
import time
import mujoco.viewer
import numpy as np
import mediapy as media
import cv2

# from robot_descriptions.loaders.mujoco import load_robot_description

# model = load_robot_description("xarm7_mj_description")
model = mujoco.MjModel.from_xml_path("sim.xml")


data = mujoco.MjData(model)
print(data.qfrc_applied)
# data.qfrc_applied[1] = 100
# data.xfrc_applied[2] = np.array([1000, 1000, 1000, 1000, 1000, 1000])

# print(data.geom("name1"))

# N = 500
# q_start = data.qpos[0]
# q_end = 4
# q = np.linspace(q_start, q_end, num=N)
# i = 0
# print(data.xfrc_applied)
# mujoco.viewer.launch(model, data)
# print(data.qpos)

duration = 5  # (seconds)
framerate = 60  # (Hz)

# Simulate and display video.
frames = []
mujoco.mj_resetData(model, data)  # Reset state and time.
with mujoco.Renderer(model) as renderer:
  data.xfrc_applied[1] = np.array([0, 10000, 0, 0, 0, 0])
  while data.time < duration:
    mujoco.mj_step(model, data)
    if len(frames) < data.time * framerate:
      renderer.update_scene(data, "side")
      pixels = renderer.render()
      frames.append(pixels)
    data.xfrc_applied[1] = np.array([0, 0, 0, 0, 0, 0])


def display_video(frames, fps=3):
    window_name = 'Video'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    for frame in frames:
        cv2.imshow(window_name, frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

display_video(frames, fps=framerate)

print(data.xpos)
# Target is 8.73