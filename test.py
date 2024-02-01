import mujoco_py
import os  

def simulate_mujoco():
    model_path = 'envs/mj_envs/dex_manip/assets/contactdb/env_xmls/cup_vhacd.xml' # Replace with the actual path to your Mujoco model
    print("1")
    sim = mujoco_py.MjSim(mujoco_py.load_model_from_path(model_path))
    print("2")
    print(sim.render(width=64, height=64,
                     camera_name='egocentric',
                     depth=True, 
                    #  device_id=1
                     ))  # Adjust width and height as needed
    print("3")
try:
    simulate_mujoco()
except mujoco_py.MujocoException as e:
    print(f"Error: {e}")
    
    
# import mujoco_py
# import mujoco_viewer

# model = mujoco.MjModel.from_xml_path('humanoid.xml')
# data = mujoco.MjData(model)

# # create the viewer object
# viewer = mujoco_viewer.MujocoViewer(model, data)

# # simulate and render
# for _ in range(10000):
#     if viewer.is_alive:
#         mujoco.mj_step(model, data)
#         viewer.render()
#     else:
#         break

# # close
# viewer.close()


# import mujoco_py
# import os
# mj_path = mujoco_py.utils.discover_mujoco()
# xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
# model = mujoco_py.load_model_from_path(xml_path)
# sim = mujoco_py.MjSim(model)

# print(sim.data.qpos)
# # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]

# sim.step()
# print(sim.data.qpos)
# # [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
# #   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
# #   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
# #  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
# #  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
# #  -2.22862221e-05]