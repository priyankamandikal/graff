from gym.envs.registration import register

# Grasp an object decomposed into convex meshes
register(
    id='graff-v0',
    entry_point='envs.mj_envs.dex_manip:GraffV0',
    max_episode_steps=200,
)
from envs.mj_envs.dex_manip.graff import GraffV0
