from gymnasium.envs.registration import register
import mavis_mujoco_gym.envs.mavis_base.mavis_base_env
import mavis_mujoco_gym.envs.pick_and_place.pick_and_place

register(
    id="RealWorldSystem-v0",
    entry_point="mavis_mujoco_gym.envs.realworld_system.realworld_system:RealWorldSystem",
)


register(
    id="MAVISBase-v0",
    entry_point="mavis_mujoco_gym.envs.mavis_base.mavis_base_env:MAVISBaseEnv",
)

register(
    id="PickAndPlace-v0",
    entry_point="mavis_mujoco_gym.envs.pick_and_place.pick_and_place:PickAndPlace",
    max_episode_steps=500
)

register(
    id="GoalBasedPickAndPlace-v0",
    entry_point="mavis_mujoco_gym.envs.pick_and_place.goal_based_pick_and_place:GoalBasedPickAndPlace",
    max_episode_steps=500
)