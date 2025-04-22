from mavis_mujoco_gym.envs.mavis_base.mavis_base_env import MAVISBaseEnv
from mavis_mujoco_gym.envs.pick_and_place.pick_and_place import PickAndPlace
from mavis_mujoco_gym.envs.microwave.microwave import Microwave

ENV_DICT = {
    "MAVISBase-v0": MAVISBaseEnv,
    "PickAndPlace-v0": PickAndPlace,
    "Microwave-v0": Microwave
}

def create_env(env_id, env_configs):
    env = ENV_DICT[env_id](**env_configs)
    return env