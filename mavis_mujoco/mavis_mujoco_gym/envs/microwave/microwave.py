import numpy as np
from typing import Optional
import enum
from scipy.spatial.transform import Rotation as R
from mavis_mujoco_gym.envs.mavis_base import MAVISBaseEnv
from mavis_mujoco_gym.utils.mujoco_utils import MujocoModelNames
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer
import mavis_mujoco_gym.utils.mavis_utils as mavis_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# DEFAULT_CAMERA_CONFIG = {
#     "distance": 2.0,
#     "azimuth": 90.0,
#     "elevation": -50.0,
#     "lookat": np.array([0.45, 0.0, 0.0]),
# }

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 180.0,
    "elevation": -50.0,
    "lookat": np.array([-0.5, -0.025, 1.6]),
}

class Microwave(MAVISBaseEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50
    }

    def __init__(
            self,
            model_path="../MJCF/microwave.xml",
            frame_skip=20,
            robot_noise_ratio: float = 0.01,
            obs_space_type: str = "config_space",
            # Choose among "config_space", "eef_euler_space", "eef_quat_space", "lookat_euler_space", "lookat_quat_space"
            act_space_type: str = "config_space",
            # Choose among "config_space", "eef_euler_space", "eef_quat_space", "lookat_euler_space", "lookat_quat_space"
            action_normalization: bool = True,
            observation_normalization: bool = True,
            render_mode="human",
            render_fps: int = 50,
            img_width: int = 640,
            img_height: int = 480,
            camera_name: str = "realsense_camera",
            default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
            enable_rgb: bool = True,
            enable_depth: bool = True,
            use_cheating_observation: bool = False,
            max_episode_steps=None,
            use_teleoperation: bool = False,
            radian_threshold: float = -0.5,
    ):
        self.metadata["render_fps"] = render_fps
        super().__init__(
            model_path=model_path,
            frame_skip=frame_skip,
            robot_noise_ratio=robot_noise_ratio,
            obs_space_type=obs_space_type,
            act_space_type=act_space_type,
            action_normalization=action_normalization,
            observation_normalization=observation_normalization,
            render_mode=render_mode,
            render_fps=render_fps,
            img_width=img_width,
            img_height=img_height,
            camera_name=camera_name,
            default_camera_config=default_camera_config,
            enable_rgb=enable_rgb,
            enable_depth=enable_depth,
            use_cheating_observation=use_cheating_observation,
            cheating_observation_dim=9,
            use_teleoperation=use_teleoperation,
        )
        self.radian_threshold = radian_threshold
        self.set_cheating_obs_bounds()

        model_names = MujocoModelNames(self.model)
        site_name2id = model_names.site_name2id
        joint_name2id = model_names.joint_name2id
        body_name2id = model_names.body_name2id
        # camera_name2id = model_names.camera_name2id

        self.grasping_point_site_id = site_name2id["link_tcp"]
        self.left_gripper_finger_body_id = body_name2id["left_finger"]
        self.right_gripper_finger_body_id = body_name2id["right_finger"]
        self.microwave_joint_id = joint_name2id["microwave"]
        self.handle_site_id = site_name2id["microhandle_site"]

        model_names = MujocoModelNames(self.model)
        self.geom_ids_of_interest = self.get_geom_ids_of_interest(model_names)

        self.segmentation_viewer = OffScreenViewer(model=self.model,
                                                   data=self.data,
                                                   width=img_width,
                                                   height=img_height)

        self.max_episode_steps = max_episode_steps
        self.current_episode_step = 0

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = super().step(action)

            self.current_episode_step += 1

            # if terminated:
            #     if self.is_success():
            #         reward += self.goal_complete_reward
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print("Step function encountered an error:", str(e))
            obs, info = self.reset()
            reward = -1000.0
            terminated = False
            truncated = False
            return obs, reward, terminated, truncated, info

    def _get_obs(self):
        obs = super()._get_obs()
        if self.use_cheating_observation:
            if self.observation_normalization:
                cheating_observation = self.normalize_cheating_observation(self.get_cheating_observation())
            else:
                cheating_observation = self.get_cheating_observation()

            if not self.enable_rgb and not self.enable_depth:
                obs = np.concatenate([obs, cheating_observation])
            else:
                obs["state"] = np.concatenate([obs["state"], cheating_observation])

        return obs

    def _get_info(self):
        info = {
            # "if_reached_block": self.if_reached_block(),
            "is_success": self.is_success(),
        }
        return info

    def _get_reward(self):
        return 0.0

    def _get_terminated(self):
        res = self.is_success()
        return bool(res)

    def _get_truncated(self):
        if self.max_episode_steps is None:
            return False
        return self.current_episode_step >= self.max_episode_steps - 1

    def get_cheating_observation(self):
        # This does not need offset on linear motor since it is already in the world frame
        # Grasping point position (2 dim)
        grasping_point_pos = self.get_grasping_point_pos()

        # Joint position (1 dim)
        joint_pos = np.array([self.get_microwave_joint_pos()]) #self.get_microwave_joint_pos()

        # Target position (1 dim)
        target_pos = np.array([self.radian_threshold]) # self.radian_threshold

        # Relative position between grasping point and block (3 dim)
        # relative_pos_grasping_point_block = grasping_point_pos - joint_pos

        # Relative position between joint and target (1 dim)
        relative_joint_pos = joint_pos - target_pos

        # Handle position (2 dim)
        handle_pos = self.get_microwave_handle_pos()

        # Relative position between gripper and handle (2 dim)
        relative_pos_grasping_point_handle = handle_pos - grasping_point_pos

        # Grasping point axis angle rotation (4 dim)
        # grasping_point_axis_angle_rot = self.get_grasping_point_axis_angle_rot()

        # Block axis angle rotation (4 dim)
        # block_axis_angle_rot = self.get_block_axis_angle_rot()

        # cheating_observation = np.concatenate([grasping_point_pos,
        #                                        joint_pos,
        #                                        target_pos,
        #                                        relative_pos_grasping_point_block,
        #                                        relative_pos_block_target,
        #                                        grasping_point_axis_angle_rot,
        #                                        block_axis_angle_rot])

        cheating_observation = np.concatenate([joint_pos,
                                               target_pos,
                                               relative_joint_pos,
                                               handle_pos,
                                               relative_pos_grasping_point_handle,
                                               grasping_point_pos])
        
        return cheating_observation

    def set_cheating_obs_bounds(self):
        # self.cheating_obs_low = np.array([-0.77, -0.77, 0.0,
        #                                   -0.77, -0.77, 0.0,
        #                                   -0.77, -0.77, 0.0,
        #                                   -0.77, -0.77, 0.0,
        #                                   -0.77, -0.77, 0.0,
        #                                   -1.0, -1.0, -1.0, 0.0,
        #                                   -1.0, -1.0, -1.0, 0.0])

        # self.cheating_obs_high = np.array([0.77, 1.51, 1.037,
        #                                   0.77, 1.51, 1.037,
        #                                   0.77, 1.51, 1.037,
        #                                   0.77, 1.51, 1.037,
        #                                   0.77, 1.51, 1.037,
        #                                   1.0, 1.0, 1.0, np.pi,
        #                                   1.0, 1.0, 1.0, np.pi])

        self.cheating_obs_low = np.array([-1.00,
                                          -1.00,
                                          -1.00,
                                          -1.00, -1.00,
                                          -1.00, -1.00,
                                          -1.00, -1.00])

        self.cheating_obs_high = np.array([0.00,
                                           0.00,
                                           0.00,
                                           1.51, 1.51,
                                           1.51, 1.51,
                                           1.51, 1.51])

    def normalize_cheating_observation(self, cheating_observation):
        return 2 * (cheating_observation - self.cheating_obs_low) / (self.cheating_obs_high - self.cheating_obs_low) - 1

    def denormalize_cheating_observation(self, cheating_observation):
        return (cheating_observation + 1) * (self.cheating_obs_high - self.cheating_obs_low) / 2 + self.cheating_obs_low

    def get_geom_ids_of_interest(self, model_names: MujocoModelNames):
        geom_names2ids = model_names.geom_name2id
        geom_names_of_interest = ["gripper_base_link",
                                  "left_outer_knuckle",
                                  "left_finger",
                                  "left_inner_knuckle",
                                  "right_outer_knuckle",
                                  "right_finger",
                                  "right_inner_knuckle"]

        geom_ids_of_interest = [geom_names2ids[geom_name] for geom_name in geom_names_of_interest]
        return geom_ids_of_interest
    
    def get_microwave_handle_pos(self):
        return self.data.site_xpos[self.handle_site_id][:2]

    def get_microwave_joint_pos(self):
        return self.data.qpos[self.microwave_joint_id]
    
    def get_grasping_point_pos(self):
        return self.data.site_xpos[self.grasping_point_site_id][:2]

    def is_success(self):
        reached_target_radian = self.get_microwave_joint_pos()
        return reached_target_radian < self.radian_threshold

    def _reset_simulation(self):
        super()._reset_simulation()
        self.set_joint_angle(self.microwave_joint_id, 0.0)
        self.current_episode_step = 0

    def set_joint_pos(self, joint_id, pos):
        joint_pos = np.array([pos])

        self.data.qpos[joint_id] = joint_pos
        # self.data.qvel[joint_id] = 0.0
        # self.model.forward(self.data)

    def set_joint_angle(self):
        self.set_joint_pos(self.microwave_joint_id, 0.0)

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              ):
        obs, info = super().reset(seed=seed, options=options)
        self.set_joint_angle()
        return obs, info
    