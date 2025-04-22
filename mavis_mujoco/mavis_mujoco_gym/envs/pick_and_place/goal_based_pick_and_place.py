import numpy as np
from typing import Optional
import enum
from collections import OrderedDict
from scipy.spatial.transform import Rotation as R
from mavis_mujoco_gym.envs.mavis_base import MAVISBaseEnv
from mavis_mujoco_gym.utils.mujoco_utils import MujocoModelNames
from gymnasium.envs.mujoco.mujoco_rendering import OffScreenViewer

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 90.0,
    "elevation": -50.0,
    "lookat": np.array([0.45, 0.0, 0.0]),
}

class State(enum.Enum):
    MOVE_TO_BLOCK = 0
    MOVE_TO_TARGET = 1

class GoalBasedPickAndPlace(MAVISBaseEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 12
    }

    def __init__(
            self,
            model_path="../MJCF/pick_and_place.xml",
            frame_skip=20,
            robot_noise_ratio: float = 0.01,
            obs_space_type: str = "config_space",
            # Choose among "config_space", "eef_euler_space", "eef_quat_space", "lookat_euler_space", "lookat_quat_space"
            act_space_type: str = "config_space",
            # Choose among "config_space", "eef_euler_space", "eef_quat_space", "lookat_euler_space", "lookat_quat_space"
            action_normalization: bool = True,
            observation_normalization: bool = True,
            render_mode="human",
            render_fps: int = 12,
            img_width: int = 640,
            img_height: int = 480,
            camera_name: str = "realsense_camera",
            default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
            enable_rgb: bool = True,
            enable_depth: bool = True,
            use_cheating_observation: bool = False,
            use_viewpoint_reward: bool = True,
            target_in_the_air: bool = False,
            random_block_target_pos: bool = True,
            block_x_range: list = [0.25, 0.65],
            block_y_range: list = [-0.35, 0.35],
            block_z: float = 0.025,
            target_x_range: list = [0.25, 0.65],
            target_y_range: list = [-0.35, 0.35],
            target_z_range: list = [0.02, 0.5],
            distance_threshold: float = 0.05,
            max_episode_steps=None,
    ):
        self.metadata["render_fps"] = render_fps
        self.distance_penalty_coefficient = 1.0
        self.viewpoint_selection_reward_coefficient = 10.0
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
            cheating_observation_dim=15,
            is_goal_env=True,
            achieved_goal_shape=(3,),
            desired_goal_shape=(3,),
        )

        self.set_cheating_obs_bounds()

        self.target_in_the_air = target_in_the_air
        self.random_block_target_pos = random_block_target_pos
        self.use_viewpoint_reward = use_viewpoint_reward
        self.block_x_range = block_x_range
        self.block_y_range = block_y_range
        self.block_z = block_z
        self.target_x_range = target_x_range
        self.target_y_range = target_y_range
        self.target_z_range = target_z_range
        self.distance_threshold = distance_threshold

        model_names = MujocoModelNames(self.model)
        site_name2id = model_names.site_name2id
        joint_name2id = model_names.joint_name2id
        body_name2id = model_names.body_name2id
        camera_name2id = model_names.camera_name2id
        self.grasping_point_site_id = site_name2id["link_tcp"]
        self.block_body_id = body_name2id["object0"]
        self.block_site_id = site_name2id["object0"]
        self.block_joint_id = joint_name2id["object0:joint"]
        self.target_body_id = body_name2id["target0"]
        self.target_site_id = site_name2id["target0"]
        self.realsense_camera_id = camera_name2id[camera_name]
        self.geom_ids_of_interest = self.get_geom_ids_of_interest(model_names)

        self.segmentation_viewer = OffScreenViewer(model=self.model,
                                                   data=self.data,
                                                   width=img_width,
                                                   height=img_height)

        self.max_episode_steps = max_episode_steps
        self.current_episode_step = 0

        self.current_state = State.MOVE_TO_BLOCK

    def step(self, action):
        try:
            obs_raw, reward, terminated, truncated, info = super().step(action)

            if self.current_state == State.MOVE_TO_BLOCK:
                achieved_goal = self.get_grasping_point_pos()
                desired_goal = self.get_block_pos()
                if self.if_reached_block():
                    self.current_state = State.MOVE_TO_TARGET
            elif self.current_state == State.MOVE_TO_TARGET:
                achieved_goal = self.get_block_pos()
                desired_goal = self.get_target_pos()
            else:
                raise ValueError("Invalid state")
            obs = OrderedDict(
                [
                    ("observation", obs_raw),
                    ("achieved_goal", achieved_goal),
                    ("desired_goal", desired_goal),
                ]
            )
            self.current_episode_step += 1
            return obs, reward, terminated, truncated, info
        except Exception as e:
            print("Step function encountered an error:", str(e))
            obs_raw, info = self.reset()
            obs = OrderedDict(
                [
                    ("observation", obs_raw),
                    ("achieved_goal", self.get_grasping_point_pos()),
                    ("desired_goal", self.get_block_pos()),
                ]
            )
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

    def compute_reward(
            self,
            achieved_goal: np.ndarray,
            desired_goal: np.ndarray,
            info
    ) -> float:
        is_batch = len(achieved_goal.shape) > 1

        if not is_batch:
            distance_penalty = -np.linalg.norm(achieved_goal - desired_goal)

            if self.use_viewpoint_reward:
                viewpoint_selection_reward = info["viewpoint_selection_reward"]
                reward = (self.distance_penalty_coefficient * distance_penalty
                            + self.viewpoint_selection_reward_coefficient * viewpoint_selection_reward)
            else:
                reward = distance_penalty
        else:
            distance_penalty = -np.linalg.norm(achieved_goal - desired_goal, axis=1)
            if self.use_viewpoint_reward and "viewpoint_selection_reward" in info[0]:
                viewpoint_selection_reward = np.array([i["viewpoint_selection_reward"] for i in info])
                reward = (self.distance_penalty_coefficient * distance_penalty
                            + self.viewpoint_selection_reward_coefficient * viewpoint_selection_reward)
            else:
                reward = distance_penalty


        return reward
    

    def _get_info(self):
        if self.use_viewpoint_reward:
            region_of_interest_img = self.calculate_region_of_interest_in_segmentation_image()
            viewpoint_selection_reward = np.count_nonzero(region_of_interest_img) / region_of_interest_img.size

            info = {
                "is_success": self.is_success(),
                "viewpoint_selection_reward": viewpoint_selection_reward,
            }
        else:
            info = {
                "is_success": self.is_success(),
            }
        return info


    def _get_reward(self):
        """
         Note that the following should always hold true:
                ob, reward, done, info = env.step()
                assert reward == env.compute_reward(ob['achieved_goal'], ob['desired_goal'], info)
        """


        if self.current_state == State.MOVE_TO_BLOCK:
            distance_penalty = -np.linalg.norm(self.get_grasping_point_pos() - self.get_block_pos())
        elif self.current_state == State.MOVE_TO_TARGET:
            distance_penalty = -np.linalg.norm(self.get_block_pos() - self.get_target_pos())
        else:
            raise ValueError("Invalid state")



        if self.use_viewpoint_reward:
            region_of_interest_img = self.calculate_region_of_interest_in_segmentation_image()
            viewpoint_selection_reward = np.count_nonzero(region_of_interest_img) / region_of_interest_img.size
            reward = (self.distance_penalty_coefficient * distance_penalty
                      + self.viewpoint_selection_reward_coefficient * viewpoint_selection_reward)
        else:
            reward = distance_penalty

        return reward

    def _get_terminated(self):
        res = bool(self.is_success())
        if res is True:
            self.reset()
        return res

    def _get_truncated(self):
        if self.max_episode_steps is None:
            return False
        res = self.current_episode_step >= self.max_episode_steps - 1
        if res is True:
            self.reset()
        return res

    def get_cheating_observation(self):
        # This does not need offset on linear motor since it is already in the world frame
        # Grasping point position (3 dim)
        grasping_point_pos = self.get_grasping_point_pos()

        # Block position (3 dim)
        block_pos = self.get_block_pos()

        # Target position (3 dim)
        target_pos = self.get_target_pos()

        # Relative position between grasping point and block (3 dim)
        relative_pos_grasping_point_block = grasping_point_pos - block_pos

        # Relative position between block and target (3 dim)
        relative_pos_block_target = block_pos - target_pos

        cheating_observation = np.concatenate([grasping_point_pos, block_pos, target_pos, relative_pos_grasping_point_block, relative_pos_block_target])
        return cheating_observation

    def set_cheating_obs_bounds(self):
        self.cheating_obs_low = np.array([-0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0])

        self.cheating_obs_high = np.array([0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037])

    def normalize_cheating_observation(self, cheating_observation):
        return 2 * (cheating_observation - self.cheating_obs_low) / (self.cheating_obs_high - self.cheating_obs_low) - 1

    def denormalize_cheating_observation(self, cheating_observation):
        return (cheating_observation + 1) * (self.cheating_obs_high - self.cheating_obs_low) / 2 + self.cheating_obs_low

    def sample_block_and_target_pos(self):
        if self.random_block_target_pos:
            block_pos = np.array([
                np.random.uniform(self.block_x_range[0], self.block_x_range[1]),
                np.random.uniform(self.block_y_range[0], self.block_y_range[1]),
                self.block_z,
            ])
            if self.target_in_the_air:
                target_pos = np.array([
                    np.random.uniform(self.target_x_range[0], self.target_x_range[1]),
                    np.random.uniform(self.target_y_range[0], self.target_y_range[1]),
                    np.random.uniform(self.target_z_range[0], self.target_z_range[1]),
                ])
            else:
                target_pos = np.array([
                    np.random.uniform(self.target_x_range[0], self.target_x_range[1]),
                    np.random.uniform(self.target_y_range[0], self.target_y_range[1]),
                    self.target_z_range[0],
                ])
        else:
            block_pos = np.array([0.45, -0.15, 0.025])
            if self.target_in_the_air:
                target_pos = np.array([0.45, 0.15, 0.5])
            else:
                target_pos = np.array([0.45, 0.15, 0.02])
        return block_pos, target_pos

    def set_block_and_target_pos(self):
        block_pos, target_pos = self.sample_block_and_target_pos()

        # print("block_pos: ", block_pos)
        # print("target_pos: ", target_pos)

        self.model.body_pos[self.target_body_id] = target_pos

        self.data.qpos[self.block_joint_id] = block_pos[0]
        self.data.qpos[self.block_joint_id + 1] = block_pos[1]
        self.data.qpos[self.block_joint_id + 2] = block_pos[2]

    def get_grasping_point_pos(self):
        return self.data.site_xpos[self.grasping_point_site_id]

    def get_block_pos(self):
        return self.data.site_xpos[self.block_site_id]

    def get_target_pos(self):
        return self.data.site_xpos[self.target_site_id]

    def compute_distance_between_grasping_point_and_block(self):
        grasping_point_pos = self.get_grasping_point_pos()
        block_pos = self.get_block_pos()
        return np.linalg.norm(grasping_point_pos - block_pos)

    def compute_distance_between_block_and_target(self):
        block_pos = self.get_block_pos()
        target_pos = self.get_target_pos()
        return np.linalg.norm(block_pos - target_pos)

    def get_segmentation_image(self):
        self.segmentation_viewer.make_context_current()
        seg_img = self.segmentation_viewer.render(render_mode='rgb_array',
                                                  camera_id=self.realsense_camera_id,
                                                  segmentation=True)
        seg_img_obj_type = seg_img[:, :, 0]
        seg_img_obj_id = seg_img[:, :, 1]

        return seg_img_obj_type, seg_img_obj_id

    def calculate_region_of_interest_in_segmentation_image(self):
        seg_img_obj_type, seg_img_obj_id = self.get_segmentation_image()
        region_of_interest = np.zeros_like(seg_img_obj_id)
        for geom_id in self.geom_ids_of_interest:
            region_of_interest[seg_img_obj_id == geom_id] = geom_id
        return region_of_interest

    def get_geom_ids_of_interest(self, model_names: MujocoModelNames):
        geom_names2ids = model_names.geom_name2id
        geom_names_of_interest = ["gripper_base_link",
                                  "left_outer_knuckle",
                                  "left_finger",
                                  "left_inner_knuckle",
                                  "right_outer_knuckle",
                                  "right_finger",
                                  "right_inner_knuckle",
                                  "object0"]

        geom_ids_of_interest = [geom_names2ids[geom_name] for geom_name in geom_names_of_interest]
        return geom_ids_of_interest

    def if_reached_block(self):
        distance_between_grasping_point_and_block = self.compute_distance_between_grasping_point_and_block()
        return distance_between_grasping_point_and_block < self.distance_threshold

    def is_success(self):
        distance_between_block_and_target = self.compute_distance_between_block_and_target()
        return distance_between_block_and_target < self.distance_threshold


    def _reset_simulation(self):
        super()._reset_simulation()
        self.set_block_and_target_pos()
        self.current_episode_step = 0
        self.current_state = State.MOVE_TO_BLOCK

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              ):
        obs_raw, info = super().reset(seed=seed, options=options)

        obs = OrderedDict(
            [
                ("observation", obs_raw),
                ("achieved_goal", self.get_grasping_point_pos()),
                ("desired_goal", self.get_block_pos()),
            ]
        )
        return obs, info
