from os import path
import os
import copy
import mujoco
import cv2
import zarr
import pygame
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.mujoco_env import MujocoEnv
from scipy.spatial.transform import Rotation as R
import mavis_mujoco_gym.utils.mavis_utils as mavis_utils

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 90.0,
    "elevation": -50.0,
    "lookat": np.array([0.45, 0.0, 0.0]),
}



class MAVISBaseEnv(MujocoEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,   # 33
    }

    def __init__(
            self,
            model_path="../MJCF/mavis_base_scene.xml",
            frame_skip=20,
            robot_noise_ratio: float = 0.01,
            obs_space_type: str = "config_space",    # Choose among "config_space", "eef_euler_space", "eef_quat_space", "eef_axis_angle_space", "lookat_euler_space", "lookat_quat_space", "lookat_axis_angle_space"
            act_space_type: str = "config_space",    # Choose among "config_space", "eef_euler_space", "eef_quat_space", "eef_axis_angle_space", "lookat_euler_space", "lookat_quat_space", "lookat_axis_angle_space"
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
            cheating_observation_dim: int = 0,
            is_goal_env=False,
            achieved_goal_shape=None,
            desired_goal_shape=None,
            use_teleoperation=False,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            model_path,
        )

        self.is_goal_env = is_goal_env
        if is_goal_env:
            if achieved_goal_shape is None:
                raise ValueError("achieved_goal_shape must be provided for goal-based environments.")

            if desired_goal_shape is None:
                raise ValueError("desired_goal_shape must be provided for goal-based environments.")

            self.achieved_goal_shape = achieved_goal_shape
            self.desired_goal_shape = desired_goal_shape

            achieved_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=achieved_goal_shape, dtype=np.float64)
            desired_goal_space = spaces.Box(low=-np.inf, high=np.inf, shape=desired_goal_shape, dtype=np.float64)

        self.metadata["render_fps"] = render_fps

        self.robot_noise_ratio = robot_noise_ratio  # TODO: implement robot noise
        self.obs_space_type = obs_space_type
        self.act_space_type = act_space_type

        self.action_normalization = action_normalization
        self.observation_normalization = observation_normalization

        self.mavis_eef_kinematics = None
        self.mavis_lookat_kinematics = None

        self.use_cheating_observation = use_cheating_observation
        self.cheating_observation_dim = cheating_observation_dim

        if self.obs_space_type == "config_space":
            # [0]: manipulation arm linear track state
            # [1:8]: manipulation arm joint state
            # [8]: viewpoint arm linear track state
            # [9:16]: viewpoint arm joint state
            # [16]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        elif self.obs_space_type == "eef_euler_space":
            # [0:3]: manipulation arm end-effector position
            # [3:6]: manipulation arm end-effector orientation in Euler angles
            # [6:9]: viewpoint arm end-effector position
            # [9:12]: viewpoint arm end-effector orientation in Euler angles
            # [12]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float64)
        elif self.obs_space_type == "eef_quat_space":
            # [0:3]: manipulation arm end-effector position
            # [3:7]: manipulation arm end-effector orientation in quaternion
            # [7:10]: viewpoint arm end-effector position
            # [10:14]: viewpoint arm end-effector orientation in quaternion
            # [14]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        elif self.obs_space_type == "eef_axis_angle_space":
            # [0:3]: manipulation arm end-effector position
            # [3:7]: manipulation arm end-effector orientation in axis-angle representation. [axis, angle]
            # [7:10]: viewpoint arm end-effector position
            # [10:14]: viewpoint arm end-effector orientation in axis-angle representation. [axis, angle]
            # [14]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)
        elif self.obs_space_type == "lookat_euler_space":
            # [0:3]: manipulation arm end-effector position
            # [3:6]: manipulation arm end-effector orientation in Euler angles
            # [6:9]: viewpoint arm end-effector position
            # [9]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float64)
        elif self.obs_space_type == "lookat_quat_space":
            # [0:3]: manipulation arm end-effector position
            # [3:7]: manipulation arm end-effector orientation in quaternion
            # [7:10]: viewpoint arm end-effector position
            # [10]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        elif self.obs_space_type == "lookat_axis_angle_space":
            # [0:3]: manipulation arm end-effector position
            # [3:7]: manipulation arm end-effector orientation in axis-angle representation. [axis, angle]
            # [7:10]: viewpoint arm end-effector position
            # [10]: gripper state
            if use_cheating_observation:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11 + cheating_observation_dim,), dtype=np.float64)
            else:
                state_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(11,), dtype=np.float64)
        else:
            raise ValueError("Invalid observation space type.")

        self.img_width = img_width
        self.img_height = img_height

        if enable_rgb:
            rgb_obs_space = spaces.Box(low=0, high=255, shape=(self.img_height, self.img_width, 3), dtype=np.uint8)

        if enable_depth:
            depth_obs_space = spaces.Box(low=0, high=1, shape=(self.img_height, self.img_width, 1), dtype=np.float64)

        if enable_rgb and enable_depth:
            if not self.is_goal_env:
                observation_space = spaces.Dict({
                    "state": state_obs_space,
                    "rgb": rgb_obs_space,
                    "depth": depth_obs_space,
                })
            else:
                observation_space = spaces.Dict(
                    dict(
                        desired_goal=desired_goal_space,
                        achieved_goal=achieved_goal_space,
                        observation=spaces.Dict({
                            "state": state_obs_space,
                            "rgb": rgb_obs_space,
                            "depth": depth_obs_space}),
                        )
                )
        elif enable_rgb:
            if not self.is_goal_env:
                observation_space = spaces.Dict({
                    "state": state_obs_space,
                    "rgb": rgb_obs_space,
                })
            else:
                observation_space = spaces.Dict(
                    dict(
                        desired_goal=desired_goal_space,
                        achieved_goal=achieved_goal_space,
                        observation=spaces.Dict({
                            "state": state_obs_space,
                            "rgb": rgb_obs_space}),
                        )
                )
        elif enable_depth:
            if not self.is_goal_env:
                observation_space = spaces.Dict({
                    "state": state_obs_space,
                    "depth": depth_obs_space,
                })
            else:
                observation_space = spaces.Dict(
                    dict(
                        desired_goal=desired_goal_space,
                        achieved_goal=achieved_goal_space,
                        observation=spaces.Dict({
                            "state": state_obs_space,
                            "depth": depth_obs_space}),
                        )
                )
        else:
            if not self.is_goal_env:
                observation_space = state_obs_space
            else:
                observation_space = spaces.Dict(
                    dict(
                        desired_goal=desired_goal_space,
                        achieved_goal=achieved_goal_space,
                        observation=state_obs_space,
                        )
                )

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth

        self.camera_name = camera_name

        super().__init__(
            model_path=xml_file_path,
            frame_skip=frame_skip,
            observation_space=observation_space,
            render_mode=render_mode,
            width=img_width,
            height=img_height,
            camera_name=camera_name,
            default_camera_config=default_camera_config,
        )

        self.state_obs_low, self.state_obs_high = self._get_state_obs_bounds(obs_space=self.obs_space_type)

        if self.obs_space_type == "eef_euler_space" or self.obs_space_type == "eef_quat_space" or self.obs_space_type == "eef_axis_angle_space":
            lower_bounds, upper_bounds = self._get_ik_optimization_bound()
            self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

        if self.obs_space_type == "lookat_quat_space" or self.obs_space_type == "lookat_euler_space" or self.obs_space_type == "lookat_axis_angle_space":
            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)

        self._set_action_space()

        # Set to initial configuration
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        self.init_qpos = copy.deepcopy(self.data.qpos)
        self.init_qvel = copy.deepcopy(self.data.qvel)

        # self.model.opt.timestep = 0.002. This is dt in the mujoco simulation
        # Therefore, for example, if we set frame_skip=40, the simulation will run at 40 * 0.002 = 0.08 seconds per step in the gym environment
        # When setting frame_skip=40, the simulation will run at 12.5 Hz
        # Similarly, when setting frame_skip=20, the simulation will run at 25 Hz
        # https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/fetch/pick_and_place.py
        # print("model.opt.timestep: ", self.model.opt.timestep)



        self.use_teleoperation = use_teleoperation

        if self.use_teleoperation:
            self.prev_configuration = None

            lower_bounds, upper_bounds = self._get_ik_optimization_bound()

            if self.mavis_eef_kinematics is None:
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds,
                                                                                   upper_bounds)

            if self.mavis_lookat_kinematics is None:
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds,
                                                                                 upper_bounds)

    def step(self, action):
        is_batch = len(action.shape) > 1

        if is_batch:
            # e.g., if action has shape (1, 10), we need to reshape it to (10,)
            action = action.squeeze(axis=0)

        if self.action_normalization:
            action = np.clip(action, -1.0, 1.0)
            action = self.denormalize_action(action)

        ctrl, configuration = self.calculate_mujoco_config_ctrl_from_action(action, action_space=self.act_space_type)

        self.do_simulation(ctrl, self.frame_skip)

        if self.render_mode == "human" or self.render_mode == "rgb_array" or self.render_mode == "depth_array":
            self.render()

        obs = self._get_obs()
        reward = self._get_reward()
        terminated = self._get_terminated()
        truncated = self._get_truncated()
        info = self._get_info()

        info["dual_arm_action_configuration"] = configuration

        if self.render_mode=="human":
            self.render()

        return obs, reward, terminated, truncated, info

    def reset_model(self):
        obs = self._get_obs()
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        return obs

    def calculate_mujoco_config_ctrl_from_action(self, action, action_space=None):
        gripper_action = action[-1]
        gripper_ctrl = self._convert_gripper_state_to_mujoco_ctrl(gripper_action)

        raw_dual_arm_configuration = None
        if action_space == "config_space":
            raw_dual_arm_configuration = copy.deepcopy(action[:-1])
            manipulation_arm_linear_motor_ctrl = action[0] - 0.37
            viewpoint_arm_linear_motor_ctrl = action[8] - 0.37
            action[0] = manipulation_arm_linear_motor_ctrl
            action[8] = viewpoint_arm_linear_motor_ctrl
            ctrl = np.concatenate([action[0:8], [gripper_ctrl], action[8:16]])

        elif action_space == "eef_euler_space":
            if self.prev_configuration is None:
                self.prev_configuration = self._reformat_init_qpos_to_initial_configuration()

            target_gripper_pos = action[0:3]
            target_gripper_rot = action[3:6]
            target_camera_pos = action[6:9]
            target_camera_rot = action[9:12]

            if self.mavis_eef_kinematics:
                configuration = self.mavis_eef_kinematics.inverse_kinematics(self.prev_configuration,
                                                                             target_gripper_pos,
                                                                             target_gripper_rot,
                                                                             target_camera_pos,
                                                                             target_camera_rot,
                                                                             rot_repr="euler")
                self.prev_configuration = copy.deepcopy(configuration)
                raw_dual_arm_configuration = copy.deepcopy(configuration)

                manipulation_arm_linear_motor_ctrl = configuration[0] - 0.37
                viewpoint_arm_linear_motor_ctrl = configuration[8] - 0.37
                configuration[0] = manipulation_arm_linear_motor_ctrl
                configuration[8] = viewpoint_arm_linear_motor_ctrl
                ctrl = np.concatenate([configuration[0:8], [gripper_ctrl], configuration[8:16]])

            else:
                raise ValueError("End-effector kinematics is not initialized.")

        elif action_space == "eef_quat_space":
            if self.prev_configuration is None:
                self.prev_configuration = self._reformat_init_qpos_to_initial_configuration()

            target_gripper_pos = action[0:3]
            target_gripper_rot = action[3:7]
            target_camera_pos = action[7:10]
            target_camera_rot = action[10:14]

            if self.mavis_eef_kinematics:
                configuration = self.mavis_eef_kinematics.inverse_kinematics(self.prev_configuration,
                                                                             target_gripper_pos,
                                                                             target_gripper_rot,
                                                                             target_camera_pos,
                                                                             target_camera_rot,
                                                                             rot_repr="quat")
                self.prev_configuration = copy.deepcopy(configuration)
                raw_dual_arm_configuration = copy.deepcopy(configuration)

                manipulation_arm_linear_motor_ctrl = configuration[0] - 0.37
                viewpoint_arm_linear_motor_ctrl = configuration[8] - 0.37
                configuration[0] = manipulation_arm_linear_motor_ctrl
                configuration[8] = viewpoint_arm_linear_motor_ctrl

                ctrl = np.concatenate([configuration[0:8], [gripper_ctrl], configuration[8:16]])

            else:
                raise ValueError("End-effector kinematics is not initialized.")

        elif action_space == "eef_axis_angle_space":
            if self.prev_configuration is None:
                self.prev_configuration = self._reformat_init_qpos_to_initial_configuration()

            target_gripper_pos = action[0:3]
            target_gripper_rot = action[3:7]
            target_camera_pos = action[7:10]
            target_camera_rot = action[10:14]

            if self.mavis_eef_kinematics:
                configuration = self.mavis_eef_kinematics.inverse_kinematics(self.prev_configuration,
                                                                             target_gripper_pos,
                                                                             target_gripper_rot,
                                                                             target_camera_pos,
                                                                             target_camera_rot,
                                                                             rot_repr="axis-angle")
                self.prev_configuration = copy.deepcopy(configuration)
                raw_dual_arm_configuration = copy.deepcopy(configuration)

                manipulation_arm_linear_motor_ctrl = configuration[0] - 0.37
                viewpoint_arm_linear_motor_ctrl = configuration[8] - 0.37
                configuration[0] = manipulation_arm_linear_motor_ctrl
                configuration[8] = viewpoint_arm_linear_motor_ctrl

                ctrl = np.concatenate([configuration[0:8], [gripper_ctrl], configuration[8:16]])
            else:
                raise ValueError("End-effector kinematics is not initialized.")

        elif action_space == "lookat_euler_space":
            if self.prev_configuration is None:
                self.prev_configuration = self._reformat_init_qpos_to_initial_configuration()

            target_gripper_pos = action[0:3]
            target_gripper_rot = action[3:6]
            target_camera_pos = action[6:9]

            if self.mavis_lookat_kinematics:
                configuration = self.mavis_lookat_kinematics.inverse_kinematics(self.prev_configuration,
                                                                                 target_gripper_pos,
                                                                                 target_gripper_rot,
                                                                                 target_camera_pos,
                                                                                 rot_repr="euler")
                self.prev_configuration = copy.deepcopy(configuration)
                raw_dual_arm_configuration = copy.deepcopy(configuration)

                manipulation_arm_linear_motor_ctrl = configuration[0] - 0.37
                viewpoint_arm_linear_motor_ctrl = configuration[8] - 0.37
                configuration[0] = manipulation_arm_linear_motor_ctrl
                configuration[8] = viewpoint_arm_linear_motor_ctrl

                ctrl = np.concatenate([configuration[0:8], [gripper_ctrl], configuration[8:16]])
            else:
                raise ValueError("lookat kinematics is not initialized.")

        elif action_space == "lookat_quat_space":
            if self.prev_configuration is None:
                self.prev_configuration = self._reformat_init_qpos_to_initial_configuration()

            target_gripper_pos = action[0:3]
            target_gripper_rot = action[3:7]
            target_camera_pos = action[7:10]

            if self.mavis_lookat_kinematics:
                configuration = self.mavis_lookat_kinematics.inverse_kinematics(self.prev_configuration,
                                                                                 target_gripper_pos,
                                                                                 target_gripper_rot,
                                                                                 target_camera_pos,
                                                                                 rot_repr="quat")
                self.prev_configuration = copy.deepcopy(configuration)
                raw_dual_arm_configuration = copy.deepcopy(configuration)

                manipulation_arm_linear_motor_ctrl = configuration[0] - 0.37
                viewpoint_arm_linear_motor_ctrl = configuration[8] - 0.37
                configuration[0] = manipulation_arm_linear_motor_ctrl
                configuration[8] = viewpoint_arm_linear_motor_ctrl

                ctrl = np.concatenate([configuration[0:8], [gripper_ctrl], configuration[8:16]])
            else:
                raise ValueError("lookat kinematics is not initialized.")
        elif action_space == "lookat_axis_angle_space":
            if self.prev_configuration is None:
                self.prev_configuration = self._reformat_init_qpos_to_initial_configuration()

            target_gripper_pos = action[0:3]
            target_gripper_rot = action[3:7]
            target_camera_pos = action[7:10]

            if self.mavis_lookat_kinematics:
                configuration = self.mavis_lookat_kinematics.inverse_kinematics(self.prev_configuration,
                                                                                 target_gripper_pos,
                                                                                 target_gripper_rot,
                                                                                 target_camera_pos,
                                                                                 rot_repr="axis-angle")
                self.prev_configuration = copy.deepcopy(configuration)
                raw_dual_arm_configuration = copy.deepcopy(configuration)

                manipulation_arm_linear_motor_ctrl = configuration[0] - 0.37
                viewpoint_arm_linear_motor_ctrl = configuration[8] - 0.37
                configuration[0] = manipulation_arm_linear_motor_ctrl
                configuration[8] = viewpoint_arm_linear_motor_ctrl

                ctrl = np.concatenate([configuration[0:8], [gripper_ctrl], configuration[8:16]])
            else:
                raise ValueError("lookat kinematics is not initialized.")
        else:
            raise ValueError("Invalid action space type.  .")
        
        return ctrl, raw_dual_arm_configuration


    def _get_obs(self):
        rgb_image = self.mujoco_renderer.render(render_mode="rgb_array")
        if self.enable_depth:
            depth_image = self.mujoco_renderer.render(render_mode="depth_array")
        state_obs = self._get_state_obs(obs_space=self.obs_space_type)

        if self.observation_normalization:
            state_obs = self.normalize_state_obs(state_obs)
            #if self.enable_rgb:
            #    rgb_image = self.normalize_rgb_image(rgb_image)
            # Depth image is already normalized
            # Each pixel in the depth image represents the normalized distance from the camera to the object at that pixel location
            # A value of 0 represents the nearest point within the camera’s field of view.
            # A value of 1 represents the farthest point within the camera’s field of view.
            # These depth values are relative to the near and far clipping planes defined in your camera setup in the mujoco environment.

        # if depth image does not have the channel dimension, add it
        if self.enable_depth:
            if len(depth_image.shape) == 2:
                depth_image = np.expand_dims(depth_image, axis=-1)

        if self.enable_rgb and self.enable_depth:
            obs = {
                "state": state_obs,
                "rgb": rgb_image,
                "depth": depth_image,
            }
        elif self.enable_rgb:
            obs = {
                "state": state_obs,
                "rgb": rgb_image,
            }
        elif self.enable_depth:
            obs = {
                "state": state_obs,
                "depth": depth_image,
            }
        else:
            obs = state_obs

        return obs

    def _get_reward(self):
        # TODO: this should be scenario-specific
        return 0.0

    def _get_terminated(self):
        # TODO: this should be scenario-specific
        return False

    def _get_truncated(self):
        # TODO: this should be scenario-specific
        return False

    def _get_info(self):
        # TODO: this should be scenario-specific
        return {}

    def _get_state_obs(self, obs_space):
        manipulation_arm_linear_track_state_obs = self.data.qpos[0] + 0.37
        manipulation_arm_joints_state_obs = self.data.qpos[1:8]

        viewpoint_arm_linear_track_state_obs = self.data.qpos[14] + 0.37
        viewpoint_arm_joints_state_obs = self.data.qpos[15:22]

        manipulation_arm_linear_track_state_obs = np.array([manipulation_arm_linear_track_state_obs])
        viewpoint_arm_linear_track_state_obs = np.array([viewpoint_arm_linear_track_state_obs])
        gripper_state_obs = self._get_gripper_state_obs()

        if obs_space == "config_space":
            state_obs = np.concatenate([
                manipulation_arm_linear_track_state_obs,
                manipulation_arm_joints_state_obs,
                viewpoint_arm_linear_track_state_obs,
                viewpoint_arm_joints_state_obs,
                gripper_state_obs,
            ])
        elif obs_space == "eef_euler_space":
            # TODO: manipulation and viewpoint arm end-effector position and orientation in Euler angles
            if self.mavis_eef_kinematics:
                current_configuration = np.concatenate([manipulation_arm_linear_track_state_obs,
                                                                    manipulation_arm_joints_state_obs,
                                                                    viewpoint_arm_linear_track_state_obs,
                                                                    viewpoint_arm_joints_state_obs])
                manipulation_arm_eef_pos, manipulation_arm_eef_rot,viewpoint_arm_eef_pos, viewpoint_arm_eef_rot = self.mavis_eef_kinematics.bimanual_forward_kinematics(current_configuration)

                manipulation_arm_eef_rot = np.array([manipulation_arm_eef_rot[1], manipulation_arm_eef_rot[2], manipulation_arm_eef_rot[3], manipulation_arm_eef_rot[0]])
                manipulation_arm_eef_rot = R.from_quat(manipulation_arm_eef_rot).as_euler("xyz", degrees=False)

                viewpoint_arm_eef_rot = np.array([viewpoint_arm_eef_rot[1], viewpoint_arm_eef_rot[2], viewpoint_arm_eef_rot[3], viewpoint_arm_eef_rot[0]])
                viewpoint_arm_eef_rot = R.from_quat(viewpoint_arm_eef_rot).as_euler("xyz", degrees=False)

                state_obs = np.concatenate([
                    manipulation_arm_eef_pos,
                    manipulation_arm_eef_rot,
                    viewpoint_arm_eef_pos,
                    viewpoint_arm_eef_rot,
                    gripper_state_obs,
                ])
            else:
                raise ValueError("End-effector kinematics is not initialized.")
        elif obs_space == "eef_quat_space":
            if self.mavis_eef_kinematics:
                current_configuration = np.concatenate([manipulation_arm_linear_track_state_obs,
                                                                    manipulation_arm_joints_state_obs,
                                                                    viewpoint_arm_linear_track_state_obs,
                                                                    viewpoint_arm_joints_state_obs])
                manipulation_arm_eef_pos, manipulation_arm_eef_rot,viewpoint_arm_eef_pos, viewpoint_arm_eef_rot = self.mavis_eef_kinematics.bimanual_forward_kinematics(current_configuration)

                state_obs = np.concatenate([
                    manipulation_arm_eef_pos,
                    manipulation_arm_eef_rot,
                    viewpoint_arm_eef_pos,
                    viewpoint_arm_eef_rot,
                    gripper_state_obs,
                ])
            else:
                raise ValueError("End-effector kinematics is not initialized.")
        elif obs_space == "eef_axis_angle_space":
            if self.mavis_eef_kinematics:
                current_configuration = np.concatenate([manipulation_arm_linear_track_state_obs,
                                                        manipulation_arm_joints_state_obs,
                                                        viewpoint_arm_linear_track_state_obs,
                                                        viewpoint_arm_joints_state_obs])
                manipulation_arm_eef_pos, manipulation_arm_eef_rot, viewpoint_arm_eef_pos, viewpoint_arm_eef_rot = self.mavis_eef_kinematics.bimanual_forward_kinematics(
                    current_configuration)

                manipulation_arm_eef_rot_axis, manipulation_arm_eef_rot_angle = mavis_utils.quaternion_to_axis_angle(manipulation_arm_eef_rot)
                viewpoint_arm_eef_rot_axis, viewpoint_arm_eef_rot_angle = mavis_utils.quaternion_to_axis_angle(viewpoint_arm_eef_rot)

                manipulation_arm_eef_rot = np.concatenate([manipulation_arm_eef_rot_axis, [manipulation_arm_eef_rot_angle]])
                viewpoint_arm_eef_rot = np.concatenate([viewpoint_arm_eef_rot_axis, [viewpoint_arm_eef_rot_angle]])

                state_obs = np.concatenate([
                    manipulation_arm_eef_pos,
                    manipulation_arm_eef_rot,
                    viewpoint_arm_eef_pos,
                    viewpoint_arm_eef_rot,
                    gripper_state_obs,
                ])
            else:
                raise ValueError("End-effector kinematics is not initialized.")
        elif obs_space == "lookat_euler_space":
            configurations = np.concatenate([manipulation_arm_linear_track_state_obs, manipulation_arm_joints_state_obs, viewpoint_arm_linear_track_state_obs, viewpoint_arm_joints_state_obs])
            lookat_state_obs = self.mavis_lookat_kinematics.get_lookat_forward_kinematics(configurations, is_quat=False)

            state_obs = np.concatenate([
                lookat_state_obs,
                gripper_state_obs,
            ])
        elif obs_space == "lookat_quat_space":
            configurations = np.concatenate([manipulation_arm_linear_track_state_obs, manipulation_arm_joints_state_obs, viewpoint_arm_linear_track_state_obs, viewpoint_arm_joints_state_obs])
            lookat_state_obs = self.mavis_lookat_kinematics.get_lookat_forward_kinematics(configurations)

            state_obs = np.concatenate([
                lookat_state_obs,
                gripper_state_obs,
            ])
        elif obs_space == "lookat_axis_angle_space":
            configurations = np.concatenate([manipulation_arm_linear_track_state_obs, manipulation_arm_joints_state_obs,
                                             viewpoint_arm_linear_track_state_obs, viewpoint_arm_joints_state_obs])
            lookat_state_obs = self.mavis_lookat_kinematics.get_lookat_forward_kinematics(configurations)

            lookat_state_obs_quat = lookat_state_obs[3:7]
            lookat_state_obs_axis, lookat_state_obs_angle = mavis_utils.quaternion_to_axis_angle(lookat_state_obs_quat)
            lookat_state_obs = np.concatenate([lookat_state_obs[0:3], lookat_state_obs_axis, [lookat_state_obs_angle], lookat_state_obs[7:10]])

            state_obs = np.concatenate([
                lookat_state_obs,
                gripper_state_obs,
            ])
        else:
            raise ValueError("Invalid observation space type.")

        return state_obs

    def _get_gripper_state_obs(self):
        gripper_components_state_obs = self.data.qpos[8:14]
        gripper_state_obs_raw = round(np.mean(gripper_components_state_obs) / 10, 3)
        gripper_state_obs_raw = max(0.0, min(0.085, gripper_state_obs_raw))
        gripper_state_obs = 0.085 - gripper_state_obs_raw
        return np.array([gripper_state_obs])

    def _set_action_space(self):
        if self.act_space_type == "config_space":
            bounds_raw = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low_raw, high_raw = bounds_raw.T

            low = copy.deepcopy(low_raw)
            low[0] = 0
            low[8] = 0
            low[9:16] = low_raw[10:17]
            low[16] = 0

            high = copy.deepcopy(high_raw)
            high[0] = 0.74
            high[8] = 0.74
            high[9:16] = high_raw[10:17]
            high[16] = 0.085

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(17,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(17,), dtype=np.float32)

        elif self.act_space_type == "eef_euler_space":
            # Based on the reachability of xArm7: https://www.robotshop.com/products/xarm-7-dof-robotic-arm
            # Y axis is the Y reachability offset by half of the linear track length
            low = np.array([-0.77, -0.77, 0.0, -np.pi, -np.pi, -np.pi, -0.77, -0.77, 0.0, -np.pi, -np.pi, -np.pi, 0.0])
            high = np.array([0.77, 1.51, 1.037, np.pi, np.pi, np.pi, 0.77, 1.51, 1.037, np.pi, np.pi, np.pi, 0.085])

            if self.mavis_eef_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

            self.prev_configuration = None

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(13,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(13,), dtype=np.float32)

        elif self.act_space_type == "eef_quat_space":
            low = np.array([-0.77, -0.77, 0.0, -1, -1, -1, -1, -0.77, -0.77, 0.0, -1, -1, -1, -1, 0.0])
            high = np.array([0.77, 1.51, 1.037, 1, 1, 1, 1, 0.77, 1.51, 1.037, 1, 1, 1, 1, 0.085])

            if self.mavis_eef_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

            self.prev_configuration = None

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(15,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)

        elif self.act_space_type == "eef_axis_angle_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            -0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                             1.0, 1.0, 1.0, np.pi,
                             0.77, 1.51, 1.037,
                             1.0, 1.0, 1.0, np.pi,
                             0.085])

            if self.mavis_eef_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

            self.prev_configuration = None

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(15,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(15,), dtype=np.float32)

        elif self.act_space_type == "lookat_euler_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -np.pi, -np.pi, -np.pi,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                             np.pi, np.pi, np.pi,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)

            self.prev_configuration = None

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(10,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)

        elif self.act_space_type == "lookat_quat_space":
            # [0:3]: manipulation arm end-effector position
            # [3:7]: manipulation arm end-effector orientation in quaternion
            # [7:10]: looker link position delta
            # [10]: looker link in-out delta
            # [11]: gripper state
            low = np.array([-0.77, -0.77, 0.0, -1, -1, -1, -1,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037, 1, 1, 1, 1,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)

            self.prev_configuration = None

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(11,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)

        elif self.act_space_type == "lookat_axis_angle_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                             1.0, 1.0, 1.0, np.pi,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)

            self.prev_configuration = None

            if not self.action_normalization:
                self.action_space = spaces.Box(low=low, high=high, shape=(11,), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)

        else:
            raise ValueError("Invalid action space type.  .")



        self.action_high = high
        self.action_low = low

        return self.action_space, self.action_high, self.action_low

    def _reformat_init_qpos_to_initial_configuration(self):
        initial_guess = np.zeros(16)
        initial_guess[0] = self.init_qpos[0] + 0.37
        initial_guess[1:8] = self.init_qpos[1:8]
        initial_guess[8] = self.init_qpos[14] + 0.37
        initial_guess[9:16] = self.init_qpos[15:22]
        return initial_guess

    def _convert_gripper_state_to_mujoco_ctrl(self, gripper_state):
        model_value = (0.085 - gripper_state) * (255 - 0.0) / (0.085 - 0.0)
        return model_value

    def convert_gripper_mujoco_ctrl_to_state(self, gripper_ctrl):
        state_value = 0.085 - (gripper_ctrl * (0.085 - 0.0) / (255 - 0.0))
        return state_value

    def _reset_simulation(self):
       super()._reset_simulation()
       mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

    def _get_ik_optimization_bound(self):
        bounds_raw = self.model.actuator_ctrlrange.copy().astype(np.float32)
        # print("bound_raw = ", bounds_raw)
        low_raw, high_raw = bounds_raw.T

        low = copy.deepcopy(low_raw)
        low[0] = 0
        low[8:16] = low_raw[9:17]

        low[8] = 0

        high = copy.deepcopy(high_raw)
        high[0] = 0.74
        high[8:16] = high_raw[9:17]

        high[8] = 0.74

        return low[0:16], high[0:16]

    def _get_action_bounds(self, action_space=None):
        if action_space == "config_space":
            bounds_raw = self.model.actuator_ctrlrange.copy().astype(np.float32)
            low_raw, high_raw = bounds_raw.T

            low = copy.deepcopy(low_raw)
            low[0] = 0
            low[8] = 0
            low[9:16] = low_raw[10:17]
            low[16] = 0

            high = copy.deepcopy(high_raw)
            high[0] = 0.74
            high[8] = 0.74
            high[9:16] = high_raw[10:17]
            high[16] = 0.085

            print("low: ", low)
            print("high: ", high)

        elif action_space == "eef_euler_space":
            low = np.array([-0.77, -0.77, 0.0, -np.pi, -np.pi, -np.pi, -0.77, -0.77, 0.0, -np.pi, -np.pi, -np.pi, 0.0])
            high = np.array([0.77, 1.51, 1.037, np.pi, np.pi, np.pi, 0.77, 1.51, 1.037, np.pi, np.pi, np.pi, 0.085])

        elif action_space == "eef_quat_space":
            low = np.array([-0.77, -0.77, 0.0, -1, -1, -1, -1, -0.77, -0.77, 0.0, -1, -1, -1, -1, 0.0])
            high = np.array([0.77, 1.51, 1.037, 1, 1, 1, 1, 0.77, 1.51, 1.037, 1, 1, 1, 1, 0.085])

        elif action_space == "eef_axis_angle_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            -0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                            1.0, 1.0, 1.0, np.pi,
                            0.77, 1.51, 1.037,
                            1.0, 1.0, 1.0, np.pi,
                            0.085])

        elif action_space == "lookat_euler_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -np.pi, -np.pi, -np.pi,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                             np.pi, np.pi, np.pi,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

        elif action_space == "lookat_quat_space":
            low = np.array([-0.77, -0.77, 0.0, -1, -1, -1, -1,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037, 1, 1, 1, 1,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

        elif action_space == "lookat_axis_angle_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                             1.0, 1.0, 1.0, np.pi,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

        else:
            raise ValueError("Invalid action space type.  .")

        return low, high

    def _get_state_obs_bounds(self, obs_space=None):
        if obs_space == "config_space":
            bounds_raw = self.model.jnt_range.copy().astype(np.float32)
            low_raw, high_raw = bounds_raw.T

            low = np.zeros(17)
            low[0] = 0
            low[1:8] = low_raw[1:8]
            low[8] = 0
            low[9:16] = low_raw[15:22]
            low[16] = 0

            high = np.zeros(17)
            high[0] = 0.74
            high[1:8] = high_raw[1:8]
            high[8] = 0.74
            high[9:16] = high_raw[15:22]
            high[16] = 0.085

        elif obs_space == "eef_euler_space":
            # https://www.robotshop.com/products/xarm-7-dof-robotic-arm
            low = np.array([-0.77, -0.77, 0.0, -np.pi, -np.pi, -np.pi, -0.77, -0.77, 0.0, -np.pi, -np.pi, -np.pi, 0.0])
            high = np.array([0.77, 1.51, 1.037, np.pi, np.pi, np.pi, 0.77, 1.51, 1.037, np.pi, np.pi, np.pi, 0.085])

            if self.mavis_eef_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

        elif obs_space == "eef_quat_space":
            low = np.array([-0.77, -0.77, 0.0, -1, -1, -1, -1, -0.77, -0.77, 0.0, -1, -1, -1, -1, 0.0])
            high = np.array([0.77, 1.51, 1.037, 1, 1, 1, 1, 0.77, 1.51, 1.037, 1, 1, 1, 1, 0.085])

            if self.mavis_eef_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

        elif obs_space == "eef_axis_angle_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            -0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                            1.0, 1.0, 1.0, np.pi,
                            0.77, 1.51, 1.037,
                            1.0, 1.0, 1.0, np.pi,
                            0.085])

            if self.mavis_eef_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_eef_kinematics = mavis_utils.MAVISEndEffectorKinematics(lower_bounds, upper_bounds)

        elif obs_space == "lookat_euler_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -np.pi, -np.pi, -np.pi,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037,
                             np.pi, np.pi, np.pi,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)

        elif obs_space == "lookat_quat_space":
            low = np.array([-0.77, -0.77, 0.0, -1, -1, -1, -1,
                            -0.77 + 0.9, -0.77, 0.0,
                            0.0])
            high = np.array([0.77, 1.51, 1.037, 1, 1, 1, 1,
                             0.77 + 0.9, 1.51, 1.037,
                             0.085])

            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)

        elif obs_space == "lookat_axis_angle_space":
            low = np.array([-0.77, -0.77, 0.0,
                            -1.0, -1.0, -1.0, 0.0,
                            -0.77, -0.77, 0.0,
                            0.0])

            high = np.array([0.77, 1.51, 1.037,
                            1.0, 1.0, 1.0, np.pi,
                            0.77, 1.51, 1.037,
                            0.085])

            if self.mavis_lookat_kinematics is None:
                lower_bounds, upper_bounds = self._get_ik_optimization_bound()
                self.mavis_lookat_kinematics = mavis_utils.MAVISLookAtKinematics(lower_bounds, upper_bounds)
        else:
            raise ValueError("Invalid observation space type.")

        return low, high


    def normalize_action(self, action):
        return 2 * (action - self.action_low) / (self.action_high - self.action_low) - 1



    def denormalize_action(self, action):
        return (action + 1) * (self.action_high - self.action_low) / 2 + self.action_low

    def normalize_state_obs(self, state_obs):
        return 2 * (state_obs - self.state_obs_low) / (self.state_obs_high - self.state_obs_low) - 1



    def denormalize_state_obs(self, state_obs):
        return (state_obs + 1) * (self.state_obs_high - self.state_obs_low) / 2 + self.state_obs_low

    def normalize_rgb_image(self, rgb_image):
        return rgb_image / 255.0

    def denormalize_rgb_image(self, rgb_image):
        return rgb_image * 255.0


