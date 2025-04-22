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

DEFAULT_CAMERA_CONFIG = {
    "distance": 2.0,
    "azimuth": 90.0,
    "elevation": -50.0,
    "lookat": np.array([0.45, 0.0, 0.0]),
}

class PickAndPlace(MAVISBaseEnv):
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
            render_fps: int = 50,
            img_width: int = 640,
            img_height: int = 480,
            camera_name: str = "realsense_camera",
            default_camera_config: dict = DEFAULT_CAMERA_CONFIG,
            enable_rgb: bool = True,
            enable_depth: bool = True,
            use_cheating_observation: bool = False,
            use_viewpoint_penalty: bool = True,
            rs_cam_distance_penalty_triggering_lower_bound: float = 0.15,
            rs_cam_distance_penalty_triggering_upper_bound: float = 1.0,
            block_z_no_penalty_lower_bound: float = 0.1,
            block_z_no_penalty_upper_bound: float = 0.3,
            distance_penalty_coefficient: float = 0.35,
            viewpoint_selection_penalty_coefficient: float = 0.15,
            goal_complete_reward: float = 500.0,
            gripper_finger_contact_reward: float = 0.035,
            block_z_penalty_coefficient: float = 1.0,
            target_in_the_air: bool = False,
            random_block_rot: bool = False,
            random_block_pos: bool = True,
            random_target_pos: bool = False,
            block_x_range: list = [0.4, 0.5],
            block_y_range: list = [-0.05, 0.05],
            block_z: float = 0.025,
            target_x_range: list = [0.3, 0.6],
            target_y_range: list = [-0.15, 0.15],
            target_z_range: list = [0.02, 0.2],
            distance_threshold: float = 0.05,
            max_episode_steps=None,
            use_teleoperation: bool = False,
    ):
        self.metadata["render_fps"] = render_fps
        self.distance_penalty_coefficient = distance_penalty_coefficient
        self.viewpoint_selection_penalty_coefficient = viewpoint_selection_penalty_coefficient
        self.gripper_finger_contact_reward = gripper_finger_contact_reward
        self.block_z_penalty_coefficient = block_z_penalty_coefficient
        self.goal_complete_reward = goal_complete_reward
        self.rs_cam_distance_penalty_triggering_lower_bound = rs_cam_distance_penalty_triggering_lower_bound
        self.rs_cam_distance_penalty_triggering_upper_bound = rs_cam_distance_penalty_triggering_upper_bound
        self.block_z_no_penalty_lower_bound = block_z_no_penalty_lower_bound
        self.block_z_no_penalty_upper_bound = block_z_no_penalty_upper_bound
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
            cheating_observation_dim=23,
            use_teleoperation=use_teleoperation,
        )
        self.set_cheating_obs_bounds()

        self.target_in_the_air = target_in_the_air
        self.random_block_rot = random_block_rot
        self.random_block_pos = random_block_pos
        self.random_target_pos = random_target_pos
        self.use_viewpoint_penalty = use_viewpoint_penalty
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
        self.left_gripper_finger_body_id = body_name2id["left_finger"]
        self.right_gripper_finger_body_id = body_name2id["right_finger"]
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

    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = super().step(action)

            self.current_episode_step += 1

            if terminated:
                if self.is_success():
                    reward += self.goal_complete_reward
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
            "if_reached_block": self.if_reached_block(),
            "is_success": self.is_success(),
        }
        return info

    def _get_reward(self):
        contact_reward = 0.0
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]

            if body1_id == self.left_gripper_finger_body_id or body1_id == self.right_gripper_finger_body_id:
                if body2_id == self.block_body_id:
                    contact_reward += self.gripper_finger_contact_reward

            if body2_id == self.left_gripper_finger_body_id or body2_id == self.right_gripper_finger_body_id:
                if body1_id == self.block_body_id:
                    contact_reward += self.gripper_finger_contact_reward

        distance_penalty = (
                            -np.linalg.norm(self.get_grasping_point_pos() - self.get_block_pos())
                           -np.linalg.norm(self.get_block_pos() - self.get_target_pos())
                            )

        block_pos = self.get_block_pos()

        block_pos_z = block_pos[2]
        curr_block_z_penalty = 0.0
        if block_pos_z < self.block_z_no_penalty_lower_bound or block_pos_z > self.block_z_no_penalty_upper_bound:
            # Calculate the error based on the nearest bound
            curr_block_z_penalty = np.minimum(block_pos_z - self.block_z_no_penalty_lower_bound, 0) - np.maximum(block_pos_z - self.block_z_no_penalty_upper_bound, 0)
            # print("curr_block_z_penalty: ", curr_block_z_penalty)
        # else:
            # print("No penalty for block ground contact")

        if self.use_viewpoint_penalty:
            #region_of_interest_img = self.calculate_region_of_interest_in_segmentation_image()
            #viewpoint_selection_reward = np.count_nonzero(region_of_interest_img) / region_of_interest_img.size
            grasping_point_pos = self.get_grasping_point_pos()

            target_pos = self.get_target_pos()
            rs_cam_pos = self.get_realsense_camera_pos()
            positions = np.array([grasping_point_pos, block_pos, target_pos])

            # Check if any of the positions are too close to the camera or too far from the camera
            distance_to_rs_cam = np.linalg.norm(positions - rs_cam_pos, axis=1)

            lower_bound_distance_penalty = np.minimum(distance_to_rs_cam - self.rs_cam_distance_penalty_triggering_lower_bound, 0)
            upper_bound_distance_penalty = -np.maximum(distance_to_rs_cam - self.rs_cam_distance_penalty_triggering_upper_bound, 0)

            rs_cam_angular_distances = self.calculate_realsense_camera_angular_distance_to_positions(positions)
            viewpoint_selection_penalty = (
                                                -np.sum(rs_cam_angular_distances)
                                                + np.sum(lower_bound_distance_penalty)
                                                + np.sum(upper_bound_distance_penalty)
                                            )

            #print("viewpoint_selection_penalty: ", viewpoint_selection_penalty)
            #print("distance_penalty: ", distance_penalty)

            # Among 100 different initial conditions:
            # Mean of viewpoint selection penalties: -5.179382027526286
            # Standard deviation of viewpoint selection penalties: 0.25554906594118254
            # Mean of distance penalties: -0.7780391100091665
            # Standard deviation of distance penalties: 0.14960356862545363
            # To make the viewpoint selection penalty have a similar scale to the distance penalty,
            # we multiply it by 0.15 (0.15021852141321274)

            #print("scaled viewpoint_selection_penalty: ", self.viewpoint_selection_penalty_coefficient * viewpoint_selection_penalty)
            #print("scaled distance_penalty: ", self.distance_penalty_coefficient * distance_penalty)

            reward = (self.distance_penalty_coefficient * distance_penalty
                      + self.viewpoint_selection_penalty_coefficient * viewpoint_selection_penalty
                      + contact_reward
                      + curr_block_z_penalty * self.block_z_penalty_coefficient)

            #print("distance penalty: ", self.distance_penalty_coefficient * distance_penalty)
            #print("viewpoint selection penalty: ", self.viewpoint_selection_penalty_coefficient * viewpoint_selection_penalty)
            #print("contact reward: ", contact_reward)
            #print("block z penalty: ", curr_block_z_penalty * self.block_z_penalty_coefficient)
            #print("-------------------")
        else:
            reward = self.distance_penalty_coefficient * distance_penalty + contact_reward + curr_block_z_penalty * self.block_z_penalty_coefficient

        return reward

    def _get_terminated(self):
        res = self.is_success()
        return bool(res)

    def _get_truncated(self):
        if self.max_episode_steps is None:
            return False
        return self.current_episode_step >= self.max_episode_steps - 1

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

        # Grasping point axis angle rotation (4 dim)
        grasping_point_axis_angle_rot = self.get_grasping_point_axis_angle_rot()

        # Block axis angle rotation (4 dim)
        block_axis_angle_rot = self.get_block_axis_angle_rot()

        cheating_observation = np.concatenate([grasping_point_pos,
                                               block_pos,
                                               target_pos,
                                               relative_pos_grasping_point_block,
                                               relative_pos_block_target,
                                               grasping_point_axis_angle_rot,
                                               block_axis_angle_rot])
        return cheating_observation

    def set_cheating_obs_bounds(self):
        self.cheating_obs_low = np.array([-0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -0.77, -0.77, 0.0,
                                          -1.0, -1.0, -1.0, 0.0,
                                          -1.0, -1.0, -1.0, 0.0])

        self.cheating_obs_high = np.array([0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          0.77, 1.51, 1.037,
                                          1.0, 1.0, 1.0, np.pi,
                                          1.0, 1.0, 1.0, np.pi])

    def normalize_cheating_observation(self, cheating_observation):
        return 2 * (cheating_observation - self.cheating_obs_low) / (self.cheating_obs_high - self.cheating_obs_low) - 1

    def denormalize_cheating_observation(self, cheating_observation):
        return (cheating_observation + 1) * (self.cheating_obs_high - self.cheating_obs_low) / 2 + self.cheating_obs_low

    def sample_block_and_target_pos(self):
        if self.random_block_pos:
            block_pos = np.array([
                np.random.uniform(self.block_x_range[0], self.block_x_range[1]),
                np.random.uniform(self.block_y_range[0], self.block_y_range[1]),
                self.block_z,
            ])
        else:
            block_pos = np.array([0.45, -0.1, 0.025])

        if self.random_target_pos:
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
            if self.target_in_the_air:
                target_pos = np.array([0.45, 0.1, 0.5])
            else:
                target_pos = np.array([0.45, 0.1, 0.02])
        return block_pos, target_pos

    def get_angular_distance(self):
        grasping_point_pos = self.get_grasping_point_pos()

        target_pos = self.get_target_pos()
        block_pos = self.get_block_pos()

        positions = np.array([grasping_point_pos, block_pos, target_pos])

        rs_cam_angular_distances = self.calculate_realsense_camera_angular_distance_to_positions(positions)

        return np.sum(rs_cam_angular_distances)

    def set_block_and_target_pos(self):
        block_pos, target_pos = self.sample_block_and_target_pos()

        while np.linalg.norm(block_pos - target_pos) < 3 * self.distance_threshold:
            block_pos, target_pos = self.sample_block_and_target_pos()

        # print("block_pos: ", block_pos)
        # print("target_pos: ", target_pos)

        self.model.body_pos[self.target_body_id] = target_pos

        self.data.qpos[self.block_joint_id] = block_pos[0]
        self.data.qpos[self.block_joint_id + 1] = block_pos[1]
        self.data.qpos[self.block_joint_id + 2] = block_pos[2]

        if self.random_block_rot:
            # Only randomize the yaw angle
            block_rot_quat_scipy = R.from_euler('xyz', [0, 0, np.random.uniform(0, 2 * np.pi)]).as_quat()
            block_rot = np.array([block_rot_quat_scipy[3], block_rot_quat_scipy[0], block_rot_quat_scipy[1], block_rot_quat_scipy[2]])

            self.data.qpos[self.block_joint_id + 3] = block_rot[0]
            self.data.qpos[self.block_joint_id + 4] = block_rot[1]
            self.data.qpos[self.block_joint_id + 5] = block_rot[2]
            self.data.qpos[self.block_joint_id + 6] = block_rot[3]

    def get_grasping_point_pos(self):
        return self.data.site_xpos[self.grasping_point_site_id]

    def get_grasping_point_axis_angle_rot(self):
        grasping_point_rot_mat = self.data.site_xmat[self.grasping_point_site_id]
        grasping_point_rot_mat = grasping_point_rot_mat.reshape(3, 3)
        grasping_point_axis, grasping_point_angle = mavis_utils.rotation_matrix_to_axis_angle(grasping_point_rot_mat)
        grasping_point_axis_angle_rot = np.concatenate([grasping_point_axis, [grasping_point_angle]])
        return grasping_point_axis_angle_rot

    def get_block_pos(self):
        return self.data.site_xpos[self.block_site_id]

    def get_block_axis_angle_rot(self):
        block_rot_mat = self.data.site_xmat[self.block_site_id]
        block_rot_mat = block_rot_mat.reshape(3, 3)
        block_axis, block_angle = mavis_utils.rotation_matrix_to_axis_angle(block_rot_mat)
        block_axis_angle_rot = np.concatenate([block_axis, [block_angle]])
        return block_axis_angle_rot

    def get_target_pos(self):
        return self.data.site_xpos[self.target_site_id]

    def get_realsense_camera_pos(self):
        return self.data.cam_xpos[self.realsense_camera_id]

    def get_realsense_camera_rot_mat(self):
        cam_xmat = self.data.cam_xmat[self.realsense_camera_id]
        cam_xmat = cam_xmat.reshape(3, 3)

        return cam_xmat


    def calculate_realsense_camera_angular_distance_to_positions(self, positions, debug_plot=False):
        realsense_camera_pos = self.get_realsense_camera_pos()
        realsense_camera_rot_mat = self.get_realsense_camera_rot_mat()

        cam_local_frame_initial_viewing_direction = np.array([0, 0, -1])

        current_viewing_direction = realsense_camera_rot_mat @ cam_local_frame_initial_viewing_direction
        current_viewing_direction /= np.linalg.norm(current_viewing_direction)

        directions_to_positions = positions - realsense_camera_pos
        directions_to_positions /= np.linalg.norm(directions_to_positions, axis=1, keepdims=True)

        cos_theta = np.dot(directions_to_positions, current_viewing_direction)
        angular_distances = np.arccos(np.clip(cos_theta, -1.0, 1.0))

        # print("angular_distances: ", angular_distances)
        if debug_plot:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(realsense_camera_pos[0], realsense_camera_pos[1], realsense_camera_pos[2], c='r', marker='o', label='realsense camera')

            viewing_end_point = realsense_camera_pos + current_viewing_direction
            ax.plot([realsense_camera_pos[0], viewing_end_point[0]],
                    [realsense_camera_pos[1], viewing_end_point[1]],
                    [realsense_camera_pos[2], viewing_end_point[2]], c='r', label='current viewing direction')

            for i in range(len(positions)):
                ax.scatter(positions[i, 0], positions[i, 1], positions[i, 2], c='b', marker='o', label='positions of interest')
                direction_to_end_point = realsense_camera_pos + directions_to_positions[i]

                ax.plot([realsense_camera_pos[0], direction_to_end_point[0]],
                        [realsense_camera_pos[1], direction_to_end_point[1]],
                        [realsense_camera_pos[2], direction_to_end_point[2]], c='g', label='directions to positions')

                ax.text(positions[i, 0], positions[i, 1], positions[i, 2], f'{np.degrees(angular_distances[i]):.2f}', color='k')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.title("Debug Plot for Calculating Angular Distances for Realsense Camera")
            plt.show()


        return angular_distances


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

        return seg_img_obj_id

    def calculate_region_of_interest_in_segmentation_image(self):
        seg_img_obj_id = self.get_segmentation_image()
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

    def is_success(self):
        distance_between_block_and_target = self.compute_distance_between_block_and_target()
        return distance_between_block_and_target < self.distance_threshold

    def if_reached_block(self):
        distance_between_grasping_point_and_block = self.compute_distance_between_grasping_point_and_block()
        return distance_between_grasping_point_and_block < self.distance_threshold

    def _reset_simulation(self):
        super()._reset_simulation()
        self.set_block_and_target_pos()
        self.current_episode_step = 0

    def reset(self,
              *,
              seed: Optional[int] = None,
              options: Optional[dict] = None,
              ):
        obs, info = super().reset(seed=seed, options=options)
        self.set_block_and_target_pos()
        return obs, info
