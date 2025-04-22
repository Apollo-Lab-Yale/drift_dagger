import time
import mujoco
import copy
import numpy as np
import math
from typing import Tuple
import nlopt
from numba import jit

from scipy.spatial.transform import Rotation as R
from mavis_mujoco_gym.utils.mujoco_utils import MujocoModelNames


class MAVISKinematics:
    def __init__(self, lower_bounds, upper_bounds):

        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self.quat_offset = np.array([0.0, 1.0, 0.0, 0.0])

        self.bounds = list(zip(lower_bounds, upper_bounds))


    def _get_position_and_rotation_in_frame(self, data, body_id, frame_id):
        # Get position and rotation in world frame
        world_pos = data.xpos[body_id]
        world_rot = data.xquat[body_id]

        # Get position and rotation of the frame in world frame
        frame_pos = data.xpos[frame_id] - np.array([0, 0.375, 0])  # Offset by 0.375 in y-axis because of the placement of the rail
        frame_rot = data.xquat[frame_id]

        # Transform position to the frame
        frame_rot_matrix = np.zeros(9)
        mujoco.mju_quat2Mat(frame_rot_matrix, frame_rot)
        frame_rot_matrix = frame_rot_matrix.reshape(3, 3)
        relative_pos = world_pos - frame_pos
        local_pos = np.dot(frame_rot_matrix.T, relative_pos)

        # Transform rotation to the frame
        neg_frame_rot = np.zeros(4)
        mujoco.mju_negQuat(neg_frame_rot, frame_rot)
        relative_rot = np.zeros(4)
        mujoco.mju_mulQuat(relative_rot, neg_frame_rot, world_rot)

        return local_pos, relative_rot

    def bimanual_forward_kinematics(self, configuration):
        manipulation_arm_position, manipulation_arm_orientation, viewpoint_arm_position, viewpoint_arm_orientation = bimanual_forward_kinematics(configuration)

        manipulation_arm_orientation = quaternion_multiply(self.quat_offset, manipulation_arm_orientation)
        viewpoint_arm_orientation = quaternion_multiply(self.quat_offset, viewpoint_arm_orientation)

        manipulation_arm_orientation = normalize_quaternion(manipulation_arm_orientation)
        viewpoint_arm_orientation = normalize_quaternion(viewpoint_arm_orientation)

        return manipulation_arm_position, manipulation_arm_orientation, viewpoint_arm_position, viewpoint_arm_orientation

    def single_arm_forward_kinematics(self, configuration):
        position, orientation_quat = single_arm_forward_kinematics(configuration)
        orientation_quat = quaternion_multiply(self.quat_offset, orientation_quat)
        orientation_quat = normalize_quaternion(orientation_quat)
        return position, orientation_quat


class MAVISEndEffectorKinematics(MAVISKinematics):
    def __init__(self, lower_bounds, upper_bounds):
        super().__init__(lower_bounds, upper_bounds)

        self.manipulation_opt = nlopt.opt(nlopt.LD_SLSQP, 8)
        self.manipulation_opt.set_lower_bounds([bound[0] for bound in self.bounds[:8]])
        self.manipulation_opt.set_upper_bounds([bound[1] for bound in self.bounds[:8]])
        self.manipulation_opt.set_xtol_rel(1e-6)
        self.manipulation_opt.set_maxtime(0.5)

        self.viewpoint_opt = nlopt.opt(nlopt.LD_SLSQP, 8)
        self.viewpoint_opt.set_lower_bounds([bound[0] for bound in self.bounds[8:]])
        self.viewpoint_opt.set_upper_bounds([bound[1] for bound in self.bounds[8:]])
        self.viewpoint_opt.set_xtol_rel(1e-6)
        self.viewpoint_opt.set_maxtime(0.5)

    def inverse_kinematics(self, initial_configuration, target_gripper_pos, target_gripper_rot,
                           target_camera_pos, target_camera_rot, rot_repr="quat"):
        # if backend == "python":
        #ik_start_time = time.time()
        if rot_repr == "quat":
            pass
        elif rot_repr == "euler":
            target_gripper_rot = R.from_euler('xyz', target_gripper_rot).as_quat()
            target_camera_rot = R.from_euler('xyz', target_camera_rot).as_quat()

            # from scipy quaternion (X-Y-Z-W) to mujoco quaternion (W-X-Y-Z)
            # Refer to: https://github.com/clemense/quaternion-conventions
            target_gripper_rot = np.array([target_gripper_rot[3], target_gripper_rot[0], target_gripper_rot[1], target_gripper_rot[2]])
            target_camera_rot = np.array([target_camera_rot[3], target_camera_rot[0], target_camera_rot[1], target_camera_rot[2]])
        elif rot_repr == "axis-angle":
            #print("target_gripper_rot axis_angle: ", target_gripper_rot)
            #print("target_camera_rot axis_angle: ", target_camera_rot)
            target_gripper_axis = target_gripper_rot[:3]
            target_gripper_angle = target_gripper_rot[3]

            target_camera_axis = target_camera_rot[:3]
            target_camera_angle = target_camera_rot[3]

            target_gripper_rot = axis_angle_to_quaternion(target_gripper_axis, target_gripper_angle)
            target_camera_rot = axis_angle_to_quaternion(target_camera_axis, target_camera_angle)

            #print("target_gripper_rot quat: ", target_gripper_rot)
            #print("target_camera_rot quat: ", target_camera_rot)
        else:
            ValueError("Invalid rotation representation. Choose from 'quat', 'euler', 'axis-angle'.")

        target_gripper_rot = normalize_quaternion(target_gripper_rot)
        target_camera_rot = normalize_quaternion(target_camera_rot)

        target_gripper_rot = quaternion_multiply(self.quat_offset, target_gripper_rot)
        target_camera_rot = quaternion_multiply(self.quat_offset, target_camera_rot)

        target_gripper_rot = normalize_quaternion(target_gripper_rot)
        target_camera_rot = normalize_quaternion(target_camera_rot)

        try:
            self.manipulation_opt.set_min_objective(
                lambda x, grad: single_arm_ik_objective_function_nlopt(x, grad, target_gripper_pos, target_gripper_rot)[
                    0])
            self.viewpoint_opt.set_min_objective(
                lambda x, grad: single_arm_ik_objective_function_nlopt(x, grad, target_camera_pos, target_camera_rot)[
                    0])

            manipulation_result = self.manipulation_opt.optimize(initial_configuration[:8])
            viewpoint_result = self.viewpoint_opt.optimize(initial_configuration[8:])
            result = np.concatenate((manipulation_result, viewpoint_result))
            return result
        except (nlopt.RoundoffLimited, nlopt.ForcedStop, Exception) as e:
            print("Inverse kinematics failed with error:", str(e))
            manipulation_result = initial_configuration[:8]
            viewpoint_result = initial_configuration[8:]
            result = np.concatenate((manipulation_result, viewpoint_result))
            return result


        #ik_end_time = time.time()
        #print("Inverse kinematics took:", ik_end_time - ik_start_time, "seconds.")




class MAVISLookAtKinematics(MAVISKinematics):
    def __init__(self, lower_bounds, upper_bounds):
        super().__init__(lower_bounds, upper_bounds)

        self.lookat_opt = nlopt.opt(nlopt.LD_SLSQP, 16)
        self.lookat_opt.set_lower_bounds([bound[0] for bound in self.bounds])
        self.lookat_opt.set_upper_bounds([bound[1] for bound in self.bounds])
        self.lookat_opt.set_xtol_rel(1e-6)
        self.lookat_opt.set_maxtime(0.5)

    def get_lookat_forward_kinematics(self, configuration, is_quat=True):
        curr_gripper_pos, curr_gripper_quat, curr_camera_pos, curr_camera_quat = jointly_bimanual_forward_kinematics(configuration)

        curr_gripper_quat = quaternion_multiply(self.quat_offset, curr_gripper_quat)

        curr_gripper_quat = normalize_quaternion(curr_gripper_quat)

        if not is_quat:
            # from mujoco quaternion (W-X-Y-Z) to scipy quaternion (X-Y-Z-W)
            curr_gripper_quat = np.array([curr_gripper_quat[1], curr_gripper_quat[2], curr_gripper_quat[3], curr_gripper_quat[0]])
            curr_gripper_rot = R.from_quat(curr_gripper_quat).as_euler('xyz')
            return np.concatenate((curr_gripper_pos, curr_gripper_rot, curr_camera_pos))

        # [0:3]: manipulation arm end-effector position
        # [3:7]: manipulation arm end-effector orientation in quaternion
        # [7:10]: viewpoint arm end-effector position
        return np.concatenate((curr_gripper_pos, curr_gripper_quat, curr_camera_pos))


    def inverse_kinematics(self, initial_configuration, target_gripper_pos, target_gripper_quat,
                           target_camera_pos, rot_repr="quat", lookat_offset=np.array([0.0, 0.0, -0.15])):
        #ik_start_time = time.time()

        if rot_repr == "quat":
            pass
        elif rot_repr == "euler":
            target_gripper_quat = R.from_euler('xyz', target_gripper_quat).as_quat()
            target_gripper_quat = np.array([target_gripper_quat[3], target_gripper_quat[0], target_gripper_quat[1], target_gripper_quat[2]])
        elif rot_repr == "axis-angle":
            #print("target_gripper_rot axis_angle: ", target_gripper_quat)
            target_gripper_axis = target_gripper_quat[:3]
            target_gripper_angle = target_gripper_quat[3]

            target_gripper_quat = axis_angle_to_quaternion(target_gripper_axis, target_gripper_angle)
            #print("target_gripper_rot quat: ", target_gripper_quat)
        else:
            ValueError("Invalid rotation representation. Choose from 'quat', 'euler', 'axis-angle'.")

        target_gripper_quat = normalize_quaternion(target_gripper_quat)

        target_gripper_quat = quaternion_multiply(self.quat_offset, target_gripper_quat)

        target_gripper_quat = normalize_quaternion(target_gripper_quat)

        try:
            self.lookat_opt.set_min_objective(lambda x, grad:
                                              bimanual_lookat_ik_objective_function_nlopt(x, grad, target_gripper_pos,
                                                                                          target_gripper_quat,
                                                                                          target_camera_pos, lookat_offset)[0])
            result = self.lookat_opt.optimize(initial_configuration)
            return result
        except (nlopt.RoundoffLimited, nlopt.ForcedStop, Exception) as e:
            print("Inverse kinematics failed with error:", str(e))
            result = initial_configuration
            return result

        #ik_end_time = time.time()

        #print("Inverse kinematics took:", ik_end_time - ik_start_time, "seconds.")

        # plot gripper pos as a 3D point, and plot the looker link as a line with arrow head for pointing direction
        #import matplotlib.pyplot as plt
        #fig = plt.figure()
        #ax = fig.add_subplot(111, projection='3d')
        #looker_link_pos, looker_link_quat = self.single_arm_forward_kinematics(result[8:])
        #gripper_pos, gripper_quat = self.single_arm_forward_kinematics(result[:8])
        #ax.scatter(gripper_pos[0], gripper_pos[1], gripper_pos[2], c='r', marker='o')
        #ax.scatter(looker_link_pos[0], looker_link_pos[1], looker_link_pos[2], c='b', marker='o')
        #looker_link_direction = quaternion_rotate(looker_link_quat, np.array([0.0, 0.0, 1.0]))
        #looker_link_direction = looker_link_direction / np.linalg.norm(looker_link_direction)
        #ax.quiver(looker_link_pos[0], looker_link_pos[1], looker_link_pos[2],
        #          looker_link_direction[0], looker_link_direction[1], looker_link_direction[2],
        #          length=0.1)
        #plt.show()


@jit(nopython=True)
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1e-6:
        return v
    return v / norm


@jit(nopython=True)
def axis_angle_to_quaternion(axis, angle):
    """Convert axis-angle representation to a unit quaternion (w, x, y, z)."""
    axis = normalize_vector(axis)
    half_angle = angle / 2.0
    sin_half_angle = np.sin(half_angle)
    cos_half_angle = np.cos(half_angle)

    w = cos_half_angle
    x = axis[0] * sin_half_angle
    y = axis[1] * sin_half_angle
    z = axis[2] * sin_half_angle

    return np.array([w, x, y, z])


@jit(nopython=True)
def quaternion_to_axis_angle(quaternion):
    """Convert a unit quaternion (w, x, y, z) to axis-angle representation."""
    w, x, y, z = quaternion
    angle = 2.0 * np.arccos(w)
    sin_half_angle = np.sqrt(1.0 - w * w)

    if sin_half_angle < 1e-6:
        axis = np.array([1.0, 0.0, 0.0])  # Default axis when angle is zero
    else:
        axis = np.array([x, y, z]) / sin_half_angle
        axis = normalize_vector(axis)

    # Adjust angle to be within [0, pi] and reverse axis if necessary
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis

    return axis, angle


@jit(nopython=True)
def rotation_matrix_to_axis_angle(matrix):
    """Convert a rotation matrix to axis-angle representation."""
    # Ensure the matrix is of type float
    matrix = matrix.astype(np.float64)

    angle = np.arccos((np.trace(matrix) - 1) / 2.0)

    if angle < 1e-6:
        # If the angle is very small, return a default axis
        axis = np.array([1.0, 0.0, 0.0])
        angle = 0.0
    else:
        x = matrix[2, 1] - matrix[1, 2]
        y = matrix[0, 2] - matrix[2, 0]
        z = matrix[1, 0] - matrix[0, 1]
        axis = np.array([x, y, z])
        axis = normalize_vector(axis)

    # Adjust angle to be within [0, pi] and reverse axis if necessary
    if angle > np.pi:
        angle = 2 * np.pi - angle
        axis = -axis

    return axis, angle

@jit(nopython=True)
def normalize_quaternion(q):
    norm = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2 + q[3] ** 2)
    return np.array([q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm])

@jit(nopython=True)
def quaternion_conjugate(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

@jit(nopython=True)
def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])


@jit(nopython=True)
def H1_wxyz_quat_logarithm(q):
    w, x, y, z = q
    theta = math.acos(w)
    sin_theta = math.sin(theta)
    if sin_theta == 0:
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        scalar = theta / sin_theta
        return np.array([0, scalar * x, scalar * y, scalar * z], dtype=np.float64)

@jit(nopython=True)
def quaternion_displacement_based_distance(q1, q2):
    # Compute the displacement-based distance between two quaternions based on Danny's notes
    # Make sure they are unit quaternions
    q1 = normalize_quaternion(q1)
    q2 = normalize_quaternion(q2)

    # Quaternion conjugate is the same as the inverse for unit quaternions
    disp1 = H1_wxyz_quat_logarithm(quaternion_multiply(quaternion_conjugate(q1), q2))
    disp2 = H1_wxyz_quat_logarithm(quaternion_multiply(quaternion_conjugate(q1), -q2))

    # Calculate norm on the vector part because of the vee operator
    error1 = np.linalg.norm(disp1[1:])
    error2 = np.linalg.norm(disp2[1:])
    return min(error1, error2)


@jit(nopython=True)
def matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    trace = np.trace(matrix)
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (matrix[2, 1] - matrix[1, 2]) / s
        y = (matrix[0, 2] - matrix[2, 0]) / s
        z = (matrix[1, 0] - matrix[0, 1]) / s
    elif (matrix[0, 0] > matrix[1, 1]) and (matrix[0, 0] > matrix[2, 2]):
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
        w = (matrix[2, 1] - matrix[1, 2]) / s
        x = 0.25 * s
        y = (matrix[0, 1] + matrix[1, 0]) / s
        z = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
        w = (matrix[0, 2] - matrix[2, 0]) / s
        x = (matrix[0, 1] + matrix[1, 0]) / s
        y = 0.25 * s
        z = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
        w = (matrix[1, 0] - matrix[0, 1]) / s
        x = (matrix[0, 2] + matrix[2, 0]) / s
        y = (matrix[1, 2] + matrix[2, 1]) / s
        z = 0.25 * s

    normalized_quat = normalize_quaternion(np.array([w, x, y, z], dtype=np.float64))

    return normalized_quat


@jit(nopython=True)
def single_arm_forward_kinematics(configuration: np.ndarray, linear_motor_x_offset=0.0) -> Tuple[np.ndarray, np.ndarray]:
    linear_position = configuration[0]
    joint_angles = configuration[1:]

    linear_motor_pos = np.array([linear_motor_x_offset, linear_position, 0.0], dtype=np.float64)
    # DH parameters: https://help.ufactory.cc/en/articles/4330809-kinematic-and-dynamic-parameters-of-ufactory-xarm-series
    dh_params = np.array([
        [0.0, 0.267, -np.pi / 2, joint_angles[0]],
        [0.0, 0.0, np.pi / 2, joint_angles[1]],
        [0.0525, 0.293, np.pi / 2, joint_angles[2]],
        [0.0775, 0.0, np.pi / 2, joint_angles[3]],
        [0.0, 0.3425, np.pi / 2, joint_angles[4]],
        [0.076, 0.0, -np.pi / 2, joint_angles[5]],
        [0.0, 0.097, 0.0, joint_angles[6]]
    ], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    for i in range(dh_params.shape[0]):
        a, d, alpha, theta = dh_params[i]
        T_i = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta)],
            [np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta)],
            [0.0, np.sin(alpha), np.cos(alpha), d],
            [0.0, 0.0, 0.0, 1.0]
        ], dtype=np.float64)
        T = T @ T_i

    T_linear = np.eye(4, dtype=np.float64)
    T_linear[:3, 3] = linear_motor_pos
    T = T_linear @ T

    position = T[:3, 3]
    orientation_matrix = T[:3, :3]
    orientation_quat = matrix_to_quaternion(orientation_matrix)

    return position, orientation_quat


@jit(nopython=True)
def bimanual_forward_kinematics(configuration: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manipulation_arm = configuration[:8]
    viewpoint_arm = configuration[8:]

    manipulation_arm_position, manipulation_arm_orientation = single_arm_forward_kinematics(manipulation_arm)
    viewpoint_arm_position, viewpoint_arm_orientation = single_arm_forward_kinematics(viewpoint_arm)

    return manipulation_arm_position, manipulation_arm_orientation, viewpoint_arm_position, viewpoint_arm_orientation


@jit(nopython=True)
def compute_transform_error(curr_pos, curr_quat, target_pos, target_quat):
    pos_error = np.linalg.norm(curr_pos - target_pos)
    quat_error = quaternion_displacement_based_distance(curr_quat, target_quat)

    return pos_error + quat_error


@jit(nopython=True)
def bimanual_eef_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat):
    curr_gripper_pos, curr_gripper_quat, curr_camera_pos, curr_camera_quat = bimanual_forward_kinematics(configuration)
    gripper_transform_error = compute_transform_error(curr_gripper_pos, curr_gripper_quat, target_gripper_pos, target_gripper_quat)
    camera_transform_error = compute_transform_error(curr_camera_pos, curr_camera_quat, target_camera_pos, target_camera_quat)
    error = gripper_transform_error + camera_transform_error
    return error

@jit(nopython=True)
def compute_bimanual_eef_ik_obj_func_grad_finite_diff(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat, perturbation=1e-6):
    configuration = np.copy(configuration)
    grad = np.zeros_like(configuration)

    for i in range(len(configuration)):
        original_value = configuration[i]
        configuration[i] += perturbation
        error_plus = bimanual_eef_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat)

        configuration[i] = original_value - perturbation
        error_minus = bimanual_eef_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat)

        grad[i] = (error_plus - error_minus) / (2 * perturbation)
        configuration[i] = original_value

    return grad

@jit(nopython=True)
def bimanual_eef_ik_objective_function_nlopt(configuration, grad, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat):
    if grad.shape[0] != configuration.shape[0]:
        grad = np.zeros_like(configuration)
    error = bimanual_eef_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat)
    grad[:] = compute_bimanual_eef_ik_obj_func_grad_finite_diff(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, target_camera_quat)
    return error, grad

@jit(nopython=True)
def single_arm_ik_objective_function(configuration, target_pos, target_quat):
    curr_pos, curr_quat = single_arm_forward_kinematics(configuration)
    error = compute_transform_error(curr_pos, curr_quat, target_pos, target_quat)
    return error

@jit(nopython=True)
def compute_single_arm_ik_obj_func_grad_finite_diff(configuration, target_pos, target_quat, perturbation=1e-6):
    configuration = np.copy(configuration)
    grad = np.zeros_like(configuration)

    for i in range(len(configuration)):
        original_value = configuration[i]
        configuration[i] += perturbation
        error_plus = single_arm_ik_objective_function(configuration, target_pos, target_quat)

        configuration[i] = original_value - perturbation
        error_minus = single_arm_ik_objective_function(configuration, target_pos, target_quat)

        grad[i] = (error_plus - error_minus) / (2 * perturbation)
        configuration[i] = original_value

    return grad


def single_arm_ik_objective_function_nlopt(configuration, grad, target_pos, target_quat):
    try:
        if grad.shape[0] != configuration.shape[0]:
            grad = np.zeros_like(configuration)
        error = single_arm_ik_objective_function(configuration, target_pos, target_quat)
        grad[:] = compute_single_arm_ik_obj_func_grad_finite_diff(configuration, target_pos, target_quat)
        return error, grad
    except Exception as e:
        print("Error in objective function:", str(e))
        return np.inf, np.zeros_like(grad)


############# lookat IK ####################


@jit(nopython=True)
def proj_scalar(v, u):
    return np.dot(v, u) / np.dot(u, u)


@jit(nopython=True)
def proj_clamped(v, u):
    scalar = proj_scalar(v, u)
    scalar = min(max(scalar, 0), 1)
    return scalar * u


@jit(nopython=True)
def proj_onto_line_segment(p, a, b):
    return proj_clamped(p - a, b - a) + a


@jit(nopython=True)
def quaternion_rotate(q, v):
    qv = np.array([0.0, v[0], v[1], v[2]])
    normalized_q = normalize_quaternion(q)
    return quaternion_multiply(quaternion_multiply(normalized_q, qv), quaternion_conjugate(normalized_q))[1:]


@jit(nopython=True)
def jointly_bimanual_forward_kinematics(configuration: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    manipulation_arm = configuration[:8]
    viewpoint_arm = configuration[8:]

    manipulation_arm_position, manipulation_arm_orientation = single_arm_forward_kinematics(manipulation_arm)
    viewpoint_arm_position, viewpoint_arm_orientation = single_arm_forward_kinematics(viewpoint_arm, linear_motor_x_offset=0.9)

    return manipulation_arm_position, manipulation_arm_orientation, viewpoint_arm_position, viewpoint_arm_orientation


@jit(nopython=True)
def calculate_look_at_error(gripper_pos, viewpoint_pos, viewpoint_quat, lookat_offset=np.array([0.0, 0.0, -0.15])):
    v = quaternion_rotate(viewpoint_quat, np.array([0.0, 0.0, 1.0]))

    a = viewpoint_pos
    b = viewpoint_pos + 999 * v

    offset = lookat_offset
    gripper_pos += offset

    proj_t_v = proj_onto_line_segment(gripper_pos, a, b)

    error = np.sum((proj_t_v - gripper_pos) ** 2)

    return error


@jit(nopython=True)
def calculate_side_axis_error(viewpoint_quat):
    s = quaternion_rotate(viewpoint_quat, np.array([0.0, 1.0, 0.0]))
    up = np.array([0.0, 0.0, 1.0])
    error = np.dot(s, up) ** 2
    return error


@jit(nopython=True)
def bimanual_lookat_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, lookat_offset=np.array([0.0, 0.0, -0.15])):
    curr_gripper_pos, curr_gripper_quat, curr_camera_pos, curr_camera_quat = jointly_bimanual_forward_kinematics(configuration)
    gripper_transform_error = compute_transform_error(curr_gripper_pos, curr_gripper_quat, target_gripper_pos, target_gripper_quat)
    camera_look_at_error = calculate_look_at_error(curr_gripper_pos, curr_camera_pos, curr_camera_quat, lookat_offset)
    camera_pos_error = np.linalg.norm(curr_camera_pos - target_camera_pos)
    side_axis_error = calculate_side_axis_error(curr_camera_quat)

    # TODO: Implement the error for delta

    error = gripper_transform_error + 0.5 * camera_look_at_error + 0.5 * side_axis_error + 0.5 * camera_pos_error
    return error


@jit(nopython=True)
def compute_bimanual_lookat_ik_obj_func_grad_finite_diff(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, perturbation=1e-6,
                                                         lookat_offset=np.array([0.0, 0.0, -0.15])):
    configuration = np.copy(configuration)
    grad = np.zeros_like(configuration)

    for i in range(len(configuration)):
        original_value = configuration[i]
        configuration[i] += perturbation
        error_plus = bimanual_lookat_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, lookat_offset)

        configuration[i] = original_value - perturbation
        error_minus = bimanual_lookat_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, lookat_offset)

        grad[i] = (error_plus - error_minus) / (2 * perturbation)
        configuration[i] = original_value

    return grad


def bimanual_lookat_ik_objective_function_nlopt(configuration, grad, target_gripper_pos, target_gripper_quat, target_camera_pos,
                                                lookat_offset=np.array([0.0, 0.0, -0.15])):
    try:
        if grad.shape[0] != configuration.shape[0]:
            grad = np.zeros_like(configuration)
        error = bimanual_lookat_ik_objective_function(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, lookat_offset=lookat_offset)
        grad[:] = compute_bimanual_lookat_ik_obj_func_grad_finite_diff(configuration, target_gripper_pos, target_gripper_quat, target_camera_pos, lookat_offset=lookat_offset)
        return error, grad
    except Exception as e:
        print("Error in objective function:", str(e))
        return np.inf, np.zeros_like(grad)

