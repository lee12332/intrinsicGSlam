import numpy as np
import torch
from scipy.spatial.transform import Rotation
from typing import Union
from src.utils.utils import np2torch


def multiply_quaternions(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Performs batch-wise quaternion multiplication.

    Given two quaternions, this function computes their product. The operation is
    vectorized and can be performed on batches of quaternions.

    Args:
        q: A tensor representing the first quaternion or a batch of quaternions. 
           Expected shape is (... , 4), where the last dimension contains quaternion components (w, x, y, z).
        r: A tensor representing the second quaternion or a batch of quaternions with the same shape as q.
    Returns:
        A tensor of the same shape as the input tensors, representing the product of the input quaternions.
    """
    w0, x0, y0, z0 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    w1, x1, y1, z1 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]

    w = -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0
    x = x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0
    y = -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0
    z = x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0
    return torch.stack((w, x, y, z), dim=-1)


def transformation_to_quaternion(RT: Union[torch.Tensor, np.ndarray]):
    """ Converts a rotation-translation matrix to a tensor representing quaternion and translation.

    This function takes a 3x4 transformation matrix (rotation and translation) and converts it
    into a tensor that combines the quaternion representation of the rotation and the translation vector.

    Args:
        RT: A 3x4 matrix representing the rotation and translation. This can be a NumPy array
            or a torch.Tensor. If it's a torch.Tensor and resides on a GPU, it will be moved to CPU.

    Returns:
        A tensor combining the quaternion (in w, x, y, z order) and translation vector. The tensor
        will be moved to the original device if the input was a GPU tensor.
    """
    gpu_id = -1
    if isinstance(RT, torch.Tensor):
        if RT.get_device() != -1:
            RT = RT.detach().cpu()
            gpu_id = RT.get_device()
        RT = RT.numpy()
    R, T = RT[:3, :3], RT[:3, 3]

    rot = Rotation.from_matrix(R)
    quad = rot.as_quat(canonical=True)
    quad = np.roll(quad, 1)
    tensor = np.concatenate([quad, T], 0)
    tensor = torch.from_numpy(tensor).float()
    if gpu_id != -1:
        tensor = tensor.to(gpu_id)
    return tensor


def extrapolate_poses(poses: np.ndarray) -> np.ndarray:
    """ Generates an interpolated pose based on the first two poses in the given array.
    Args:
        poses: An array of poses, where each pose is represented by a 4x4 transformation matrix.
    Returns:
        A 4x4 numpy ndarray representing the interpolated transformation matrix.
    """
    return poses[1, :] @ np.linalg.inv(poses[0, :]) @ poses[1, :]


def compute_camera_opt_params(estimate_rel_w2c: np.ndarray) -> tuple:
    """ Computes the camera's rotation and translation parameters from an world-to-camera transformation matrix.
    This function extracts the rotation component of the transformation matrix, converts it to a quaternion,
    and reorders it to match a specific convention. Both rotation and translation parameters are converted
    to torch Parameters and intended to be optimized in a PyTorch model.
    Args:
        estimate_rel_w2c: A 4x4 numpy ndarray representing the estimated world-to-camera transformation matrix.
    Returns:
        A tuple containing two torch.nn.Parameters: camera's rotation and camera's translation.
    """
    quaternion = Rotation.from_matrix(estimate_rel_w2c[:3, :3]).as_quat(canonical=True)
    quaternion = quaternion[[3, 0, 1, 2]]
    opt_cam_rot = torch.nn.Parameter(np2torch(quaternion, "cuda"))
    opt_cam_trans = torch.nn.Parameter(np2torch(estimate_rel_w2c[:3, 3], "cuda"))
    return opt_cam_rot, opt_cam_trans

def random_extrapolate_poses(poses: np.ndarray, 
                     rotation_noise_deg: float = 0.1, 
                     translation_noise: float = 0.001) -> np.ndarray:
    delta_pose = poses[1] @ np.linalg.inv(poses[0])
    predicted_pose = poses[1] @ delta_pose
    return _add_pose_noise(predicted_pose, rotation_noise_deg, translation_noise)

def _add_pose_noise(
    pose: np.ndarray,
    rotation_noise_deg: float = 1.0,
    translation_noise: float = 0.01,
    max_attempts: int = 10,
    ortho_threshold: float = 1e-6
) -> np.ndarray:
    """鲁棒的位姿扰动函数（带正交性验证和循环修复）"""
    original_pose = pose.copy()
    for _ in range(max_attempts):
        # 生成随机扰动
        delta_rot = _generate_rotation_noise(rotation_noise_deg)
        delta_trans = _generate_translation_noise(translation_noise)
        # 应用扰动
        noisy_pose = _apply_perturbation(original_pose, delta_rot, delta_trans)
        # 正交性验证
        if _validate_orthogonality(noisy_pose, ortho_threshold):
            return noisy_pose
        # 正交性修复 (SVD 正交化)
        noisy_pose = _orthogonalize(noisy_pose)
        if _validate_orthogonality(noisy_pose, ortho_threshold):
            return noisy_pose
    # 最终修复保证正交性
    return _orthogonalize(original_pose)

def _generate_rotation_noise(noise_deg: float) -> np.ndarray:
    """生成满足SO(3)的随机旋转扰动"""
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.deg2rad(np.random.uniform(-noise_deg, noise_deg))
    return Rotation.from_rotvec(angle * axis).as_matrix()

def _generate_translation_noise(noise: float) -> np.ndarray:
    """生成平移扰动"""
    return np.random.uniform(-noise, noise, 3)

def _apply_perturbation(pose: np.ndarray, delta_rot: np.ndarray, delta_trans: np.ndarray) -> np.ndarray:
    """应用旋转和平移扰动"""
    noisy_pose = np.eye(4)
    noisy_pose[:3, :3] = delta_rot @ pose[:3, :3]  # 旋转组合
    noisy_pose[:3, 3] = pose[:3, 3] + delta_trans
    return noisy_pose

def _validate_orthogonality(pose: np.ndarray, threshold: float) -> bool:
    """正交性验证 (双重验证策略)"""
    R = pose[:3, :3]
    # 条件1: R^T R ≈ I
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
    # 条件2: det(R) = 1
    det_error = abs(np.linalg.det(R) - 1)
    return (ortho_error < threshold) and (det_error < threshold)

def _orthogonalize(pose: np.ndarray) -> np.ndarray:
    """SVD正交化修复"""
    U, _, Vt = np.linalg.svd(pose[:3, :3])
    repaired_rot = U @ Vt
    # 保证右手坐标系
    if np.linalg.det(repaired_rot) < 0:
        U[:, -1] *= -1
        repaired_rot = U @ Vt
    pose[:3, :3] = repaired_rot
    return pose
