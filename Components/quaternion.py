import torch
import torch.nn as nn
import torch.nn.functional as F





#################################################
#
# Quaternion representation and conversion functions
#   q = (w, x, y, z) with real part first
#
#################################################



import torch
import numpy as np


def to_axis_angle(quaternions):
    """
    Convert quaternions to axis-angle representation.

    Args:
        quaternions: Tensor of shape (..., 4), real part first (w, x, y, z).

    Returns:
        Tensor of shape (..., 3): axis-angle vectors.
    """


    # Normalize to be safe
    quaternions = quaternions / quaternions.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)

    w = quaternions[..., :1]           # (..., 1)
    v = quaternions[..., 1:]           # (..., 3)
    v_norm = v.norm(p=2, dim=-1, keepdim=True)  # (..., 1)

    # Compute full rotation angle
    angles = 2.0 * torch.atan2(v_norm, w)

    # Avoid division by zero (small angles)
    small = v_norm < 1e-8
    scale = torch.where(
        small,
        torch.ones_like(v_norm),          # arbitrary axis for tiny angles
        angles / v_norm
    )

    axis_angle = v * scale
    return axis_angle

def split_axis_angle(axis_angle):
    """
    Split axis-angle representation into axis and angle.

    Args:
        axis_angle: Tensor of shape (..., 3), where the magnitude is
            the rotation angle in radians around the vector's direction.    
    Returns:
        A tuple of two tensors:
            - axes: Tensor of shape (..., 3), unit vectors.
            - angles: Tensor of shape (...,), rotation angles in radians.
    """


    angles = torch.norm(axis_angle, p=2, dim=-1)  # (...,)
    axes = axis_angle / angles.unsqueeze(-1).clamp_min(1e-8)  # (..., 3)
    return axes, angles



def from_axis_angle(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Tensor of shape (..., 3), where the magnitude is
            the rotation angle in radians around the vector's direction.

    Returns:
        Tensor of shape (..., 4): quaternions with real part first.
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)  # (..., 1)
    half_angles = 0.5 * angles

    # Handle small angles robustly
    # sinc(x) = sin(pi*x)/(pi*x), so torch.sinc(half_angles/pi) = sin(half_angles)/half_angles
    sin_half_over_angle = torch.sinc(half_angles / torch.pi) * 0.5

    quats = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_over_angle], dim=-1
    )
    return quats



def to_matrix(quaternions, dtype):
    """
    Convert quaternions to rotation matrices.

    Args:
        quaternions: Tensor of shape (..., 4), real part first (w, x, y, z).

    Returns:
        Tensor of shape (..., 3, 3): rotation matrices.
    """
    # Normalize to unit quaternions
    quaternions = quaternions / quaternions.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)

    w, x, y, z = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1, keepdim=True)

    o = torch.stack([
        1 - two_s * (y * y + z * z),
        two_s * (x * y - z * w),
        two_s * (x * z + y * w),

        two_s * (x * y + z * w),
        1 - two_s * (x * x + z * z),
        two_s * (y * z - x * w),

        two_s * (x * z - y * w),
        two_s * (y * z + x * w),
        1 - two_s * (x * x + y * y),
    ], dim=-1)

    R = o.reshape(quaternions.shape[:-1] + (3, 3))
    R = R.to(dtype=dtype)
    return R


def slerp(q1, q2, alpha):

        # Normalize both quaternions to ensure valid rotations
        q1 = F.normalize(q1, dim=0)
        q2 = F.normalize(q2, dim=0)

        dot = torch.dot(q1, q2)

        # If the dot product is negative, slerp the opposite quaternion to avoid long-path interpolation
        if dot < 0.0:
            q2 = -q2
            dot = -dot

        # Clamp dot to avoid numerical errors
        dot = torch.clamp(dot, -1.0, 1.0)

        # If the quaternions are very close, use linear interpolation to avoid instability
        if dot > 0.9995:
            result = (1 - alpha) * q1 + alpha * q2
            result = F.normalize(result, dim=0)
            return result
        
        theta_0 = torch.acos(dot)  # angle between input vectors
        sin_theta_0 = torch.sin(theta_0)

        theta = theta_0 * alpha
        sin_theta = torch.sin(theta)

        s1 = torch.sin(theta_0 - theta) / sin_theta_0
        s2 = sin_theta / sin_theta_0

        result = s1 * q1 + s2 * q2
        return result


def norm(quaternions: torch.Tensor) -> torch.Tensor:
        """
        Compute the L2 norm of quaternions.
        Args:
            quaternions: Tensor of shape (..., 4).
        Returns:
            Tensor of shape (...,): L2 norms.
        """
        norm = torch.norm(quaternions, p=2, dim=-1)
        return norm

def normalize(quaternions: torch.Tensor) -> torch.Tensor:
    """Normalize quaternions to unit length.
    Args:
        quaternions: Tensor of shape (..., 4).
        Returns:
        Tensor of shape (..., 4): normalized quaternions.
    """
    quaternions = F.normalize(quaternions, p=2, dim=-1)
    return quaternions


