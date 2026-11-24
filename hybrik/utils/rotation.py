"""Rotation conversions implemented with PyTorch only.

These helpers replace the small subset of ``pytorch3d.transforms`` that HybrIK
used for SMPL-X inference. Keeping them here means the SMPL-X code path no
longer requires PyTorch3D (which has to be compiled from source for current
Python versions). PyTorch3D is still used for the optional mesh overlay
renderer in :mod:`hybrik.utils.render_pytorch3d`.

The quaternion conventions and conversion math follow PyTorch3D's
``pytorch3d.transforms.rotation_conversions`` (BSD-3-Clause): quaternions
are ``(w, x, y, z)``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
]


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """Return ``sqrt(max(x, 0))`` so tiny negative round-off does not produce NaN."""
    return torch.sqrt(torch.clamp(x, min=0.0))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors ``(..., 3)`` to quaternions ``(..., 4)``."""
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps

    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # For small x, sin(x/2)/x ~= 1/2 - x^2/48.
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )

    return torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles],
        dim=-1,
    )


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``(..., 4)`` to rotation matrices ``(..., 3, 3)``."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` to quaternions ``(..., 4)``."""
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # Each candidate is the desired quaternion multiplied by r, i, j or k.
    quat_by_rijk = torch.stack(
        [
            torch.stack(
                [q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1
            ),
            torch.stack(
                [m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1
            ),
            torch.stack(
                [m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1
            ),
            torch.stack(
                [m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1
            ),
        ],
        dim=-2,
    )

    # Floor the divisor so a candidate that is numerically zero is not selected.
    flr = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    best = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    return quat_candidates[best, :].reshape(batch_dim + (4,))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions ``(..., 4)`` to axis-angle vectors ``(..., 3)``."""
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles

    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )

    return quaternions[..., 1:] / sin_half_angles_over_angles


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle vectors ``(..., 3)`` to rotation matrices ``(..., 3, 3)``."""
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices ``(..., 3, 3)`` to axis-angle vectors ``(..., 3)``."""
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))
