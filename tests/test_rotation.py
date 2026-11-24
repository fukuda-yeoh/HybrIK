import math

import pytest
import torch

from hybrik.utils.rotation import axis_angle_to_matrix, matrix_to_axis_angle


def test_axis_angle_to_matrix_quarter_turn():
    axis_angle = torch.tensor([[0.0, 0.0, math.pi / 2]])
    expected = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    assert torch.allclose(axis_angle_to_matrix(axis_angle), expected, atol=1e-6)


def test_round_trip_batch():
    axis_angle = torch.tensor(
        [[[0.0, 0.0, 0.0], [0.1, -0.2, 0.3]]], dtype=torch.float64
    )
    matrix = axis_angle_to_matrix(axis_angle)
    recovered = matrix_to_axis_angle(matrix)
    assert recovered.shape == axis_angle.shape
    assert torch.allclose(axis_angle_to_matrix(recovered), matrix, atol=1e-8)


@pytest.mark.parametrize("shape", [(3,), (2, 3), (2, 3, 4, 3)])
def test_shapes(shape):
    axis_angle = torch.randn(shape)
    matrix = axis_angle_to_matrix(axis_angle)
    assert matrix.shape == axis_angle.shape[:-1] + (3, 3)
    assert matrix_to_axis_angle(matrix).shape == axis_angle.shape
