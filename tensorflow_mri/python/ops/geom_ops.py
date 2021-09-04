# Copyright 2021 University College London. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Geometry operations."""

from tensorflow_graphics.geometry.transformation import rotation_matrix_2d
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d


def rotate_2d(points, euler):
  """Rotates an array of 2D coordinates.

  Args:
    points: A `Tensor` of shape `[A1, A2, ..., An, 2]`, where the last dimension
      represents a 2D point.
    euler: A `Tensor` of shape `[A1, A2, ..., An, 1]`, where the last dimension
      represents an angle in radians.

  Returns:
    A `Tensor` of shape `[A1, A2, ..., An, 2]`, where the last dimension
    represents a 2D point.
  """
  return rotation_matrix_2d.rotate(
      points, rotation_matrix_2d.from_euler(euler))


def rotate_3d(points, euler):
  """Rotates an array of 3D coordinates.

  Args:
    points: A `Tensor` of shape `[A1, A2, ..., An, 3]`, where the last dimension
      represents a 3D point.
    euler: A `Tensor` of shape `[A1, A2, ..., An, 3]`, where the last dimension
      represents the three Euler angles.

  Returns:
    A `Tensor` of shape `[A1, A2, ..., An, 3]`, where the last dimension
    represents a 3D point.
  """
  return rotation_matrix_3d.rotate(
      points, rotation_matrix_3d.from_euler(euler))
