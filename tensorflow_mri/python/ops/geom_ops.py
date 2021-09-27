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

import tensorflow as tf

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


def euler_to_rotation_matrix_3d(angles, order='XYZ', name='rotation_matrix_3d'):
  r"""Convert an Euler angle representation to a rotation matrix.

  The resulting matrix is $$\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$$.

  .. note::
    In the following, A1 to An are optional batch dimensions.

  Args:
    angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the three Euler angles. `[A1, ..., An, 0]` is the angle about
      `x` in radians `[A1, ..., An, 1]` is the angle about `y` in radians and
      `[A1, ..., An, 2]` is the angle about `z` in radians.
    order: A `str`. The order in which the rotations are applied. Defaults to
      `"XYZ"`.
    name: A name for this op that defaults to "rotation_matrix_3d_from_euler".

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If the shape of `angles` is not supported.
  """
  with tf.name_scope(name):
    angles = tf.convert_to_tensor(value=angles)

    if angles.shape[-1] != 3:
      raise ValueError(f"The last dimension of `angles` must have size 3, "
                       f"but got shape: {angles.shape}")

    sin_angles = tf.math.sin(angles)
    cos_angles = tf.math.cos(angles)
    return _build_matrix_from_sines_and_cosines(
        sin_angles, cos_angles, order=order)


def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles, order='XYZ'):
  """Builds a rotation matrix from sines and cosines of Euler angles.

  .. note::
    In the following, A1 to An are optional batch dimensions.

  Args:
    sin_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the cosine of the Euler angles.
    order: A `str`. The order in which the rotations are applied. Defaults to
      `"XYZ"`.

  Returns:
    A tensor of shape `[A1, ..., An, 3, 3]`, where the last two dimensions
    represent a 3d rotation matrix.

  Raises:
    ValueError: If any of the input arguments has an invalid value.
  """
  sin_angles.shape.assert_is_compatible_with(cos_angles.shape)
  output_shape = tf.concat((tf.shape(sin_angles)[:-1], (3, 3)), -1)

  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  ones = tf.ones_like(sx)
  zeros = tf.zeros_like(sx)
  # rx
  m00 = ones
  m01 = zeros
  m02 = zeros
  m10 = zeros
  m11 = cx
  m12 = -sx
  m20 = zeros
  m21 = sx
  m22 = cx
  rx = tf.stack((m00, m01, m02,
                 m10, m11, m12,
                 m20, m21, m22),
                axis=-1)
  rx = tf.reshape(rx, output_shape)
  # ry
  m00 = cy
  m01 = zeros
  m02 = sy
  m10 = zeros
  m11 = ones
  m12 = zeros
  m20 = -sy
  m21 = zeros
  m22 = cy
  ry = tf.stack((m00, m01, m02,
                 m10, m11, m12,
                 m20, m21, m22),
                axis=-1)
  ry = tf.reshape(ry, output_shape)
  # rz
  m00 = cz
  m01 = -sz
  m02 = zeros
  m10 = sz
  m11 = cz
  m12 = zeros
  m20 = zeros
  m21 = zeros
  m22 = ones
  rz = tf.stack((m00, m01, m02,
                 m10, m11, m12,
                 m20, m21, m22),
                axis=-1)
  rz = tf.reshape(rz, output_shape)

  matrix = tf.eye(output_shape[-2], output_shape[-1],
                  batch_shape=output_shape[:-2])

  for r in order.upper():
    if r == 'X':
      matrix = rx @ matrix
    elif r == 'Y':
      matrix = ry @ matrix
    elif r == 'Z':
      matrix = rz @ matrix
    else:
      raise ValueError(f"Invalid value for `order`: {order}")

  return matrix
