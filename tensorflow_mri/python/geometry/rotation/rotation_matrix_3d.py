# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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

# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""3D rotation matrices."""

import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import rotation_matrix


def from_euler(angles):
  """Converts Euler angles to a 3D rotation matrix.

  Args:
    angles: A `tf.Tensor` of shape `[..., 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 3, 3]`.

  Raises:
    ValueError: If the shape of `angles` is invalid.
  """
  angles = tf.convert_to_tensor(angles)

  if angles.shape[-1] != 3:
    raise ValueError(
        f"angles must have shape `[..., 3]`, but got: {angles.shape}")

  sin_angles = tf.math.sin(angles)
  cos_angles = tf.math.cos(angles)
  return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def from_small_euler(angles):
  """Converts small Euler angles to a 3D rotation matrix.

  Args:
    angles: A `tf.Tensor` of shape `[..., 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 3, 3]`.

  Raises:
    ValueError: If the shape of `angles` is invalid.
  """
  angles = tf.convert_to_tensor(angles)

  if angles.shape[-1:] != 3:
    raise ValueError(
        f"angles must have shape `[..., 3]`, but got: {angles.shape}")

  sin_angles = angles
  cos_angles = 1.0 - 0.5 * tf.math.square(angles)
  return _build_matrix_from_sines_and_cosines(sin_angles, cos_angles)


def from_axis_angle(axis, angle):
  """Converts an axis-angle to a 3D rotation matrix."""
  axis = tf.convert_to_tensor(axis)
  angle = tf.convert_to_tensor(angle)

  if axis.shape[-1] != 3:
    raise ValueError(
        f"axis must have shape `[..., 3]`, but got: {axis.shape}")
  if angle.shape[-1:] != 1:
    raise ValueError(
        f"angle must have shape `[..., 1]`, but got: {angle.shape}")

  try:
    static_batch_shape = tf.broadcast_static_shape(
        axis.shape[:-1], angle.shape[:-1])
  except ValueError as err:
    raise ValueError(
        f"The batch shapes of axis and angle do not "
        f"broadcast: {axis.shape[:-1]} vs. {angle.shape[:-1]}") from err

  sin_axis = tf.sin(angle) * axis
  cos_angle = tf.cos(angle)
  cos1_axis = (1.0 - cos_angle) * axis
  _, axis_y, axis_z = tf.unstack(axis, axis=-1)
  cos1_axis_x, cos1_axis_y, _ = tf.unstack(cos1_axis, axis=-1)
  sin_axis_x, sin_axis_y, sin_axis_z = tf.unstack(sin_axis, axis=-1)
  tmp = cos1_axis_x * axis_y
  m01 = tmp - sin_axis_z
  m10 = tmp + sin_axis_z
  tmp = cos1_axis_x * axis_z
  m02 = tmp + sin_axis_y
  m20 = tmp - sin_axis_y
  tmp = cos1_axis_y * axis_z
  m12 = tmp - sin_axis_x
  m21 = tmp + sin_axis_x
  diag = cos1_axis * axis + cos_angle
  diag_x, diag_y, diag_z = tf.unstack(diag, axis=-1)
  matrix = tf.stack([diag_x, m01, m02,
                      m10, diag_y, m12,
                      m20, m21, diag_z], axis=-1)
  output_shape = tf.concat([tf.shape(axis)[:-1], [3, 3]], axis=-1)
  return tf.reshape(matrix, output_shape)


def from_quaternion(quaternion):
  """Converts a quaternion to a 3D rotation matrix."""
  quaternion = tf.convert_to_tensor(quaternion)

  if quaternion.shape[-1] != 4:
    raise ValueError(f"quaternion must have shape `[..., 4]`, ",
                     f"but got: {quaternion.shape}")

  x, y, z, w = tf.unstack(quaternion, axis=-1)
  tx = 2.0 * x
  ty = 2.0 * y
  tz = 2.0 * z
  twx = tx * w
  twy = ty * w
  twz = tz * w
  txx = tx * x
  txy = ty * x
  txz = tz * x
  tyy = ty * y
  tyz = tz * y
  tzz = tz * z
  matrix = tf.stack([1.0 - (tyy + tzz), txy - twz, txz + twy,
                     txy + twz, 1.0 - (txx + tzz), tyz - twx,
                     txz - twy, tyz + twx, 1.0 - (txx + tyy)], axis=-1)
  output_shape = tf.concat([tf.shape(quaternion)[:-1], [3, 3]], axis=-1)
  return tf.reshape(matrix, output_shape)


def _build_matrix_from_sines_and_cosines(sin_angles, cos_angles):
  """Builds a 3D rotation matrix from sines and cosines of Euler angles.

  Args:
    sin_angles: A tensor of shape `[..., 3]`, where the last dimension
      represents the sine of the Euler angles.
    cos_angles: A tensor of shape `[..., 3]`, where the last dimension
      represents the cosine of the Euler angles.

  Returns:
    A `tf.Tensor` of shape `[..., 3, 3]`, where the last two dimensions
    represent a 3D rotation matrix.
  """
  sin_angles.shape.assert_is_compatible_with(cos_angles.shape)

  sx, sy, sz = tf.unstack(sin_angles, axis=-1)
  cx, cy, cz = tf.unstack(cos_angles, axis=-1)
  m00 = cy * cz
  m01 = (sx * sy * cz) - (cx * sz)
  m02 = (cx * sy * cz) + (sx * sz)
  m10 = cy * sz
  m11 = (sx * sy * sz) + (cx * cz)
  m12 = (cx * sy * sz) - (sx * cz)
  m20 = -sy
  m21 = sx * cy
  m22 = cx * cy
  matrix = tf.stack([m00, m01, m02,
                     m10, m11, m12,
                     m20, m21, m22],
                    axis=-1)
  output_shape = tf.concat([tf.shape(sin_angles)[:-1], [3, 3]], axis=-1)
  return tf.reshape(matrix, output_shape)


def inverse(matrix):
  """Inverts a 3D rotation matrix.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 3, 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 3, 3]`.

  Raises:
    ValueError: If the shape of `matrix` is invalid.
  """
  return rotation_matrix.inverse(3, matrix)


def is_valid(matrix, atol=1e-3):
  """Checks if a 3D rotation matrix is valid.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 3, 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 1]` indicating whether the matrix is valid.
  """
  return rotation_matrix.is_valid(3, matrix, atol=atol)


def rotate(point, matrix):
  """Rotates a 3D point using rotation matrix.

  Args:
    point: A `tf.Tensor` of shape `[..., 3]`.
    matrix: A `tf.Tensor` of shape `[..., 3, 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 3]`.

  Raises:
    ValueError: If the shape of `point` or `matrix` is invalid.
  """
  return rotation_matrix.rotate(3, point, matrix)


def check_shape(matrix):
  """Checks the shape of `point` and `matrix`.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 3, 3]`.

  Raises:
    ValueError: If the shape of `matrix` is invalid.
  """
  rotation_matrix.check_shape(3, matrix)
