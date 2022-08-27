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
"""2D rotation matrices."""

import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import rotation_matrix


def from_euler(angle):
  """Converts an angle to a 2D rotation matrix.

  Args:
    angle: A `tf.Tensor` of shape `[..., 1]`.

  Returns:
    A `tf.Tensor` of shape `[..., 2, 2]`.

  Raises:
    ValueError: If the shape of `angle` is invalid.
  """
  angle = tf.convert_to_tensor(angle)

  if angle.shape[-1] != 1:
    raise ValueError(
        f"angle must have shape `[..., 1]`, but got: {angle.shape}")

  cos_angle = tf.math.cos(angle)
  sin_angle = tf.math.sin(angle)
  matrix = tf.stack([cos_angle, -sin_angle, sin_angle, cos_angle], axis=-1)
  output_shape = tf.concat([tf.shape(angle)[:-1], [2, 2]], axis=-1)
  return tf.reshape(matrix, output_shape)


def from_small_euler(angle):
  """Converts a small angle to a 2D rotation matrix.

  Args:
    angle: A `tf.Tensor` of shape `[..., 1]`.

  Returns:
    A `tf.Tensor` of shape `[..., 2, 2]`.

  Raises:
    ValueError: If the shape of `angle` is invalid.
  """
  angle = tf.convert_to_tensor(angle)

  if angle.shape[-1] != 1:
    raise ValueError(
        f"angle must have shape `[..., 1]`, but got: {angle.shape}")

  cos_angle = 1.0 - 0.5 * angle * angle
  sin_angle = angle
  matrix = tf.stack([cos_angle, -sin_angle, sin_angle, cos_angle], axis=-1)
  output_shape = tf.concat([tf.shape(angle)[:-1], [2, 2]], axis=-1)
  return tf.reshape(matrix, output_shape)


def inverse(matrix):
  """Inverts a 2D rotation matrix.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 2, 2]`.

  Returns:
    A `tf.Tensor` of shape `[..., 2, 2]`.

  Raises:
    ValueError: If the shape of `matrix` is invalid.
  """
  return rotation_matrix.inverse(2, matrix)


def is_valid(matrix, atol=1e-3):
  """Checks if a 2D rotation matrix is valid.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 2, 2]`.

  Returns:
    A `tf.Tensor` of shape `[..., 1]` indicating whether the matrix is valid.
  """
  return rotation_matrix.is_valid(2, matrix, atol=atol)


def rotate(point, matrix):
  """Rotates a 2D point using rotation matrix.

  Args:
    point: A `tf.Tensor` of shape `[..., 2]`.
    matrix: A `tf.Tensor` of shape `[..., 2, 2]`.

  Returns:
    A `tf.Tensor` of shape `[..., 2]`.

  Raises:
    ValueError: If the shape of `point` or `matrix` is invalid.
  """
  return rotation_matrix.rotate(2, point, matrix)


def check_shape(matrix):
  """Checks the shape of `point` and `matrix`.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 2, 2]`.

  Raises:
    ValueError: If the shape of `matrix` is invalid.
  """
  rotation_matrix.check_shape(2, matrix)
