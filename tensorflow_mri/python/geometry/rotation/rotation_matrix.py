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
"""Rotation matrices."""

import tensorflow as tf


def rotate(n, point, matrix):
  """Rotates an N-D point using rotation matrix.

  Args:
    n: An `int`. The dimension of the point and matrix.
    point: A `tf.Tensor` of shape `[..., N]`.
    matrix: A `tf.Tensor` of shape `[..., N, N]`.

  Returns:
    A `tf.Tensor` of shape `[..., N]`.

  Raises:
    ValueError: If the shape of the point or matrix is invalid.
  """
  point = tf.convert_to_tensor(point)
  matrix = tf.convert_to_tensor(matrix)

  if point.shape[-1] != n:
    raise ValueError(
        f"point must have shape [..., {n}], but got: {point.shape}")
  if matrix.shape[-1] != n or matrix.shape[-2] != n:
    raise ValueError(
        f"matrix must have shape [..., {n}, {n}], but got: {matrix.shape}")
  try:
    static_batch_shape = tf.broadcast_static_shape(
        point.shape[:-1], matrix.shape[:-2])
  except ValueError as err:
    raise ValueError(
        f"The batch shapes of point and this rotation matrix do not "
        f"broadcast: {point.shape[:-1]} vs. {matrix.shape[:-2]}") from err

  common_batch_shape = tf.broadcast_dynamic_shape(
      tf.shape(point)[:-1], tf.shape(matrix)[:-2])
  point = tf.broadcast_to(point, tf.concat(
      [common_batch_shape, [n]], 0))
  matrix = tf.broadcast_to(matrix, tf.concat(
      [common_batch_shape, [n, n]], 0))

  rotated_point = tf.linalg.matvec(matrix, point)
  output_shape = static_batch_shape.concatenate([n])
  return tf.ensure_shape(rotated_point, output_shape)


def inverse(n, matrix):
  """Inverts an N-D rotation matrix.

  Args:
    n: An `int`. The dimension of the matrix.
    matrix: A `tf.Tensor` of shape `[..., N, N]`.

  Returns:
    A `tf.Tensor` of shape `[..., N, N]`.

  Raises:
    ValueError: If the shape of the matrix is invalid.
  """
  matrix = tf.convert_to_tensor(matrix)

  if matrix.shape[-1] != n or matrix.shape[-2] != n:
    raise ValueError(
        f"matrix must have shape [..., {n}, {n}], but got: {matrix.shape}")

  return tf.linalg.matrix_transpose(matrix)


def is_valid(n, matrix, atol=1e-3):
  """Checks if an N-D rotation matrix is valid.

  Args:
    n: An `int`. The dimension of the matrix.
    matrix: A `tf.Tensor` of shape `[..., N, N]`.
    atol: A `float`. The absolute tolerance for checking if the matrix is valid.

  Returns:
    A boolean `tf.Tensor` of shape `[..., 1]`.

  Raises:
    ValueError: If the shape of the matrix is invalid.
  """
  matrix = tf.convert_to_tensor(matrix)

  if matrix.shape[-1] != n or matrix.shape[-2] != n:
    raise ValueError(
        f"matrix must have shape [..., {n}, {n}], but got: {matrix.shape}")

  # Compute how far the determinant of the matrix is from 1.
  distance_determinant = tf.abs(tf.linalg.det(matrix) - 1.)

  # Computes how far the product of the transposed rotation matrix with itself
  # is from the identity matrix.
  identity = tf.eye(n, dtype=matrix.dtype)
  inverse_matrix = tf.linalg.matrix_transpose(matrix)
  distance_identity = tf.matmul(inverse_matrix, matrix) - identity
  distance_identity = tf.norm(distance_identity, axis=[-2, -1])

  # Computes the mask of entries that satisfies all conditions.
  mask = tf.math.logical_and(distance_determinant < atol,
                             distance_identity < atol)
  return tf.expand_dims(mask, axis=-1)


def check_shape(n, matrix):
  matrix = tf.convert_to_tensor(matrix)
  if matrix.shape.rank is not None and matrix.shape.rank < 2:
    raise ValueError(
        f"matrix must have rank >= 2, but got: {matrix.shape}")
  if matrix.shape[-2] != n or matrix.shape[-1] != n:
    raise ValueError(
        f"matrix must have shape [..., {n}, {n}], "
        f"but got: {matrix.shape}")
