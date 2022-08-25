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

import tensorflow as tf

from tensorflow_mri.python.util import api_util


@api_util.export("geometry.euler_to_rotation_matrix_2d")
def from_euler(angle, name=None):
  r"""Converts an angle to a 2D rotation matrix.

  Converts an angle $$\theta$$ to a 2D rotation matrix following the equation

  $$
    \mathbf{R} =
    \begin{bmatrix}
    \cos(\theta) & -\sin(\theta) \\
    \sin(\theta) & \cos(\theta)
    \end{bmatrix}.
  $$

  Note:
    The resulting matrix rotates points in the $$xy$$-plane counterclockwise.

  Args:
    angle: A tensor of shape `[..., 1]`, where the last dimension
      represents an angle in radians.
    name: A name for this op.

  Returns:
    A tensor of shape `[..., 2, 2]`, where the last dimension represents
    a 2D rotation matrix.

  Raises:
    ValueError: If the shape of `angle` is invalid.

  References:
    This operator is based on
    `tfg.geometry.transformation.rotation_matrix_2d.from_euler`.
  """
  with tf.name_scope(name or "euler_to_rotation_matrix_2d"):
    angle = tf.convert_to_tensor(angle)

    if not angle.shape[-1:].is_compatible_with([1]):
      raise ValueError(
          f"angle must have shape `[..., 1]`, but got: {angle.shape}")

    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)
    matrix = tf.stack((cos_angle, -sin_angle,
                       sin_angle, cos_angle),
                      axis=-1)
    output_shape = tf.concat((tf.shape(input=angle)[:-1], (2, 2)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


@api_util.export("geometry.rotate_with_rotation_matrix_2d")
def rotate(point, matrix, name=None):
  """Rotates a 2D point using a 2D rotation matrix.

  Args:
    point: A tensor of shape `[..., 2]`, where the last dimension
      represents a 2D point and `...` represents any number of batch dimensions.
    matrix: A tensor of shape `[..., 2, 2]`, where the last two
      dimensions represent a 2D rotation matrix and `...` represents any
      number of batch dimensions, which must be broadcastable with those in
      shape.
    name: A name for this op.

  Returns:
    A tensor of shape `[..., 2]`, where the last dimension represents a 2D
    point and `...` is the result of broadcasting the batch shapes of `point`
    and `matrix`.

  Raises:
    ValueError: If the shape of `point` or `matrix` is not supported.

  References:
    This operator is based on
    `tfg.geometry.transformation.rotation_matrix_2d.rotate`.
  """
  with tf.name_scope(name or "rotate_with_rotation_matrix_2d"):
    point = tf.convert_to_tensor(point)
    matrix = tf.convert_to_tensor(matrix)

    if not point.shape[-1:].is_compatible_with(2):
      raise ValueError(
          f"point must have shape [..., 2], but got: {point.shape}")
    if (not matrix.shape[-1:].is_compatible_with([2]) or
        not matrix.shape[-2:-1].is_compatible_with([2])):
      raise ValueError(
          f"matrix must have shape [..., 2, 2], but got: {matrix.shape}")
    try:
      static_batch_shape = tf.broadcast_static_shape(point.shape[:-1],
                                                     matrix.shape[:-2])
    except ValueError as err:
      raise ValueError(
          f"The batch shapes of point and matrix could not be broadcasted. "
          f"Received: {point.shape} and {matrix.shape}") from err

    common_batch_shape = tf.broadcast_dynamic_shape(tf.shape(point)[:-1],
                                                    tf.shape(matrix)[:-2])

    point = tf.broadcast_to(point, tf.concat([common_batch_shape, [2]], 0))
    matrix = tf.broadcast_to(matrix, tf.concat([common_batch_shape, [2, 2]], 0))
    rotated_point = tf.linalg.matvec(matrix, point)
    return tf.ensure_shape(rotated_point, static_batch_shape.concatenate([2]))
