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

import contextlib

import tensorflow as tf


class RotationMatrix(tf.experimental.ExtensionType):
  """Represents a {n}D rotation matrix.

  References:
    1. https://en.wikipedia.org/wiki/Rotation_matrix
    2. https://www.tensorflow.org/graphics/api_docs/python/tfg/geometry/transformation/rotation_matrix_2d
  """
  matrix: tf.Tensor
  name: str = "rotation_matrix"

  def __init__(self, matrix, name=None):
    self.matrix = matrix
    self.name = name or self._default_name()

  def __validate__(self):
    self._validate_shape()

  def __eq__(self, other):
    return tf.math.equal(self.matrix, other.matrix)

  def inverse(self, name=None):
    r"""Computes the inverse of this rotation matrix.

    Args:
      name: A name for this op. Defaults to `"inverse"`.

    Returns:
      A `RotationMatrix{n}D` representing the inverse of this rotation matrix.
    """
    with self._name_scope(name or "inverse"):
      return type(self)(tf.linalg.matrix_transpose(self.matrix),
                        name=self.name + '_inverse')

  def is_valid(self, atol=1e-3, name=None):
    r"""Determines if this matrix is a valid rotation matrix.

    A matrix $\mathbf{{R}}$ is a valid rotation matrix if
    $\mathbf{{R}}^T\mathbf{{R}} = \mathbf{{I}}$ and $\det(\mathbf{{R}}) = 1$.

    Args:
      atol: The absolute tolerance parameter.
      name: A name for this op. Defaults to `"is_valid"`.

    Returns:
      A boolean `tf.Tensor` with shape `[..., 1]`, `True` if the corresponding
      matrix is valid and `False` otherwise.
    """
    with self._name_scope(name or "is_valid"):
      # Compute how far the determinant of the matrix is from 1.
      distance_determinant = tf.abs(tf.linalg.det(self.matrix) - 1.)

      # Computes how far the product of the transposed rotation matrix with itself
      # is from the identity matrix.
      identity = tf.eye(tf.shape(self.matrix)[-1], dtype=self.dtype)
      inverse = tf.linalg.matrix_transpose(self.matrix)
      distance_identity = tf.matmul(inverse, self.matrix) - identity
      distance_identity = tf.norm(distance_identity, axis=[-2, -1])

      # Computes the mask of entries that satisfies all conditions.
      mask = tf.math.logical_and(distance_determinant < atol,
                                distance_identity < atol)
      return tf.expand_dims(mask, axis=-1)

  def rotate(self, point, name=None):
    r"""Rotates a {n}D point as described by this rotation matrix.

    Args:
      point: A `tf.Tensor` of shape `[..., {n}]`, where the last dimension
        represents a {n}D point and `...` represents any number of batch
        dimensions, which must be broadcastable with the batch shape of the
        rotation matrix.
      name: A name for this op. Defaults to `"rotate"`.

    Returns:
      A `tf.Tensor` of shape `[..., {n}]`, where the last dimension represents
      a {n}D point and `...` is the result of broadcasting the batch shapes of
      `point` and this rotation matrix.

    Raises:
      ValueError: If the shape of `point` is invalid.
    """
    with self._name_scope(name or "rotate"):
      point = tf.convert_to_tensor(point)

      if not point.shape[-1:].is_compatible_with(2):
        raise ValueError(
            f"point must have shape [..., 2], but got: {point.shape}")
      try:
        static_batch_shape = tf.broadcast_static_shape(
            point.shape[:-1], self.shape[:-2])
      except ValueError as err:
        raise ValueError(
            f"The batch shapes of point and this rotation matrix do not "
            f"broadcast: {point.shape[:-1]} vs. {self.shape[:-2]}") from err

      common_batch_shape = tf.broadcast_dynamic_shape(
          tf.shape(point)[:-1], tf.shape(self.matrix)[:-2])
      point = tf.broadcast_to(point, tf.concat(
          [common_batch_shape, [self._n()]], 0))
      matrix = tf.broadcast_to(self.matrix, tf.concat(
          [common_batch_shape, [self._n(), self._n()]], 0))

      rotated_point = tf.linalg.matvec(matrix, point)

      output_shape = static_batch_shape.concatenate([self._n()])
      return tf.ensure_shape(rotated_point, output_shape)

  @property
  def shape(self):
    """Returns the shape of this rotation matrix."""
    return self.matrix.shape

  @property
  def dtype(self):
    """Returns the dtype of this rotation matrix."""
    return self.matrix.dtype

  @contextlib.contextmanager
  def _name_scope(self, name=None):
    """Helper function to standardize op scope."""
    with tf.name_scope(self.name):
      with tf.name_scope(name) as scope:
        yield scope

  def _default_name(self):
    return {2: 'rotation_matrix_2d', 3: 'rotation_matrix_3d'}[self._n()]

  def _validate_shape(self):
    if self.matrix.shape.rank is not None:
      if self.matrix.shape.rank < 2:
        raise ValueError(
            f"matrix must have rank >= 2, but got: {self.matrix.shape}")
      if not self.matrix.shape[-2:].is_compatible_with([self._n(), self._n()]):
        raise ValueError(
            f"matrix must have shape [..., {self._n()}, {self._n()}], "
            f"but got: {self.matrix.shape}")

  def _n(self):
    return {'RotationMatrix2D': 2, 'RotationMatrix3D': 3}[type(self).__name__]


@tf.experimental.dispatch_for_api(tf.shape, {'input': RotationMatrix})
def rotation_matrix_shape(input, out_type=tf.int32, name=None):
  return tf.shape(input.matrix, out_type=out_type, name=name)
