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
"""2D rotation."""

import contextlib

import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import rotation_matrix_2d
from tensorflow_mri.python.util import api_util


@api_util.export("geometry.Rotation2D")
class Rotation2D(tf.experimental.ExtensionType):
  """Represents a 2D rotation (or a batch thereof)."""
  _matrix: tf.Tensor
  _name: str = "rotation_2d"

  @classmethod
  def from_matrix(cls, matrix, name=None):
    r"""Creates a 2D rotation from a rotation matrix.

    Args:
      matrix: A `tf.Tensor` of shape `[..., 2, 2]`, where `...` represents
        any number of batch dimensions.
      name: A name for this op. Defaults to `"rotation_2d/from_matrix"`.

    Returns:
      A `Rotation2D`.
    """
    name = name or "rotation_2d/from_matrix"
    with tf.name_scope(name):
      return cls(_matrix=matrix, _name=name)

  @classmethod
  def from_euler(cls, angle, name=None):
    r"""Creates a 2D rotation from an angle.

    The resulting rotation acts like the following rotation matrix:

    $$
      \mathbf{R} =
      \begin{bmatrix}
      \cos(\theta) & -\sin(\theta) \\
      \sin(\theta) & \cos(\theta)
      \end{bmatrix}.
    $$

    ```{note}
    The resulting rotation rotates points in the $xy$-plane counterclockwise.
    ```

    Args:
      angle: A `tf.Tensor` of shape `[..., 1]`, where the last dimension
        represents an angle in radians.
      name: A name for this op. Defaults to `"rotation_2d/from_euler"`.

    Returns:
      A `Rotation2D`.

    Raises:
      ValueError: If the shape of `angle` is invalid.
    """
    name = name or "rotation_2d/from_euler"
    with tf.name_scope(name):
      return cls(_matrix=rotation_matrix_2d.from_euler(angle), _name=name)

  @classmethod
  def from_small_euler(cls, angle, name=None):
    r"""Creates a 2D rotation from a small angle.

    Uses the small angle approximation to compute the rotation. Under the
    small angle assumption, $\sin(x)$$ and $$\cos(x)$ can be approximated by
    their second order Taylor expansions, where $\sin(x) \approx x$ and
    $\cos(x) \approx 1 - \frac{x^2}{2}$.

    The resulting rotation acts like the following rotation matrix:

    $$
      \mathbf{R} =
      \begin{bmatrix}
      1.0 - 0.5\theta^2 & -\theta \\
      \theta & 1.0 - 0.5\theta^2
      \end{bmatrix}.
    $$

    ```{note}
    The resulting rotation rotates points in the $xy$-plane counterclockwise.
    ```

    ```{note}
    This function does not verify the smallness of the angles.
    ```

    Args:
      angle: A `tf.Tensor` of shape `[..., 1]`, where the last dimension
        represents an angle in radians.
      name: A name for this op. Defaults to "rotation_2d/from_small_euler".

    Returns:
      A `Rotation2D`.

    Raises:
      ValueError: If the shape of `angle` is invalid.
    """
    name = name or "rotation_2d/from_small_euler"
    with tf.name_scope(name):
      return cls(_matrix=rotation_matrix_2d.from_small_euler(angle), _name=name)

  def as_matrix(self, name=None):
    r"""Returns the rotation matrix that represents this rotation.

    Args:
      name: A name for this op. Defaults to `"as_matrix"`.

    Returns:
      A `tf.Tensor` of shape `[..., 2, 2]`.
    """
    with self._name_scope(name or "as_matrix"):
      return tf.identity(self._matrix)

  def inverse(self, name=None):
    r"""Computes the inverse of this rotation.

    Args:
      name: A name for this op. Defaults to `"inverse"`.

    Returns:
      A `Rotation2D` representing the inverse of this rotation.
    """
    with self._name_scope(name or "inverse"):
      return Rotation2D(_matrix=rotation_matrix_2d.inverse(self._matrix),
                        _name=self._name + "/inverse")

  def is_valid(self, atol=1e-3, name=None):
    r"""Determines if this is a valid rotation.

    A rotation matrix $\mathbf{R}$ is a valid rotation matrix if
    $\mathbf{R}^T\mathbf{R} = \mathbf{I}$ and $\det(\mathbf{R}) = 1$.

    Args:
      atol: A `float`. The absolute tolerance parameter.
      name: A name for this op. Defaults to `"is_valid"`.

    Returns:
      A boolean `tf.Tensor` with shape `[..., 1]`, `True` if the corresponding
      matrix is valid and `False` otherwise.
    """
    with self._name_scope(name or "is_valid"):
      return rotation_matrix_2d.is_valid(self._matrix, atol=atol)

  def rotate(self, point, name=None):
    r"""Rotates a 2D point.

    Args:
      point: A `tf.Tensor` of shape `[..., 2]`, where the last dimension
        represents a 2D point and `...` represents any number of batch
        dimensions, which must be broadcastable with the batch shape of this
        rotation.
      name: A name for this op. Defaults to `"rotate"`.

    Returns:
      A `tf.Tensor` of shape `[..., 2]`, where the last dimension represents
      a 2D point and `...` is the result of broadcasting the batch shapes of
      `point` and this rotation matrix.

    Raises:
      ValueError: If the shape of `point` is invalid.
    """
    with self._name_scope(name or "rotate"):
      return rotation_matrix_2d.rotate(point, self._matrix)

  def __eq__(self, other):
    """Returns true if this rotation is equivalent to the other rotation."""
    return tf.math.reduce_all(
        tf.math.equal(self._matrix, other._matrix), axis=[-2, -1])

  def __matmul__(self, other):
    """Composes this rotation with another rotation."""
    return Rotation2D(_matrix=self._matrix @ other._matrix,)

  def __validate__(self):
    """Checks that this rotation is a valid rotation.

    Only performs static checks.
    """
    rotation_matrix_2d.check_shape(self._matrix)

  @property
  def shape(self):
    """Returns the shape of this rotation."""
    return self._matrix.shape[:-2]

  @property
  def dtype(self):
    """Returns the dtype of this rotation."""
    return self._matrix.dtype

  @property
  def name(self):
    """Returns the name of this rotation."""
    return self._name

  @contextlib.contextmanager
  def _name_scope(self, name=None):
    """Helper function to standardize op scope."""
    with tf.name_scope(self.name):
      with tf.name_scope(name) as scope:
        yield scope


@tf.experimental.dispatch_for_api(tf.shape, {'input': Rotation2D})
def rotation_2d_shape(input, out_type=tf.int32, name=None):
  return tf.shape(input._matrix, out_type=out_type, name=name)[:-2]
