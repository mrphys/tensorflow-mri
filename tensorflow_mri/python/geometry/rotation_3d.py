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
"""3D rotation."""

import contextlib

import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import rotation_matrix_3d
from tensorflow_mri.python.util import api_util


@api_util.export("geometry.Rotation3D")
class Rotation3D(tf.experimental.ExtensionType):
  """Represents a 3D rotation (or a batch thereof)."""
  _matrix: tf.Tensor
  _name: str = "rotation_3d"

  @classmethod
  def from_matrix(cls, matrix, name=None):
    r"""Creates a 3D rotation from a rotation matrix.

    Args:
      matrix: A `tf.Tensor` of shape `[..., 3, 3]`, where the last two
        dimensions represent a rotation matrix.
      name: A name for this op. Defaults to `"rotation_3d/from_matrix"`.

    Returns:
      A `Rotation3D`.
    """
    name = name or "rotation_3d/from_matrix"
    with tf.name_scope(name):
      return cls(_matrix=matrix, _name=name)

  @classmethod
  def from_euler(cls, angles, name=None):
    r"""Creates a 3D rotation from Euler angles.

    The resulting rotation acts like the rotation matrix
    $\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$.

    ```{note}
    Uses the $z$-$y$-$x$ rotation convention (Tait-Bryan angles).
    ```

    Args:
      angles: A `tf.Tensor` of shape `[..., 3]`, where the last dimension
        represents the three Euler angles in radians. `angles[..., 0]`
        is the angles about `x`, `angles[..., 1]` is the angles about `y`,
        and `angles[..., 2]` is the angles about `z`.
      name: A name for this op. Defaults to `"rotation_3d/from_euler"`.

    Returns:
      A `Rotation3D`.

    Raises:
      ValueError: If the shape of `angles` is invalid.
    """
    name = name or "rotation_3d/from_matrix"
    with tf.name_scope(name):
      return cls(_matrix=rotation_matrix_3d.from_euler(angles), _name=name)

  @classmethod
  def from_small_euler(cls, angles, name=None):
    r"""Creates a 3D rotation from small Euler angles.

    The resulting rotation acts like the rotation matrix
    $\mathbf{R} = \mathbf{R}_z\mathbf{R}_y\mathbf{R}_x$.

    Uses the small angle approximation to compute the rotation. Under the
    small angle assumption, $\sin(x)$$ and $$\cos(x)$ can be approximated by
    their second order Taylor expansions, where $\sin(x) \approx x$ and
    $\cos(x) \approx 1 - \frac{x^2}{2}$.

    ```{note}
    Uses the $z$-$y$-$x$ rotation convention (Tait-Bryan angles).
    ```

    ```{note}
    This function does not verify the smallness of the angles.
    ```

    Args:
      angles: A `tf.Tensor` of shape `[..., 3]`, where the last dimension
        represents the three Euler angles in radians. `angles[..., 0]`
        is the angles about `x`, `angles[..., 1]` is the angles about `y`,
        and `angles[..., 2]` is the angles about `z`.
      name: A name for this op. Defaults to "rotation_3d/from_small_euler".

    Returns:
      A `Rotation3D`.

    Raises:
      ValueError: If the shape of `angles` is invalid.
    """
    name = name or "rotation_3d/from_small_euler"
    with tf.name_scope(name):
      return cls(_matrix=rotation_matrix_3d.from_small_euler(angles),
                 _name=name)

  @classmethod
  def from_axis_angle(cls, axis, angle, name=None):
    """Creates a 3D rotation from an axis-angle representation.

    Args:
      axis: A `tf.Tensor` of shape `[..., 3]`, where the last dimension
        represents a normalized axis.
      angle: A `tf.Tensor` of shape `[..., 1]`, where the last dimension
        represents a normalized axis.
      name: A name for this op. Defaults to "rotation_3d/from_axis_angle".

    Returns:
      A `Rotation3D`.

    Raises:
      ValueError: If the shape of `axis` or `angle` is invalid.
    """
    name = name or "rotation_3d/from_axis_angle"
    with tf.name_scope(name):
      return cls(_matrix=rotation_matrix_3d.from_axis_angle(axis, angle),
                 _name=name)

  @classmethod
  def from_quaternion(cls, quaternion, name=None):
    """Creates a 3D rotation from a quaternion.

    Args:
      quaternion: A `tf.Tensor` of shape `[..., 4]`, where the last dimension
        represents a normalized quaternion.
      name: A name for this op. Defaults to `"rotation_3d/from_quaternion"`.

    Returns:
      A `Rotation3D`.

    Raises:
      ValueError: If the shape of `quaternion` is invalid.
    """
    name = name or "rotation_3d/from_quaternion"
    with tf.name_scope(name):
      return cls(_matrix=rotation_matrix_3d.from_quaternion(quaternion),
                 _name=name)

  def as_matrix(self, name=None):
    r"""Returns the rotation matrix that represents this rotation.

    Args:
      name: A name for this op. Defaults to `"as_matrix"`.

    Returns:
      A `tf.Tensor` of shape `[..., 3, 3]`.
    """
    with self._name_scope(name or "as_matrix"):
      return tf.identity(self._matrix)

  def inverse(self, name=None):
    r"""Computes the inverse of this rotation.

    Args:
      name: A name for this op. Defaults to `"inverse"`.

    Returns:
      A `Rotation3D` representing the inverse of this rotation.
    """
    with self._name_scope(name or "inverse"):
      return Rotation3D(_matrix=rotation_matrix_3d.inverse(self._matrix),
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
      return rotation_matrix_3d.is_valid(self._matrix, atol=atol)

  def rotate(self, point, name=None):
    r"""Rotates a 3D point.

    Args:
      point: A `tf.Tensor` of shape `[..., 3]`, where the last dimension
        represents a 3D point and `...` represents any number of batch
        dimensions, which must be broadcastable with the batch shape of this
        rotation.
      name: A name for this op. Defaults to `"rotate"`.

    Returns:
      A `tf.Tensor` of shape `[..., 3]`, where the last dimension represents
      a 3D point and `...` is the result of broadcasting the batch shapes of
      `point` and this rotation matrix.

    Raises:
      ValueError: If the shape of `point` is invalid.
    """
    with self._name_scope(name or "rotate"):
      return rotation_matrix_3d.rotate(point, self._matrix)

  def __eq__(self, other):
    """Returns true if this rotation is equivalent to the other rotation."""
    return tf.math.reduce_all(
        tf.math.equal(self._matrix, other._matrix), axis=[-2, -1])

  def __matmul__(self, other):
    """Composes this rotation with another rotation."""
    return Rotation3D(_matrix=self._matrix @ other._matrix,
                      _name=self._name + "_o_" + other._name)

  def __validate__(self):
    """Checks that this rotation is a valid rotation.

    Only performs static checks.
    """
    rotation_matrix_3d.check_shape(self._matrix)

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
