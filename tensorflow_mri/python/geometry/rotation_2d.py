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

import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import rotation_matrix_2d
from tensorflow_mri.python.util import api_util


@api_util.export("geometry.Rotation2D")
class Rotation2D(tf.experimental.BatchableExtensionType):
  """Represents a rotation in 2D space (or a batch thereof).

  You can initialize a `Rotation2D` object using one of the `from_*` class
  methods:

  - `from_matrix`, to initialize using a
    [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix).
  - `from_euler`, to initialize using an angle (in radians).
  - `from_small_euler`, to initialize using an angle which is small enough
    to fall under the [small angle approximation](https://en.wikipedia.org/wiki/Small-angle_approximation).

  In all cases the above methods can accept a batch, in which case the returned
  `Rotation2D` object will also have a batch shape.

  Once initialized, `Rotation2D` objects expose several methods to operate
  easily. These can all be used in the same way regardless of how the
  `Rotation2D` was originally initialized.

  - `rotate` rotates a point or a batch of points. The batch shapes of the
    point and this rotation will be broadcasted.
  - `inverse` returns a new `Rotation2D` object representing the inverse of
    the current rotation.
  - `is_valid` can be used to check if the rotation is valid.

  Finally, the `as_*` methods can be used to obtain an explicit representation
  of this rotation.

  - `as_matrix` returns the corresponding rotation matrix.

  Example:

    >>> # Initialize a rotation object using a rotation matrix.
    >>> rot = tfmri.geometry.Rotation2D.from_matrix([[0.0, -1.0], [1.0, 0.0]])
    >>> print(rot)
    tfmri.geometry.Rotation2D(shape=(), dtype=float32)
    >>> # Rotate a point.
    >>> point = tf.constant([1.0, 0.0], dtype=tf.float32)
    >>> rotated = rot.rotate(point)
    >>> print(rotated)
    tf.Tensor([0. 1.], shape=(2,), dtype=float32)
    >>> # Rotate the point back using the inverse rotation.
    >>> inv_rot = rot.inverse()
    >>> restored = inv_rot.rotate(rotated)
    >>> print(restored)
    tf.Tensor([1. 0.], shape=(2,), dtype=float32)
    >>> # Get the rotation matrix for the inverse rotation.
    >>> print(inv_rot.as_matrix())
    tf.Tensor(
    [[ 0.  1.]
      [-1.  0.]], shape=(2, 2), dtype=float32)
    >>> # You can also initialize a rotation using an angle:
    >>> rot2 = tfmri.geometry.Rotation2D.from_euler([np.pi / 2])
    >>> rotated2 = rot.rotate(point)
    >>> np.allclose(rotated2, rotated)
    True

  """
  __name__ = "tfmri.geometry.Rotation2D"
  _matrix: tf.Tensor

  @classmethod
  def from_matrix(cls, matrix, name=None):
    r"""Creates a 2D rotation from a rotation matrix.

    Args:
      matrix: A `tf.Tensor` of shape `[..., 2, 2]`, where the last two
        dimensions represent a rotation matrix.
      name: A name for this op. Defaults to `"rotation_2d/from_matrix"`.

    Returns:
      A `Rotation2D`.
    """
    with tf.name_scope(name or "rotation_2d/from_matrix"):
      return cls(_matrix=matrix)

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
    with tf.name_scope(name or "rotation_2d/from_euler"):
      return cls(_matrix=rotation_matrix_2d.from_euler(angle))

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
    with tf.name_scope("rotation_2d/from_small_euler"):
      return cls(_matrix=rotation_matrix_2d.from_small_euler(angle))

  def as_matrix(self, name=None):
    r"""Returns the rotation matrix that represents this rotation.

    Args:
      name: A name for this op. Defaults to `"rotation_2d/as_matrix"`.

    Returns:
      A `tf.Tensor` of shape `[..., 2, 2]`.
    """
    with tf.name_scope(name or "rotation_2d/as_matrix"):
      return tf.identity(self._matrix)

  def inverse(self, name=None):
    r"""Computes the inverse of this rotation.

    Args:
      name: A name for this op. Defaults to `"rotation_2d/inverse"`.

    Returns:
      A `Rotation2D` representing the inverse of this rotation.
    """
    with tf.name_scope(name or "rotation_2d/inverse"):
      return Rotation2D(_matrix=rotation_matrix_2d.inverse(self._matrix))

  def is_valid(self, atol=1e-3, name=None):
    r"""Determines if this is a valid rotation.

    A rotation matrix $\mathbf{R}$ is a valid rotation matrix if
    $\mathbf{R}^T\mathbf{R} = \mathbf{I}$ and $\det(\mathbf{R}) = 1$.

    Args:
      atol: A `float`. The absolute tolerance parameter.
      name: A name for this op. Defaults to `"rotation_2d/is_valid"`.

    Returns:
      A boolean `tf.Tensor` with shape `[..., 1]`, `True` if the corresponding
      matrix is valid and `False` otherwise.
    """
    with tf.name_scope(name or "rotation_2d/is_valid"):
      return rotation_matrix_2d.is_valid(self._matrix, atol=atol)

  def rotate(self, point, name=None):
    r"""Rotates a 2D point.

    Args:
      point: A `tf.Tensor` of shape `[..., 2]`, where the last dimension
        represents a 2D point and `...` represents any number of batch
        dimensions, which must be broadcastable with the batch shape of this
        rotation.
      name: A name for this op. Defaults to `"rotation_2d/rotate"`.

    Returns:
      A `tf.Tensor` of shape `[..., 2]`, where the last dimension represents
      a 2D point and `...` is the result of broadcasting the batch shapes of
      `point` and this rotation matrix.

    Raises:
      ValueError: If the shape of `point` is invalid.
    """
    with tf.name_scope(name or "rotation_2d/rotate"):
      return rotation_matrix_2d.rotate(point, self._matrix)

  def __eq__(self, other):
    """Returns true if this rotation is equivalent to the other rotation."""
    return tf.math.reduce_all(
        tf.math.equal(self._matrix, other._matrix), axis=[-2, -1])

  def __matmul__(self, other):
    """Composes this rotation with another rotation."""
    return Rotation2D(_matrix=self._matrix @ other._matrix)

  def __repr__(self):
    """Returns a string representation of this rotation."""
    name = self.__name__
    return f"<{name}(shape={str(self.shape)}, dtype={self.dtype.name})>"

  def __str__(self):
    """Returns a string representation of this rotation."""
    return self.__repr__()[1:-1]

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


@tf.experimental.dispatch_for_api(tf.shape, {'input': Rotation2D})
def rotation_2d_shape(input, out_type=tf.int32, name=None):
  return tf.shape(input._matrix, out_type=out_type, name=name)[:-2]
