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

from tensorflow_mri.python.geometry.rotation import euler_2d
from tensorflow_mri.python.geometry.rotation import rotation_matrix_2d
from tensorflow_mri.python.util import api_util


@api_util.export("geometry.Rotation2D")
class Rotation2D(tf.experimental.BatchableExtensionType):  # pylint: disable=abstract-method
  """Represents a rotation in 2D space (or a batch thereof).

  A `Rotation2D` contains all the information needed to represent a rotation
  in 2D space (or a multidimensional array of rotations) and provides
  convenient methods to work with rotations.

  ## Initialization

  You can initialize a `Rotation2D` object using one of the `from_*` class
  methods:

  - `from_matrix`, to initialize using a
    [rotation matrix](https://en.wikipedia.org/wiki/Rotation_matrix).
  - `from_euler`, to initialize using an angle (in radians).
  - `from_small_euler`, to initialize using an angle which is small enough
    to fall under the [small angle approximation](https://en.wikipedia.org/wiki/Small-angle_approximation).

  All of the above methods accept batched inputs, in which case the returned
  `Rotation2D` object will represent a batch of rotations.

  ## Methods

  Once initialized, `Rotation2D` objects expose several methods to operate
  easily with rotations. These methods are all used in the same way regardless
  of how the `Rotation2D` was originally initialized.

  - `rotate` rotates a point or a batch of points. The batch shapes of the
    point and this rotation will be broadcasted.
  - `inverse` returns a new `Rotation2D` object representing the inverse of
    the current rotation.
  - `is_valid` can be used to check if the rotation is valid.

  ## Conversion to other representations

  The `as_*` methods can be used to obtain an explicit representation
  of this rotation as a standard `tf.Tensor`.

  - `as_matrix` returns the corresponding rotation matrix.
  - `as_euler` returns the corresponding angle (in radians).

  ## Shape and dtype

  `Rotation2D` objects have a shape and a dtype, accessible via the `shape` and
  `dtype` properties. Because this operator acts like a rotation matrix, its
  shape corresponds to the shape of the rotation matrix. In other words,
  `rot.shape` is equal to `rot.as_matrix().shape`.

  ```{note}
    As with `tf.Tensor`s, the `shape` attribute contains the static shape
    as a `tf.TensorShape` and may not be fully defined outside eager execution.
    To obtain the dynamic shape of a `Rotation2D` object, use `tf.shape`.
  ```

  ## Operators

  `Rotation2D` objects also override a few operators for concise and intuitive
  use.

  - `==` (equality operator) can be used to check if two `Rotation2D` objects
    are equal. This checks if the rotations are equivalent, regardless of how
    they were defined (`rot1 == rot2`).
  - `@` (matrix multiplication operator) can be used to compose two rotations
    (`rot = rot1 @ rot2`).

  ## Compatibility with TensorFlow APIs

  Some TensorFlow APIs are explicitly overriden to operate with `Rotation2D`
  objects. These include:

  ```{list-table}
  ---
  header-rows: 1
  ---

  * - API
    - Description
    - Notes
  * - `tf.convert_to_tensor`
    - Converts a `Rotation2D` to a `tf.Tensor` containing the corresponding
      rotation matrix.
    - `tf.convert_to_tensor(rot)` is equivalent to `rot.as_matrix()`.
  * - `tf.linalg.matmul`
    - Composes two `Rotation2D` objects.
    - `tf.linalg.matmul(rot1, rot2)` is equivalent to `rot1 @ rot2`.
  * - `tf.linalg.matvec`
    - Rotates a point or a batch of points.
    - `tf.linalg.matvec(rot, point)` is equivalent to `rot.rotate(point)`.
  * - `tf.shape`
    - Returns the dynamic shape of a `Rotation2D` object.
    -
  ```

  ```{tip}
  In general, a `Rotation2D` object behaves like a rotation matrix, although
  its internal representation may differ.
  ```

  ```{warning}
  While other TensorFlow APIs may also work as expected when passed a
  `Rotation2D`, this is not supported and their behavior may change in the
  future.
  ```

  Example:

    >>> # Initialize a rotation object using a rotation matrix.
    >>> rot = tfmri.geometry.Rotation2D.from_matrix([[0.0, -1.0], [1.0, 0.0]])
    >>> print(rot)
    tfmri.geometry.Rotation2D(shape=(2, 2), dtype=float32)
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
    with tf.name_scope(name or "rotation_2d/from_small_euler"):
      return cls(_matrix=rotation_matrix_2d.from_small_euler(angle))

  def as_matrix(self, name=None):
    r"""Returns a rotation matrix representation of this rotation.

    Args:
      name: A name for this op. Defaults to `"rotation_2d/as_matrix"`.

    Returns:
      A `tf.Tensor` of shape `[..., 2, 2]`, where the last two dimensions
      represent a rotation matrix.
    """
    with tf.name_scope(name or "rotation_2d/as_matrix"):
      return tf.identity(self._matrix)

  def as_euler(self, name=None):
    r"""Returns an angle representation of this rotation.

    Args:
      name: A name for this op. Defaults to `"rotation_2d/as_euler"`.

    Returns:
      A `tf.Tensor` of shape `[..., 1]`, where the last dimension represents an
      angle in radians.
    """
    with tf.name_scope(name or "rotation_2d/as_euler"):
      return euler_2d.from_matrix(self._matrix)

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
    if isinstance(other, Rotation2D):
      return Rotation2D(_matrix=tf.matmul(self._matrix, other._matrix))
    raise ValueError(
        f"Cannot compose a `Rotation2D` with a `{type(other).__name__}`.")

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
    """Returns the shape of this rotation.

    Returns:
      A `tf.TensorShape`.
    """
    return self._matrix.shape

  @property
  def dtype(self):
    """Returns the dtype of this rotation.

    Returns:
      A `tf.dtypes.DType`.
    """
    return self._matrix.dtype


@tf.experimental.dispatch_for_api(tf.convert_to_tensor, {'value': Rotation2D})
def convert_to_tensor(value, dtype=None, dtype_hint=None, name=None):
  """Overrides `tf.convert_to_tensor` for `Rotation2D` objects."""
  return tf.convert_to_tensor(
      value.as_matrix(), dtype=dtype, dtype_hint=dtype_hint, name=name)


@tf.experimental.dispatch_for_api(
    tf.linalg.matmul, {'a': Rotation2D, 'b': Rotation2D})
def matmul(a, b,  # pylint: disable=missing-param-doc
           transpose_a=False,
           transpose_b=False,
           adjoint_a=False,
           adjoint_b=False,
           a_is_sparse=False,
           b_is_sparse=False,
           output_type=None,
           name=None):
  """Overrides `tf.linalg.matmul` for `Rotation2D` objects."""
  if a_is_sparse or b_is_sparse:
    raise ValueError("Rotation2D does not support sparse matmul.")
  return Rotation2D(_matrix=tf.linalg.matmul(a.as_matrix(), b.as_matrix(),
                                             transpose_a=transpose_a,
                                             transpose_b=transpose_b,
                                             adjoint_a=adjoint_a,
                                             adjoint_b=adjoint_b,
                                             output_type=output_type,
                                             name=name))


@tf.experimental.dispatch_for_api(tf.linalg.matvec, {'a': Rotation2D})
def matvec(a, b,  # pylint: disable=missing-param-doc
           transpose_a=False,
           adjoint_a=False,
           a_is_sparse=False,
           b_is_sparse=False,
           name=None):
  """Overrides `tf.linalg.matvec` for `Rotation2D` objects."""
  if a_is_sparse or b_is_sparse:
    raise ValueError("Rotation2D does not support sparse matvec.")
  return tf.linalg.matvec(a.as_matrix(), b,
                          transpose_a=transpose_a,
                          adjoint_a=adjoint_a,
                          name=name)


@tf.experimental.dispatch_for_api(tf.shape, {'input': Rotation2D})
def shape(input, out_type=tf.int32, name=None):  # pylint: disable=redefined-builtin
  """Overrides `tf.shape` for `Rotation2D` objects."""
  return tf.shape(input.as_matrix(), out_type=out_type, name=name)
