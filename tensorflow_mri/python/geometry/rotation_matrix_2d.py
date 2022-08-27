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
"""2D rotation matrix."""
# This file is partly inspired by TensorFlow Graphics.

import tensorflow as tf

from tensorflow_mri.python.geometry import rotation_matrix
from tensorflow_mri.python.util import api_util


FORMAT_KWARGS = dict(n=2)


@api_util.export("geometry.RotationMatrix2D")
class RotationMatrix2D(rotation_matrix.RotationMatrix):
  __doc__ = rotation_matrix.RotationMatrix.__doc__.format(**FORMAT_KWARGS)

  @classmethod
  def from_euler(cls, angle, name=None):
    r"""Creates a rotation matrix from an angle.

    Converts an angle $\theta$ to a 2D rotation matrix following the equation

    $$
      \mathbf{R} =
      \begin{bmatrix}
      \cos(\theta) & -\sin(\theta) \\
      \sin(\theta) & \cos(\theta)
      \end{bmatrix}.
    $$

    ```{note}
    The resulting matrix rotates points in the $xy$-plane counterclockwise.
    ```

    Args:
      angle: A `tf.Tensor` of shape `[..., 1]`, where the last dimension
        represents an angle in radians.
      name: A name for this op. Defaults to `"from_euler"`.

    Returns:
      A `RotationMatrix2D`.

    Raises:
      ValueError: If the shape of `angle` is invalid.
    """
    name = name or "from_euler"
    with tf.name_scope(f"rotation_matrix_2d/{name}"):
      angle = tf.convert_to_tensor(angle)

      if not angle.shape[-1:].is_compatible_with([1]):
        raise ValueError(
            f"angle must have shape `[..., 1]`, but got: {angle.shape}")

      cos_angle = tf.math.cos(angle)
      sin_angle = tf.math.sin(angle)
      matrix = tf.stack([cos_angle, -sin_angle, sin_angle, cos_angle], axis=-1)
      output_shape = tf.concat([tf.shape(angle)[:-1], [2, 2]], axis=-1)
      return cls(tf.reshape(matrix, output_shape))

  @classmethod
  def from_euler_with_small_angles_approximation(cls, angle, name=None):
    r"""Creates a rotation matrix from an angle using small angle approximation.

    Under the small angle assumption, $\sin(x)$ and $\cos(x)$ can be
    approximated by their second order Taylor expansions, where
    $\sin(x) \approx x$ and $\cos(x) \approx 1 - \frac{x^2}{2}$. The 2D
    rotation matrix will then be approximated as

    $$
      \mathbf{R} =
      \begin{bmatrix}
      1.0 - 0.5\theta^2 & -\theta \\
      \theta & 1.0 - 0.5\theta^2
      \end{bmatrix}.
    $$

    ```{note}
    The resulting matrix rotates points in the $xy$-plane counterclockwise.
    ```

    ```{note}
    This function does not verify the smallness of the angles.
    ```

    Args:
      angle: A `tf.Tensor` of shape `[..., 1]`, where the last dimension
        represents an angle in radians.
      name: A name for this op. Defaults to
        "from_euler_with_small_angles_approximation".

    Returns:
      A `RotationMatrix2D`.

    Raises:
      ValueError: If the shape of `angle` is invalid.
    """
    name = name or "from_euler_with_small_angles_approximation"
    with tf.name_scope(f"rotation_matrix_2d/{name}"):
      angle = tf.convert_to_tensor(angle)

      if not angle.shape[-1:].is_compatible_with([1]):
        raise ValueError(
            f"angle must have shape `[..., 1]`, but got: {angle.shape}")

      cos_angle = 1.0 - 0.5 * angle * angle
      sin_angle = angle
      matrix = tf.stack([cos_angle, -sin_angle, sin_angle, cos_angle], axis=-1)
      output_shape = tf.concat([tf.shape(angle)[:-1], [2, 2]], axis=-1)
      return cls(tf.reshape(matrix, output_shape))

  # The following methods are overridden only to generate the docstrings.
  def inverse(self, name=None):
    return super().inverse(name=name)
  inverse.__doc__ = rotation_matrix.RotationMatrix.inverse.__doc__.format(
      **FORMAT_KWARGS)

  def is_valid(self, atol=1e-3, name=None):
    return super().is_valid(atol=atol, name=name)
  is_valid.__doc__ = rotation_matrix.RotationMatrix.is_valid.__doc__.format(
      **FORMAT_KWARGS)

  def rotate(self, point, name=None):
    return super().rotate(point=point, name=name)
  rotate.__doc__ = rotation_matrix.RotationMatrix.rotate.__doc__.format(
      **FORMAT_KWARGS)
