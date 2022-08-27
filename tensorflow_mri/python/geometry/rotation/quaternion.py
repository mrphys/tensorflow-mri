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
"""Quaternions."""

import tensorflow as tf


def from_euler(angles):
  """Converts Euler angles to a quaternion.

  Args:
    angles: A `tf.Tensor` of shape `[..., 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 4]`.

  Raises:
    ValueError: If the shape of `angles` is invalid.
  """
  angles = tf.convert_to_tensor(angles)

  if angles.shape[-1] != 3:
    raise ValueError(f"angles must have shape `[..., 3]`, "
                     f"but got: {angles.shape}")

  half_angles = angles / 2.0
  cos_half_angles = tf.math.cos(half_angles)
  sin_half_angles = tf.math.sin(half_angles)
  return _build_quaternion_from_sines_and_cosines(sin_half_angles,
                                                  cos_half_angles)


def from_small_euler(angles):
  """Converts small Euler angles to a quaternion.

  Args:
    angles: A `tf.Tensor` of shape `[..., 3]`.

  Returns:
    A `tf.Tensor` of shape `[..., 4]`.

  Raises:
    ValueError: If the shape of `angles` is invalid.
  """
  angles = tf.convert_to_tensor(angles)

  if angles.shape[-1] != 3:
    raise ValueError(f"angles must have shape `[..., 3]`, "
                    f"but got: {angles.shape}")

  half_angles = angles / 2.0
  cos_half_angles = 1.0 - 0.5 * half_angles * half_angles
  sin_half_angles = half_angles
  quaternion = _build_quaternion_from_sines_and_cosines(
      sin_half_angles, cos_half_angles)

  # We need to normalize the quaternion due to the small angle approximation.
  return tf.nn.l2_normalize(quaternion, axis=-1)


def _build_quaternion_from_sines_and_cosines(sin_half_angles, cos_half_angles):
  """Builds a quaternion from sines and cosines of half Euler angles.

  Args:
    sin_half_angles: A tensor of shape `[..., 3]`, where the last
      dimension represents the sine of half Euler angles.
    cos_half_angles: A tensor of shape `[..., 3]`, where the last
      dimension represents the cosine of half Euler angles.

  Returns:
    A `tf.Tensor` of shape `[..., 4]`, where the last dimension represents
    a quaternion.
  """
  c1, c2, c3 = tf.unstack(cos_half_angles, axis=-1)
  s1, s2, s3 = tf.unstack(sin_half_angles, axis=-1)
  w = c1 * c2 * c3 + s1 * s2 * s3
  x = -c1 * s2 * s3 + s1 * c2 * c3
  y = c1 * s2 * c3 + s1 * c2 * s3
  z = -s1 * s2 * c3 + c1 * c2 * s3
  return tf.stack((x, y, z, w), axis=-1)


def multiply(quaternion1, quaternion2):
  """Multiplies two quaternions.

  Args:
    quaternion1: A `tf.Tensor` of shape `[..., 4]`, where the last dimension
      represents a quaternion.
    quaternion2: A `tf.Tensor` of shape `[..., 4]`, where the last dimension
      represents a quaternion.

  Returns:
    A `tf.Tensor` of shape `[..., 4]` representing quaternions.

  Raises:
    ValueError: If the shape of `quaternion1` or `quaternion2` is invalid.
  """
  quaternion1 = tf.convert_to_tensor(value=quaternion1)
  quaternion2 = tf.convert_to_tensor(value=quaternion2)

  if quaternion1.shape[-1] != 4:
    raise ValueError(f"quaternion1 must have shape `[..., 4]`, "
                     f"but got: {quaternion1.shape}")
  if quaternion2.shape[-1] != 4:
    raise ValueError(f"quaternion2 must have shape `[..., 4]`, "
                     f"but got: {quaternion2.shape}")

  x1, y1, z1, w1 = tf.unstack(quaternion1, axis=-1)
  x2, y2, z2, w2 = tf.unstack(quaternion2, axis=-1)
  x = x1 * w2 + y1 * z2 - z1 * y2 + w1 * x2
  y = -x1 * z2 + y1 * w2 + z1 * x2 + w1 * y2
  z = x1 * y2 - y1 * x2 + z1 * w2 + w1 * z2
  w = -x1 * x2 - y1 * y2 - z1 * z2 + w1 * w2
  return tf.stack((x, y, z, w), axis=-1)
