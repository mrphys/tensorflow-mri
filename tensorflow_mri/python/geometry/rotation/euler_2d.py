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
"""2D angles."""

import tensorflow as tf


def from_matrix(matrix):
  """Converts a 2D rotation matrix to an angle.

  Args:
    matrix: A `tf.Tensor` of shape `[..., 2, 2]`.

  Returns:
    A `tf.Tensor` of shape `[..., 1]`.

  Raises:
    ValueError: If the shape of `matrix` is invalid.
  """
  matrix = tf.convert_to_tensor(matrix)

  if matrix.shape[-1] != 2 or matrix.shape[-2] != 2:
    raise ValueError(
        f"matrix must have shape `[..., 2, 2]`, but got: {matrix.shape}")

  angle = tf.math.atan2(matrix[..., 1, 0], matrix[..., 0, 0])
  return tf.expand_dims(angle, axis=-1)
