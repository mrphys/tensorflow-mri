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
"""Tests for module `rotation_matrix_2d`."""
# This file is partly inspired by TensorFlow Graphics.

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.geometry import test_data as td
from tensorflow_mri.python.geometry import test_helpers
from tensorflow_mri.python.geometry.rotation_matrix_2d import RotationMatrix2D
from tensorflow_mri.python.util import test_util


class RotationMatrix2DTest(test_util.TestCase):
  """Tests for `RotationMatrix2D`."""
  def test_shape(self):
    matrix = RotationMatrix2D.from_euler([0.0])
    self.assertAllEqual([2, 2], matrix.shape)
    self.assertAllEqual([2, 2], tf.shape(matrix))

  def test_from_euler_normalized(self):
    """Tests that an angle maps to correct matrix."""
    euler_angles = test_helpers.generate_preset_test_euler_angles(dimensions=1)

    matrix = RotationMatrix2D.from_euler(euler_angles)
    self.assertAllEqual(np.ones(euler_angles.shape[0:-1] + (1,), dtype=bool),
                        matrix.is_valid())

  @parameterized.named_parameters(
      ("0", td.ANGLE_0, td.MAT_2D_ID),
      ("45", td.ANGLE_45, td.MAT_2D_45),
      ("90", td.ANGLE_90, td.MAT_2D_90),
      ("180", td.ANGLE_180, td.MAT_2D_180),
  )
  def test_from_euler(self, angle, expected):
    """Tests that an angle maps to correct matrix."""
    matrix = RotationMatrix2D.from_euler(angle)
    self.assertAllClose(expected, matrix.matrix)

  def test_from_euler_with_small_angles_approximation_random(self):
    """Tests small_angles approximation by comparing to exact calculation."""
    # Only generate small angles. For a test tolerance of 1e-3, 0.17 was found
    # empirically to be the range where the small angle approximation works.
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        min_angle=-0.17, max_angle=0.17, dimensions=1)

    exact_matrix = RotationMatrix2D.from_euler(
        random_euler_angles)
    approximate_matrix = (
        RotationMatrix2D.from_euler_with_small_angles_approximation(
            random_euler_angles))

    self.assertAllClose(exact_matrix.matrix, approximate_matrix.matrix,
                        atol=1e-3)

  def test_inverse_random(self):
    """Checks that inverting rotated points results in no transformation."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        dimensions=1)
    tensor_shape = random_euler_angles.shape[:-1]

    random_matrix = RotationMatrix2D.from_euler(random_euler_angles)
    random_point = np.random.normal(size=tensor_shape + (2,))
    rotated_random_points = random_matrix.rotate(random_point)
    predicted_invert_random_matrix = random_matrix.inverse()
    predicted_invert_rotated_random_points = (
        predicted_invert_random_matrix.rotate(rotated_random_points))

    self.assertAllClose(random_point, predicted_invert_rotated_random_points)

  @parameterized.named_parameters(
      ("preset1", td.AXIS_2D_0, td.ANGLE_90, td.AXIS_2D_0),
      ("preset2", td.AXIS_2D_X, td.ANGLE_90, td.AXIS_2D_Y),
  )
  def test_rotate(self, point, angle, expected):
    """Tests that the rotate function correctly rotates points."""
    result = RotationMatrix2D.from_euler(angle).rotate(point)
    self.assertAllClose(expected, result)


if __name__ == "__main__":
  tf.test.main()
