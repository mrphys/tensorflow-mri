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
"""Tests for 2D rotation."""
# This file is partly inspired by TensorFlow Graphics.
# pylint: disable=missing-param-doc

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import test_data as td
from tensorflow_mri.python.geometry.rotation import test_helpers
from tensorflow_mri.python.geometry.rotation_2d import Rotation2D
from tensorflow_mri.python.util import test_util


class Rotation2DTest(test_util.TestCase):
  """Tests for `Rotation2D`."""
  def test_shape(self):
    """Tests shape."""
    rot = Rotation2D.from_euler([0.0])
    self.assertAllEqual([2, 2], rot.shape)
    self.assertAllEqual([2, 2], tf.shape(rot))

    rot = Rotation2D.from_euler([[0.0], [np.pi]])
    self.assertAllEqual([2, 2, 2], rot.shape)
    self.assertAllEqual([2, 2, 2], tf.shape(rot))

  def test_equal(self):
    """Tests equality operator."""
    rot1 = Rotation2D.from_euler([0.0])
    rot2 = Rotation2D.from_euler([0.0])
    self.assertAllEqual(True, rot1 == rot2)

    rot1 = Rotation2D.from_euler([0.0])
    rot2 = Rotation2D.from_euler([np.pi])
    self.assertAllEqual(False, rot1 == rot2)

    rot1 = Rotation2D.from_euler([[0.0], [np.pi]])
    rot2 = Rotation2D.from_euler([[0.0], [np.pi]])
    self.assertAllEqual([True, True], rot1 == rot2)

    rot1 = Rotation2D.from_euler([[0.0], [0.0]])
    rot2 = Rotation2D.from_euler([[0.0], [np.pi]])
    self.assertAllEqual([True, False], rot1 == rot2)

  def test_repr(self):
    """Tests that repr works."""
    expected = "<tfmri.geometry.Rotation2D(shape=(2, 2), dtype=float32)>"
    rot = Rotation2D.from_euler([0.0])
    self.assertEqual(expected, repr(rot))
    self.assertEqual(expected[1:-1], str(rot))

  def test_matmul(self):
    """Tests that matmul works."""
    rot = Rotation2D.from_euler([np.pi])
    composed = rot @ rot
    self.assertAllClose(np.eye(2), composed.as_matrix())

    composed = tf.linalg.matmul(rot, rot)
    self.assertAllClose(np.eye(2), composed.as_matrix())

  def test_matvec(self):
    """Tests that matvec works."""
    rot = Rotation2D.from_euler([np.pi])
    vec = tf.constant([1.0, -1.0])
    self.assertAllClose(rot.rotate(vec), tf.linalg.matvec(rot, vec))

  def test_convert_to_tensor(self):
    """Tests that conversion to tensor works."""
    rot = Rotation2D.from_euler([0.0])
    self.assertIsInstance(tf.convert_to_tensor(rot), tf.Tensor)
    self.assertAllClose(np.eye(2), tf.convert_to_tensor(rot))

  @parameterized.named_parameters(
      ("0", [0.0]),
      ("45", [np.pi / 4]),
      ("90", [np.pi / 2]),
      ("135", [np.pi * 3 / 4]),
      ("-45", [-np.pi / 4]),
      ("-90", [-np.pi / 2]),
      ("-135", [-np.pi * 3 / 4])
  )
  def test_as_euler(self, angle):  # pylint: disable=missing-param-doc
    """Tests that `as_euler` returns the correct angle."""
    rot = Rotation2D.from_euler(angle)
    self.assertAllClose(angle, rot.as_euler())

  def test_from_matrix(self):
    """Tests that rotation can be initialized from matrix."""
    rot = Rotation2D.from_matrix(np.eye(2))
    self.assertAllClose(np.eye(2), rot.as_matrix())

  def test_from_euler_normalized(self):
    """Tests that an angle maps to correct matrix."""
    euler_angles = test_helpers.generate_preset_test_euler_angles(dimensions=1)

    rot = Rotation2D.from_euler(euler_angles)
    self.assertAllEqual(np.ones(euler_angles.shape[0:-1] + (1,), dtype=bool),
                        rot.is_valid())

  @parameterized.named_parameters(
      ("0", td.ANGLE_0, td.MAT_2D_ID),
      ("45", td.ANGLE_45, td.MAT_2D_45),
      ("90", td.ANGLE_90, td.MAT_2D_90),
      ("180", td.ANGLE_180, td.MAT_2D_180),
  )
  def test_from_euler(self, angle, expected):
    """Tests that an angle maps to correct matrix."""
    self.assertAllClose(expected, Rotation2D.from_euler(angle).as_matrix())

  def test_from_euler_with_small_angles_approximation_random(self):
    """Tests small angles approximation by comparing to exact calculation."""
    # Only generate small angles. For a test tolerance of 1e-3, 0.17 was found
    # empirically to be the range where the small angle approximation works.
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        min_angle=-0.17, max_angle=0.17, dimensions=1)

    exact_rot = Rotation2D.from_euler(random_euler_angles)
    approx_rot = Rotation2D.from_small_euler(random_euler_angles)

    self.assertAllClose(exact_rot.as_matrix(), approx_rot.as_matrix(),
                        atol=1e-3)

  def test_inverse_random(self):
    """Checks that inverting rotated points results in no transformation."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        dimensions=1)
    tensor_shape = random_euler_angles.shape[:-1]

    random_rot = Rotation2D.from_euler(random_euler_angles)
    random_point = np.random.normal(size=tensor_shape + (2,))
    rotated_random_points = random_rot.rotate(random_point)
    predicted_invert_random_matrix = random_rot.inverse()
    predicted_invert_rotated_random_points = (
        predicted_invert_random_matrix.rotate(rotated_random_points))

    self.assertAllClose(random_point, predicted_invert_rotated_random_points)

  @parameterized.named_parameters(
      ("preset1", td.AXIS_2D_0, td.ANGLE_90, td.AXIS_2D_0),
      ("preset2", td.AXIS_2D_X, td.ANGLE_90, td.AXIS_2D_Y),
  )
  def test_rotate(self, point, angle, expected):
    """Tests that the rotate function correctly rotates points."""
    result = Rotation2D.from_euler(angle).rotate(point)
    self.assertAllClose(expected, result)


if __name__ == "__main__":
  tf.test.main()
