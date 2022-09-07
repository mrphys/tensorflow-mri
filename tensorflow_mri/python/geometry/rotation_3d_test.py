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
"""Tests for 3D rotation."""
# This file is partly inspired by TensorFlow Graphics.
# pylint: disable=missing-param-doc

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.geometry.rotation import test_data as td
from tensorflow_mri.python.geometry.rotation import test_helpers
from tensorflow_mri.python.geometry.rotation_3d import Rotation3D
from tensorflow_mri.python.util import test_util


class Rotation3DTest(test_util.TestCase):
  """Tests for `Rotation3D`."""
  def test_shape(self):
    """Tests shape."""
    rot = Rotation3D.from_euler([0.0, 0.0, 0.0])
    self.assertAllEqual([3, 3], rot.shape)
    self.assertAllEqual([3, 3], tf.shape(rot))

    rot = Rotation3D.from_euler([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    self.assertAllEqual([2, 3, 3], rot.shape)
    self.assertAllEqual([2, 3, 3], tf.shape(rot))

  def test_equal(self):
    """Tests equality operator."""
    rot1 = Rotation3D.from_euler([0.0, 0.0, 0.0])
    rot2 = Rotation3D.from_euler([0.0, 0.0, 0.0])
    self.assertAllEqual(True, rot1 == rot2)

    rot1 = Rotation3D.from_euler([0.0, 0.0, 0.0])
    rot2 = Rotation3D.from_euler([np.pi, 0.0, 0.0])
    self.assertAllEqual(False, rot1 == rot2)

    rot1 = Rotation3D.from_euler([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    rot2 = Rotation3D.from_euler([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    self.assertAllEqual([True, True], rot1 == rot2)

    rot1 = Rotation3D.from_euler([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    rot2 = Rotation3D.from_euler([[0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]])
    self.assertAllEqual([True, False], rot1 == rot2)

  def test_repr(self):
    rot = Rotation3D.from_euler([0.0, 0.0, 0.0])
    self.assertEqual(
        "<tfmri.geometry.Rotation3D(shape=(3, 3), dtype=float32)>", repr(rot))

  def test_convert_to_tensor(self):
    """Tests that conversion to tensor works."""
    rot = Rotation3D.from_euler([0.0, 0.0, 0.0])
    self.assertIsInstance(tf.convert_to_tensor(rot), tf.Tensor)
    self.assertAllClose(np.eye(3), tf.convert_to_tensor(rot))

  def test_from_axis_angle_normalized_random(self):
    """Tests that axis-angles can be converted to rotation matrices."""
    tensor_shape = np.random.randint(1, 10, size=np.random.randint(3)).tolist()
    random_axis = np.random.normal(size=tensor_shape + [3])
    random_axis /= np.linalg.norm(random_axis, axis=-1, keepdims=True)
    random_angle = np.random.normal(size=tensor_shape + [1])

    rotation = Rotation3D.from_axis_angle(random_axis, random_angle)

    self.assertAllEqual(rotation.is_valid(), np.ones(tensor_shape + [1]))

  @parameterized.named_parameters(
      ("preset0", td.AXIS_3D_X, td.ANGLE_45, td.MAT_3D_X_45),
      ("preset1", td.AXIS_3D_Y, td.ANGLE_45, td.MAT_3D_Y_45),
      ("preset2", td.AXIS_3D_Z, td.ANGLE_45, td.MAT_3D_Z_45),
      ("preset3", td.AXIS_3D_X, td.ANGLE_90, td.MAT_3D_X_90),
      ("preset4", td.AXIS_3D_Y, td.ANGLE_90, td.MAT_3D_Y_90),
      ("preset5", td.AXIS_3D_Z, td.ANGLE_90, td.MAT_3D_Z_90),
      ("preset6", td.AXIS_3D_X, td.ANGLE_180, td.MAT_3D_X_180),
      ("preset7", td.AXIS_3D_Y, td.ANGLE_180, td.MAT_3D_Y_180),
      ("preset8", td.AXIS_3D_Z, td.ANGLE_180, td.MAT_3D_Z_180)
  )
  def test_from_axis_angle(self, axis, angle, matrix):
    """Tests that an axis-angle maps to correct matrix."""
    self.assertAllClose(
        matrix, Rotation3D.from_axis_angle(axis, angle).as_matrix())

  def test_from_axis_angle_random(self):
    """Tests conversion to matrix."""
    tensor_shape = np.random.randint(1, 10, size=np.random.randint(3)).tolist()
    random_axis = np.random.normal(size=tensor_shape + [3])
    random_axis /= np.linalg.norm(random_axis, axis=-1, keepdims=True)
    random_angle = np.random.normal(size=tensor_shape + [1])

    rotation = Rotation3D.from_axis_angle(random_axis, random_angle)

    # Checks that resulting rotation matrices are normalized.
    self.assertAllEqual(rotation.is_valid(), np.ones(tensor_shape + [1]))

  @parameterized.named_parameters(
      ("preset0", td.AXIS_3D_X, td.ANGLE_90, td.AXIS_3D_X, td.AXIS_3D_X),
      ("preset1", td.AXIS_3D_X, td.ANGLE_90, td.AXIS_3D_Y, td.AXIS_3D_Z),
      ("preset2", td.AXIS_3D_X, -td.ANGLE_90, td.AXIS_3D_Z, td.AXIS_3D_Y),
      ("preset3", td.AXIS_3D_Y, -td.ANGLE_90, td.AXIS_3D_X, td.AXIS_3D_Z),
      ("preset4", td.AXIS_3D_Y, td.ANGLE_90, td.AXIS_3D_Y, td.AXIS_3D_Y),
      ("preset5", td.AXIS_3D_Y, td.ANGLE_90, td.AXIS_3D_Z, td.AXIS_3D_X),
      ("preset6", td.AXIS_3D_Z, td.ANGLE_90, td.AXIS_3D_X, td.AXIS_3D_Y),
      ("preset7", td.AXIS_3D_Z, -td.ANGLE_90, td.AXIS_3D_Y, td.AXIS_3D_X),
      ("preset8", td.AXIS_3D_Z, td.ANGLE_90, td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  def test_from_axis_angle_rotate_vector_preset(
      self, axis, angle, point, expected):
    """Tests the directionality of axis-angle rotations."""
    self.assertAllClose(
        expected, Rotation3D.from_axis_angle(axis, angle).rotate(point))

  def test_from_euler_normalized_preset(self):
    """Tests that euler angles can be converted to rotation matrices."""
    euler_angles = test_helpers.generate_preset_test_euler_angles()

    matrix = Rotation3D.from_euler(euler_angles)
    self.assertAllEqual(
        matrix.is_valid(), np.ones(euler_angles.shape[0:-1] + (1,)))

  def test_from_euler_normalized_random(self):
    """Tests that euler angles can be converted to rotation matrices."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    matrix = Rotation3D.from_euler(random_euler_angles)
    self.assertAllEqual(
        matrix.is_valid(), np.ones(random_euler_angles.shape[0:-1] + (1,)))

  @parameterized.named_parameters(
      ("preset0", td.AXIS_3D_0, td.MAT_3D_ID),
      ("preset1", td.ANGLE_45 * td.AXIS_3D_X, td.MAT_3D_X_45),
      ("preset2", td.ANGLE_45 * td.AXIS_3D_Y, td.MAT_3D_Y_45),
      ("preset3", td.ANGLE_45 * td.AXIS_3D_Z, td.MAT_3D_Z_45),
      ("preset4", td.ANGLE_90 * td.AXIS_3D_X, td.MAT_3D_X_90),
      ("preset5", td.ANGLE_90 * td.AXIS_3D_Y, td.MAT_3D_Y_90),
      ("preset6", td.ANGLE_90 * td.AXIS_3D_Z, td.MAT_3D_Z_90),
      ("preset7", td.ANGLE_180 * td.AXIS_3D_X, td.MAT_3D_X_180),
      ("preset8", td.ANGLE_180 * td.AXIS_3D_Y, td.MAT_3D_Y_180),
      ("preset9", td.ANGLE_180 * td.AXIS_3D_Z, td.MAT_3D_Z_180),
  )
  def test_from_euler(self, angle, expected):
    """Tests that Euler angles create the expected matrix."""
    rotation = Rotation3D.from_euler(angle)
    self.assertAllClose(expected, rotation.as_matrix())

  def test_from_euler_random(self):
    """Tests that Euler angles produce the same result as axis-angle."""
    angles = test_helpers.generate_random_test_euler_angles()
    matrix = Rotation3D.from_euler(angles)
    tensor_tile = angles.shape[:-1]

    x_axis = np.tile(td.AXIS_3D_X, tensor_tile + (1,))
    y_axis = np.tile(td.AXIS_3D_Y, tensor_tile + (1,))
    z_axis = np.tile(td.AXIS_3D_Z, tensor_tile + (1,))
    x_angle = np.expand_dims(angles[..., 0], axis=-1)
    y_angle = np.expand_dims(angles[..., 1], axis=-1)
    z_angle = np.expand_dims(angles[..., 2], axis=-1)
    x_rotation = Rotation3D.from_axis_angle(x_axis, x_angle)
    y_rotation = Rotation3D.from_axis_angle(y_axis, y_angle)
    z_rotation = Rotation3D.from_axis_angle(z_axis, z_angle)
    expected_matrix = z_rotation @ (y_rotation @ x_rotation)

    self.assertAllClose(expected_matrix.as_matrix(), matrix.as_matrix(),
                        rtol=1e-3)

  def test_from_quaternion_normalized_random(self):
    """Tests that random quaternions can be converted to rotation matrices."""
    random_quaternion = test_helpers.generate_random_test_quaternions()
    tensor_shape = random_quaternion.shape[:-1]

    random_rot = Rotation3D.from_quaternion(random_quaternion)

    self.assertAllEqual(
        random_rot.is_valid(),
        np.ones(tensor_shape + (1,)))

  def test_from_quaternion(self):
    """Tests that a quaternion maps to correct matrix."""
    preset_quaternions = test_helpers.generate_preset_test_quaternions()

    preset_matrices = test_helpers.generate_preset_test_rotation_matrices_3d()

    self.assertAllClose(
        preset_matrices,
        Rotation3D.from_quaternion(preset_quaternions).as_matrix())

  def test_inverse_normalized_random(self):
    """Checks that inverted rotation matrices are valid rotations."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]

    random_rot = Rotation3D.from_euler(random_euler_angle)
    predicted_invert_random_rot = random_rot.inverse()

    self.assertAllEqual(
        predicted_invert_random_rot.is_valid(),
        np.ones(tensor_tile + (1,)))

  def test_inverse_random(self):
    """Checks that inverting rotated points results in no transformation."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]
    random_rot = Rotation3D.from_euler(random_euler_angle)
    random_point = np.random.normal(size=tensor_tile + (3,))

    rotated_random_points = random_rot.rotate(random_point)
    inv_random_rot = random_rot.inverse()
    inv_rotated_random_points = inv_random_rot.rotate(rotated_random_points)

    self.assertAllClose(random_point, inv_rotated_random_points, rtol=1e-6)

  def test_is_valid_random(self):
    """Tests that is_valid works as intended."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]

    rotation = Rotation3D.from_euler(random_euler_angle)
    pred_normalized = rotation.is_valid()

    with self.subTest(name="all_normalized"):
      self.assertAllEqual(pred_normalized,
                          np.ones(shape=tensor_tile + (1,), dtype=bool))

    with self.subTest(name="non_orthonormal"):
      test_matrix = np.array([[2., 0., 0.], [0., 0.5, 0], [0., 0., 1.]])
      rotation = Rotation3D.from_matrix(test_matrix)
      pred_normalized = rotation.is_valid()
      self.assertAllEqual(pred_normalized, np.zeros(shape=(1,), dtype=bool))

    with self.subTest(name="negative_orthonormal"):
      test_matrix = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
      rotation = Rotation3D.from_matrix(test_matrix)
      pred_normalized = rotation.is_valid()
      self.assertAllEqual(pred_normalized, np.zeros(shape=(1,), dtype=bool))

  @parameterized.named_parameters(
      ("preset0", td.ANGLE_90 * td.AXIS_3D_X, td.AXIS_3D_X, td.AXIS_3D_X),
      ("preset1", td.ANGLE_90 * td.AXIS_3D_X, td.AXIS_3D_Y, td.AXIS_3D_Z),
      ("preset2", -td.ANGLE_90 * td.AXIS_3D_X, td.AXIS_3D_Z, td.AXIS_3D_Y),
      ("preset3", -td.ANGLE_90 * td.AXIS_3D_Y, td.AXIS_3D_X, td.AXIS_3D_Z),
      ("preset4", td.ANGLE_90 * td.AXIS_3D_Y, td.AXIS_3D_Y, td.AXIS_3D_Y),
      ("preset5", td.ANGLE_90 * td.AXIS_3D_Y, td.AXIS_3D_Z, td.AXIS_3D_X),
      ("preset6", td.ANGLE_90 * td.AXIS_3D_Z, td.AXIS_3D_X, td.AXIS_3D_Y),
      ("preset7", -td.ANGLE_90 * td.AXIS_3D_Z, td.AXIS_3D_Y, td.AXIS_3D_X),
      ("preset8", td.ANGLE_90 * td.AXIS_3D_Z, td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  def test_rotate_vector_preset(self, angles, point, expected):
    """Tests that the rotate function produces the expected results."""
    self.assertAllClose(expected, Rotation3D.from_euler(angles).rotate(point))


if __name__ == "__main__":
  tf.test.main()
