# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Tests for module `util.check_util`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.util import test_util


class VerifyCompatibleTrajectoryTest(test_util.TestCase):
  """Tests for `check_util.verify_compatible_trajectory`."""
  @parameterized.parameters(
      # kspace_shape, traj_shape
      ([100], [100, 2]),
      ([3, 100], [3, 100, 2]),
      ([3, 100], [1, 100, 2]),
      ([100], [100, 3])
  )  # pylint: disable=missing-function-docstring
  def test_correct(self, kspace_shape, traj_shape):
    kspace = tf.zeros(shape=kspace_shape, dtype=tf.complex64)
    traj = tf.zeros(shape=traj_shape, dtype=tf.float32)
    valid_kspace, valid_traj = check_util.verify_compatible_trajectory(
        kspace, traj)
    self.assertAllClose(kspace, valid_kspace)
    self.assertAllClose(traj, valid_traj)

  def test_incompatible_dtypes(self):
    kspace = tf.zeros(shape=[3, 100], dtype=tf.complex64)
    traj = tf.zeros(shape=[100, 2], dtype=tf.complex64)
    with self.assertRaisesRegex(TypeError, "incompatible dtypes"):
      check_util.verify_compatible_trajectory(kspace, traj)

  def test_incompatible_static_samples(self):
    kspace = tf.zeros(shape=[3, 200], dtype=tf.complex64)
    traj = tf.zeros(shape=[100, 2], dtype=tf.float32)
    with self.assertRaisesRegex(ValueError, "number of samples"):
      check_util.verify_compatible_trajectory(kspace, traj)

  def test_incompatible_static_batch_shapes(self):
    kspace = tf.zeros(shape=[3, 100], dtype=tf.complex64)
    traj = tf.zeros(shape=[4, 100, 2], dtype=tf.float32)
    with self.assertRaisesRegex(ValueError, "incompatible batch shapes"):
      check_util.verify_compatible_trajectory(kspace, traj)
