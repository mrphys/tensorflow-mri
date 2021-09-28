# Copyright 2021 University College London. All Rights Reserved.
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
"""Tests for module `signal_ops`."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import signal_ops
from tensorflow_mri.python.util import test_util


class FilterTest(test_util.TestCase):
  """Test filters."""
  def test_hamming(self):
    """Test Hamming filter."""
    x = tf.linspace(-np.pi, np.pi, 100)
    result = signal_ops.hamming(x)
    self.assertAllClose(result, np.hamming(100))


class KSpaceFilterTest(test_util.TestCase):
  """Test k-space filters."""

  def test_crop(self):
    """Test k-space cropping."""
    kspace = [1. + 2.j, 2. + 2.j, 3. - 4.j]
    traj = [[0.4, 1.0], [3.0, 2.0], [0, 0.5]]

    res_kspace, res_traj = signal_ops.crop_kspace(kspace, traj, np.pi / 2,
                                                  mode='low_pass')
    self.assertAllClose(res_kspace, [1. + 2.j, 3. - 4.j])
    self.assertAllClose(res_traj, [[0.4, 1.0], [0, 0.5]])

    res_kspace, res_traj = signal_ops.crop_kspace(kspace, traj, np.pi / 2,
                                                  mode='high_pass')
    self.assertAllClose(res_kspace, [2. + 2.j])
    self.assertAllClose(res_traj, [[3.0, 2.0]])

  def test_filter(self):
    """Test k-space filtering."""
    kspace = [1. + 2.j, 2. + 2.j, 3. - 4.j]
    traj = [[0.4, 1.0], [3.0, 2.0], [0, 0.5]]
    radius = tf.norm(traj, axis=-1)

    result = signal_ops.filter_kspace(kspace, traj)
    self.assertAllClose(
        result, kspace * tf.cast(signal_ops.hamming(radius),
                                 tf.complex64))


if __name__ == '__main__':
  tf.test.main()
