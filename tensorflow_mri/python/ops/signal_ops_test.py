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

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
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

    res_kspace, res_traj = signal_ops.crop_kspace(kspace, traj=traj,
                                                  cutoff=np.pi / 2,
                                                  mode='low_pass')
    self.assertAllClose(res_kspace, [1. + 2.j, 3. - 4.j])
    self.assertAllClose(res_traj, [[0.4, 1.0], [0, 0.5]])

    res_kspace, res_traj = signal_ops.crop_kspace(kspace, traj=traj,
                                                  cutoff=np.pi / 2,
                                                  mode='high_pass')
    self.assertAllClose(res_kspace, [2. + 2.j])
    self.assertAllClose(res_traj, [[3.0, 2.0]])

  @parameterized.product(filter_type=['hamming', 'hann', 'atanfilt'])
  @test_util.run_in_graph_and_eager_modes
  def test_filter_noncart(self, filter_type):  # pylint: disable=missing-param-doc
    """Test non-Cartesian k-space filtering."""
    filt_fn = {
        'hamming': signal_ops.hamming,
        'hann': signal_ops.hann,
        'atanfilt': signal_ops.atanfilt
    }

    kspace = [1. + 2.j, 2. + 2.j, 3. - 4.j]
    traj = [[0.4, 1.0], [3.0, 2.0], [0, 0.5]]
    radius = tf.norm(traj, axis=-1)

    expected = kspace * tf.cast(filt_fn[filter_type](radius), tf.complex64)
    result = signal_ops.filter_kspace(
        kspace, traj=traj, filter_type=filter_type)
    self.assertAllClose(expected, result)

  def test_filter_cart(self):
    """Test k-space filtering."""
    shape = [16, 16]
    kspace = tf.complex(
        tf.random.stateless_normal(shape, seed=[42, 231]),
        tf.random.stateless_normal(shape, seed=[42, 77]))

    vecs = [tf.linspace(-np.pi, np.pi - (2.0 * np.pi / s), s)
            for s in shape]  # pylint: disable=invalid-unary-operand-type
    grid = array_ops.meshgrid(*vecs)
    radius = tf.norm(grid, axis=-1)
    expected = kspace * tf.cast(signal_ops.hamming(radius), tf.complex64)

    result = signal_ops.filter_kspace(kspace)
    self.assertAllClose(expected, result)

  def test_filter_cart_batch(self):
    """Test k-space filtering."""
    shape = [16, 16]
    kspace = tf.complex(
        tf.random.stateless_normal([4] + shape, seed=[42, 231]),
        tf.random.stateless_normal([4] + shape, seed=[42, 77]))

    vecs = [tf.linspace(-np.pi, np.pi - (2.0 * np.pi / s), s)
            for s in shape]  # pylint: disable=invalid-unary-operand-type
    grid = array_ops.meshgrid(*vecs)
    radius = tf.norm(grid, axis=-1)
    expected = kspace * tf.cast(signal_ops.hamming(radius), tf.complex64)

    result = signal_ops.filter_kspace(kspace, filter_rank=2)
    self.assertAllClose(expected, result)


if __name__ == '__main__':
  tf.test.main()
