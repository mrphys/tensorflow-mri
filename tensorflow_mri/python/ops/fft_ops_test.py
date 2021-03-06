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
"""Tests for module `fft_ops`."""

import distutils
import itertools

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import fft_ops
from tensorflow_mri.python.util import test_util


class FFTOpsTest(test_util.TestCase):
  """Tests for FFT ops."""
  # pylint: disable=missing-function-docstring

  @test_util.run_in_graph_and_eager_modes
  def test_fftn(self):
    """Test forward FFT."""
    self._test_fftn_internal('forward')

  @test_util.run_in_graph_and_eager_modes
  def test_ifftn(self):
    """Test inverse FFT."""
    self._test_fftn_internal('backward')

  def _test_fftn_internal(self, transform): # pylint: disable=missing-param-doc
    """All FFT tests.

    Checks that FFT results match those of NumPy.
    """
    if transform == 'forward':
      tf_op = fft_ops.fftn
      np_op = np.fft.fftn
    elif transform == 'backward':
      tf_op = fft_ops.ifftn
      np_op = np.fft.ifftn

    rng = np.random.default_rng(0)
    x = rng.random((30, 20, 10), dtype=np.float32) + 1j * \
        rng.random((30, 20, 10), dtype=np.float32)
    input_rank = len(x.shape)

    shape = (None, (20, 10), (20, 20), (10, 10, 10))
    axes = (None, (0, 1), (2, 3), (-1,))
    norm = ('forward', 'backward', 'ortho')
    shift = (False,)

    keys = ('shape', 'axes', 'norm', 'shift')
    values = itertools.product(shape, axes, norm, shift)
    params = [dict(zip(keys, v)) for v in values]

    for p in params:
      with self.subTest(**p):

        # We can only run tests for {'forward', 'backward'}
        # normalization modes if NumPy version is at least 1.20, which
        # is when these modes were introduced.
        if (distutils.version.StrictVersion(np.__version__) <
            distutils.version.StrictVersion('1.20.0')):
          if p['norm'] in ('forward', 'backward'):
            self.skipTest("numpy version < 1.20.0")

        # Handle invalid parameters and check that appropriate
        # exceptions are raised.
        if p['axes'] is not None and any(
            ax >= input_rank or ax < -input_rank
            for ax in p['axes']):
          # Axes outside valid range.
          with self.assertRaises(tf.errors.InvalidArgumentError):
            y = tf_op(x, **p)
          continue
        if (p['axes'] is not None and p['shape'] is not None and
            not len(p['axes']) == len(p['shape'])):
          # Shape and axes do not have equal length.
          with self.assertRaises(tf.errors.InvalidArgumentError):
            y = tf_op(x, **p)
          continue

        # Test op.
        y = tf_op(x, **p)

        # Now get reference from NumPy. 'shift' is not supported in
        # NumPy, so we implement it manually.
        p['s'] = p.pop('shape')
        shift = p.pop('shift')
        x_ref = x

        # Compute using NumPy, with additional fftshifts as necessary.
        if shift:
          x_ref = np.fft.ifftshift(x_ref)
        y_ref = np_op(x_ref, **p)
        if shift:
          y_ref = np.fft.fftshift(y_ref)

        self.assertAllClose(y, y_ref, rtol=1e-4, atol=1e-4)

if __name__ == '__main__':
  tf.test.main()
