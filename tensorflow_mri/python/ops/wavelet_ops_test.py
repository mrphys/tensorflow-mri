# ==============================================================================
# Copyright 2022 University College London. All Rights Reserved.
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
"""Tests for module `wavelet_ops`."""

from absl.testing import parameterized
import numpy as np
import pywt

from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import test_util


class DiscreteWaveletTransformTest(test_util.TestCase):
  """Tests for `dwt` and `idwt` operators."""
  # pylint: disable=missing-function-docstring
  @parameterized.product(wavelet=pywt.wavelist(kind='discrete'),
                         odd_length=[True, False])
  def test_dwt_idwt(self, wavelet, odd_length):
    wt = wavelet_ops._as_wavelet(wavelet)  # pylint: disable=protected-access
    if len(wt.dec_lo) > 8:
      self.skipTest('Wavelet too long for this test.')

    if odd_length:
      x = np.asarray([3, 7, 1, 1, -2, 5, 4, 6, 1], dtype=np.float32)
    else:
      x = np.asarray([3, 7, 1, 1, -2, 5, 4, 6], dtype=np.float32)

    a, d = pywt.dwt(x, wavelet)
    result = wavelet_ops.dwt(x, wavelet)
    self.assertAllClose(a, result['a'])
    self.assertAllClose(d, result['d'])

    y = pywt.idwt(a, d, wavelet)
    result = wavelet_ops.idwt(result, wavelet)
    self.assertAllClose(y, result)


  @parameterized.product(wavelet=pywt.wavelist(kind='discrete'),
                         axes=[None, [0], [1], [0, 1]])
  def test_dwt_idwt_2d(self, wavelet, axes):
    wt = wavelet_ops._as_wavelet(wavelet)  # pylint: disable=protected-access
    if len(wt.dec_lo) > 4:
      self.skipTest('Wavelet too long for this test.')

    x = np.array([[0, 4, 1, 5, 1, 4],
                  [0, 5, 26, 3, 2, 1],
                  [5, 8, 2, 33, 4, 9],
                  [2, 5, 19, 4, 19, 1]], dtype=np.float32)

    expected = pywt.dwtn(x, wavelet, axes=axes)
    result = wavelet_ops.dwt(x, wavelet, axes=axes)
    self.assertAllClose(expected, result, atol=1e-4, rtol=1e-4)

    expected = pywt.idwtn(result, wavelet, axes=axes)
    result = wavelet_ops.idwt(result, wavelet, axes=axes)
    self.assertAllClose(expected, result, atol=1e-4, rtol=1e-4)


  @test_util.run_all_execution_modes
  def test_dwt_idwt_complex(self):
    x = (np.asarray([3, 7, 1, 1, -2, 5, 4, 6], dtype=np.float32) +
         1j * np.asarray([-1, 2, 5, 2, -3, 3, 4, 1], dtype=np.float32))
    wavelet = 'haar'

    a, d = pywt.dwt(x, wavelet)
    result = wavelet_ops.dwt(x, wavelet)
    self.assertAllClose(a, result['a'])
    self.assertAllClose(d, result['d'])

    y = pywt.idwt(a, d, wavelet)
    result = wavelet_ops.idwt(result, wavelet)
    self.assertAllClose(y, result)


  @test_util.run_all_execution_modes
  def test_dwt_idwt_static_shape(self, wavelet='haar'):
    x = np.asarray([3, 7, 1, 1, -2, 5, 4, 6], dtype=np.float32)

    result = wavelet_ops.dwt(x, wavelet)
    self.assertAllEqual([4], result['a'].shape)
    self.assertAllClose([4], result['d'].shape)

    result = wavelet_ops.idwt(result, wavelet)
    self.assertAllEqual([8], result.shape)
