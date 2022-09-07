# ==============================================================================
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

# ==============================================================================
# Copyright (c) 2006-2012 Filip Wasilewski <http://en.ig.ma/>
# Copyright (c) 2012-2020 The PyWavelets Developers <https://github.com/PyWavelets/pywt>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
"""Tests for module `wavelet_ops`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import warnings

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import pywt

from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import test_util


# determine which wavelets to test
WAVELIST = pywt.wavelist()
if 'dmey' in WAVELIST:
  # accuracy is very low for dmey, so omit it
  WAVELIST.remove('dmey')

# removing wavelets with dwt_possible == False
del_list = []
for w in WAVELIST:
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', FutureWarning)
    if not isinstance(pywt.DiscreteContinuousWavelet(w),
                      pywt.Wavelet):
      del_list.append(w)
for del_ind in del_list:
  WAVELIST.remove(del_ind)


class DiscreteWaveletTransformTest(test_util.TestCase):
  """Tests for wavelet operators."""
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
  def test_dwt_idwt_double(self):
    x = np.asarray([3, 7, 1, 1, -2, 5, 4, 6], dtype=np.float64)
    wavelet = 'haar'

    a, d = pywt.dwt(x, wavelet)
    result = wavelet_ops.dwt(x, wavelet)
    self.assertAllClose(a, result['a'])
    self.assertAllClose(d, result['d'])

    y = pywt.idwt(a, d, wavelet)
    result = wavelet_ops.idwt(result, wavelet)
    self.assertAllClose(y, result)


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


  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      # name, shape, wavelet, level, axes, dtype
      ("test0", [16], 'haar', 2, None, 'float32'),
      ("test1", [16], 'haar', 2, None, 'complex64'),
      ("test2", [8, 8], 'db2', 1, None, 'float32'),
      ("test3", [8, 8], 'db2', None, 1, 'float32'),
  )
  def test_wavedec_waverec(self, shape, wavelet, level, axes, dtype):
    x = np.random.uniform(size=shape).astype(dtype)

    expected = pywt.wavedecn(x, wavelet, level=level, axes=axes)
    result = wavelet_ops.wavedec(x, wavelet, level=level, axes=axes)

    self.assertAllClose(expected, result)

    expected = pywt.waverecn(result, wavelet, axes=axes)
    result = wavelet_ops.waverec(result, wavelet, axes=axes)
    self.assertAllClose(expected, result)


  @parameterized.named_parameters(
      # name, shape, wavelet, axes, expected
      ("test0", [16, 20], 'haar', None, None),
      ("test1", [64, 32], 'db2', None, 3),
      ("test2", [64, 32], 4, None, 3),
      ("test3", 32, 'db2', None, 3),
  )
  def test_dwt_max_level(self, shape, wavelet, axes, expected):
    if expected is None:
      expected = pywt.dwtn_max_level(shape, wavelet, axes=axes)
    result = wavelet_ops.dwt_max_level(shape, wavelet, axes=axes)
    self.assertEqual(expected, result)


class CoeffsToArrayTest(test_util.TestCase):
  def test_coeffs_to_array(self):
    # single element list returns the first element
    a_coeffs = [tf.reshape(tf.range(8), (2, 4))]
    arr, arr_slices = wavelet_ops.coeffs_to_tensor(a_coeffs)
    self.assertAllClose(arr, a_coeffs[0])
    self.assertAllClose(arr, arr[arr_slices[0]])

    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor, [])
    # invalid second element:  array as in wavedec, but not 1D
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor,
                      [a_coeffs[0], ] * 2)
    # invalid second element:  tuple as in wavedec2, but not a 3-tuple
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor,
                      [a_coeffs[0], (a_coeffs[0], )])

    # coefficients as None is not supported
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor, [None, ])
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor,
                      [a_coeffs, (None, None, None)])

    # invalid type for second coefficient list element
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor,
                      [a_coeffs, None])

    # use an invalid key name in the coef dictionary
    coeffs = [np.array([0]), dict(d=np.array([0]), c=np.array([0]))]
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor, coeffs)


  def test_wavedecn_coeff_reshape_even(self):
    # verify round trip is correct:
    #   wavedec - >coeffs_to_tensor-> tensor_to_coeffs -> waverec
    # This is done for wavedec
    rng = np.random.RandomState(1234)
    n = 28
    x1 = rng.randn(*([n, n, n]))
    for mode in ['symmetric']:
      for wave in WAVELIST:
        with self.subTest(mode=mode, wave=wave):
          wave = pywt.Wavelet(wave)
          maxlevel = wavelet_ops.dwt_max_level(np.min(x1.shape), wave.dec_len)
          if maxlevel == 0:
            continue

          coeffs = wavelet_ops.wavedec(x1, wave, mode=mode)
          coeff_arr, coeff_slices = wavelet_ops.coeffs_to_tensor(coeffs)
          coeffs2 = wavelet_ops.tensor_to_coeffs(coeff_arr, coeff_slices)
          x1r = wavelet_ops.waverec(coeffs2, wave, mode=mode)

          self.assertAllClose(x1, x1r, rtol=1e-4, atol=1e-4)


  def test_wavedecn_coeff_reshape_axes_subset(self):
    # verify round trip is correct when only a subset of axes are transformed:
    #   wavedec - >coeffs_to_tensor-> tensor_to_coeffs -> waverec
    # This is done for wavedec{1, 2, n}
    rng = np.random.RandomState(1234)
    mode = 'symmetric'
    wave = pywt.Wavelet('db2')
    n = 16
    ndim = 3
    for axes in [(-1, ), (0, ), (1, ), (0, 1), (1, 2), (0, 2), None]:
      with self.subTest(axes=axes):
        x1 = rng.randn(*([n] * ndim))
        coeffs = wavelet_ops.wavedec(x1, wave, mode=mode, axes=axes)
        coeff_arr, coeff_slices = wavelet_ops.coeffs_to_tensor(
            coeffs, axes=axes)
        if axes is not None:
          # if axes is not None, it must be provided to coeffs_to_tensor
          self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor, coeffs)

        # mismatched axes size
        self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor, coeffs,
                          axes=(0, 1, 2, 3))
        self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor, coeffs,
                          axes=())

        coeffs2 = pywt.array_to_coeffs(coeff_arr, coeff_slices)
        x1r = wavelet_ops.waverec(coeffs2, wave, mode=mode, axes=axes)

        self.assertAllClose(x1, x1r, rtol=1e-4, atol=1e-4)


  def test_coeffs_to_array_padding(self):
    rng = np.random.RandomState(1234)
    x1 = rng.randn(32, 32)
    mode = 'symmetric'
    coeffs = wavelet_ops.wavedec(x1, 'db2', mode=mode)

    # padding=None raises a ValueError when tight packing is not possible
    self.assertRaises(ValueError, wavelet_ops.coeffs_to_tensor,
                      coeffs, padding=None)

    # set padded values to nan
    coeff_arr, _ = wavelet_ops.coeffs_to_tensor(coeffs, padding=np.nan)
    npad = np.sum(np.isnan(coeff_arr))
    self.assertTrue(npad > 0)

    # pad with zeros
    coeff_arr, _ = wavelet_ops.coeffs_to_tensor(coeffs, padding=0)
    self.assertTrue(np.sum(np.isnan(coeff_arr)) == 0)
    self.assertTrue(np.sum(coeff_arr == 0) == npad)

    # Haar case with N as a power of 2 can be tightly packed
    coeffs_haar = wavelet_ops.wavedec(x1, 'haar', mode=mode)
    coeff_arr, _ = wavelet_ops.coeffs_to_tensor(coeffs_haar, padding=None)
    # shape of coeff_arr will match in this case, but not in general
    self.assertEqual(coeff_arr.shape, x1.shape)


  def test_waverecn_coeff_reshape_odd(self):
    # verify round trip is correct:
    #   wavedec - >coeffs_to_tensor-> tensor_to_coeffs -> waverec
    rng = np.random.RandomState(1234)
    x1 = rng.randn(35, 33)
    for mode in ['symmetric']:
      for wave in ['haar']:
        with self.subTest(mode=mode, wave=wave):
          wave = pywt.Wavelet(wave)
          maxlevel = wavelet_ops.dwt_max_level(np.min(x1.shape), wave.dec_len)
          if maxlevel == 0:
            continue
          coeffs = wavelet_ops.wavedec(x1, wave, mode=mode)
          coeff_arr, coeff_slices = wavelet_ops.coeffs_to_tensor(coeffs)
          coeffs2 = wavelet_ops.tensor_to_coeffs(coeff_arr, coeff_slices)
          x1r = wavelet_ops.waverec(coeffs2, wave, mode=mode)
          # truncate reconstructed values to original shape
          x1r = x1r[tuple(slice(s) for s in x1.shape)]
          self.assertAllClose(x1, x1r, rtol=1e-4, atol=1e-4)


  def test_array_to_coeffs_invalid_inputs(self):
    coeffs = wavelet_ops.wavedec(np.ones(2), 'haar')
    arr, _ = wavelet_ops.coeffs_to_tensor(coeffs)

    # empty list of array slices
    self.assertRaises(ValueError, wavelet_ops.tensor_to_coeffs, arr, [])


if __name__ == '__main__':
  tf.test.main()
