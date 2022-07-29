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
"""Tests for module `linear_operator_wavelet`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_wavelet
from tensorflow_mri.python.ops import wavelet_ops
from tensorflow_mri.python.util import test_util


class LinearOperatorWaveletTest(test_util.TestCase):
  @parameterized.named_parameters(
      # name, wavelet, level, axes, domain_shape, range_shape
      ("test0", "haar", None, None, [6, 6], [7, 7]),
      ("test1", "haar", 1, None, [6, 6], [6, 6]),
      ("test2", "haar", None, -1, [6, 6], [6, 7]),
      ("test3", "haar", None, [-1], [6, 6], [6, 7])
  )
  def test_general(self, wavelet, level, axes, domain_shape, range_shape):
    # Instantiate.
    linop = linear_operator_wavelet.LinearOperatorWavelet(
        domain_shape, wavelet=wavelet, level=level, axes=axes)

    # Example data.
    data = np.arange(np.prod(domain_shape)).reshape(domain_shape)
    data = data.astype("float32")

    # Forward and adjoint.
    expected_forward, coeff_slices = wavelet_ops.coeffs_to_tensor(
        wavelet_ops.wavedec(data, wavelet=wavelet, level=level, axes=axes),
        axes=axes)
    expected_adjoint = wavelet_ops.waverec(
        wavelet_ops.tensor_to_coeffs(expected_forward, coeff_slices),
        wavelet=wavelet, axes=axes)

    # Test shapes.
    self.assertAllClose(domain_shape, linop.domain_shape)
    self.assertAllClose(domain_shape, linop.domain_shape_tensor())
    self.assertAllClose(range_shape, linop.range_shape)
    self.assertAllClose(range_shape, linop.range_shape_tensor())

    # Test transform.
    result_forward = linop.transform(data)
    result_adjoint = linop.transform(result_forward, adjoint=True)
    self.assertAllClose(expected_forward, result_forward)
    self.assertAllClose(expected_adjoint, result_adjoint)

  def test_with_batch_inputs(self):
    """Test batch shape."""
    axes = [-2, -1]
    data = np.arange(4 * 8 * 8).reshape(4, 8, 8).astype("float32")
    linop = linear_operator_wavelet.LinearOperatorWavelet(
        (8, 8), wavelet="haar", level=1)

    # Forward and adjoint.
    expected_forward, coeff_slices = wavelet_ops.coeffs_to_tensor(
        wavelet_ops.wavedec(data, wavelet='haar', level=1, axes=axes),
        axes=axes)
    expected_adjoint = wavelet_ops.waverec(
        wavelet_ops.tensor_to_coeffs(expected_forward, coeff_slices),
        wavelet='haar', axes=axes)

    result_forward = linop.transform(data)
    self.assertAllClose(expected_forward, result_forward)

    result_adjoint = linop.transform(result_forward, adjoint=True)
    self.assertAllClose(expected_adjoint, result_adjoint)


if __name__ == '__main__':
  tf.test.main()
