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
"""Tests for normalization layers."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.layers import normalization
from tensorflow_mri.python.util import test_util


class NormalizedTest(test_util.TestCase):
  @test_util.run_all_execution_modes
  def test_normalized_dense(self):
    layer = normalization.Normalized(
        tf.keras.layers.Dense(2, bias_initializer='random_uniform'))
    layer.build((None, 4))

    input_data = np.random.uniform(size=(2, 4))

    def _compute_output(input_data, normalized=False):
      if normalized:
        mean = input_data.mean(axis=-1, keepdims=True)
        std = input_data.std(axis=-1, keepdims=True)
        input_data = (input_data - mean) / std
      output_data = layer.layer(input_data)
      if normalized:
        output_data = output_data * std + mean
      return output_data

    expected_unnorm = _compute_output(input_data, normalized=False)
    expected_norm = _compute_output(input_data, normalized=True)

    result_unnorm = layer.layer(input_data)
    result_norm = layer(input_data)

    self.assertAllClose(expected_unnorm, result_unnorm)
    self.assertAllClose(expected_norm, result_norm)


if __name__ == '__main__':
  tf.test.main()
