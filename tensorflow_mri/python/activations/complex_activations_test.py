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
"""Tests for module `complex_activations`."""

import tensorflow as tf

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.util import test_util


class ReluTest(test_util.TestCase):
  @test_util.run_all_execution_modes
  def test_complex_relu(self):
    inputs = [1.0 - 2.0j, 1.0 + 3.0j, -2.0 + 1.0j, -3.0 - 4.0j]
    expected = [1.0 + 0.0j, 1.0 + 3.0j, 0.0 + 1.0j, 0.0 + 0.0j]
    result = complex_activations.complex_relu(inputs)
    self.assertAllClose(expected, result)


if __name__ == '__main__':
  tf.test.main()
