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
"""Tests for `ResizeAndConcatenate` layers."""

import tensorflow as tf

from tensorflow_mri.python.layers import concatenate
from tensorflow_mri.python.util import test_util


class ResizeAndConcatenateTest(test_util.TestCase):
  """Tests for layer `ResizeAndConcatenate`."""
  def test_resize_and_concatenate(self):
    """Test `ResizeAndConcatenate` layer."""
    # Test data.
    x1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    x2 = tf.constant([[5.0], [6.0]])

    # Test concatenation along axis 1.
    layer = concatenate.ResizeAndConcatenate(axis=-1)

    result = layer([x1, x2])
    self.assertAllClose([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]], result)

    result = layer([x2, x1])
    self.assertAllClose([[5.0, 1.0, 2.0], [6.0, 3.0, 4.0]], result)

    # Test concatenation along axis 0.
    layer = concatenate.ResizeAndConcatenate(axis=0)

    result = layer([x1, x2])
    self.assertAllClose(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [6.0, 0.0]], result)

    result = layer([x2, x1])
    self.assertAllClose([[5.0], [6.0], [1.0], [3.0]], result)


if __name__ == '__main__':
  tf.test.main()
