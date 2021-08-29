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
"""Tests for module `array_ops`."""

import tensorflow as tf

from tensorflow_mri.python.ops import array_ops


class RavelMultiIndexTest(tf.test.TestCase):
  """Tests for the `ravel_multi_index` op."""

  def test_ravel_multi_index_2d(self):
    """Test multi-index ravelling (2D, 1D indices array)."""
    indices = [[0, 0], [0, 1], [2, 2], [3, 1]]
    expected = [0, 1, 10, 13]

    result = array_ops.ravel_multi_index(indices, [4, 4])
    self.assertAllEqual(result, expected)

  def test_ravel_multi_index_2d_scalar(self):
    """Test multi-index ravelling (2D, scalar index)."""
    indices = [2, 2]
    expected = 12

    result = array_ops.ravel_multi_index(indices, [4, 5])
    self.assertAllEqual(result, expected)

  def test_ravel_multi_index_2d_batch(self):
    """Test multi-index ravelling (2D, 2D indices array)."""
    indices = [[[0, 0], [0, 1], [2, 2], [3, 1]],
               [[0, 2], [2, 0], [1, 3], [3, 3]]]
    expected = [[0, 1, 10, 13],
                [2, 8, 7, 15]]

    result = array_ops.ravel_multi_index(indices, [4, 4])
    self.assertAllEqual(result, expected)

  def test_ravel_multi_index_3d(self):
    """Test multi-index ravelling (3D, 1D indices array)."""
    indices = [[0, 0, 0], [0, 1, 1], [2, 2, 3], [3, 1, 2]]
    expected = [0, 5, 43, 54]

    result = array_ops.ravel_multi_index(indices, [4, 4, 4])
    self.assertAllEqual(result, expected)

  def test_ravel_multi_index_3d_scalar(self):
    """Test multi-index ravelling (3D, scalar index)."""
    indices = [2, 2, 1]
    expected = 41

    result = array_ops.ravel_multi_index(indices, [4, 4, 4])
    self.assertAllEqual(result, expected)


if __name__ == '__main__':
  tf.test.main()