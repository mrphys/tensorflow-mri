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
"""Tests for FFT ops."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops


class CentralCropTest(tf.test.TestCase):
  """Tests for central cropping operation."""
  # pylint: disable=missing-function-docstring

  def test_cropping(self):

    shape = [2, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[6, 7], [10, 11]])

    y_tf = array_ops.central_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_cropping_2(self):

    shape = [-1, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[2, 3], [6, 7], [10, 11], [14, 15]])

    y_tf = array_ops.central_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


class SymmetricPadOrCropTest(tf.test.TestCase):
  """Tests for symmetric padding/cropping operation."""
  # pylint: disable=missing-function-docstring

  def test_cropping(self):

    shape = [2, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[6, 7], [10, 11]])

    y_tf = array_ops.symmetric_pad_or_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_padding(self):

    shape = [4, 4]
    x_np = np.array([[1, 2], [3, 4]])
    y_np = np.array([[0, 0, 0, 0],
                     [0, 1, 2, 0],
                     [0, 3, 4, 0],
                     [0, 0, 0, 0]])

    y_tf = array_ops.symmetric_pad_or_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_padding_cropping(self):

    shape = [1, 5]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[0, 4, 5, 6, 0]])

    y_tf = array_ops.symmetric_pad_or_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_padding_cropping_2(self):

    shape = [1, -1]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[4, 5, 6]])

    y_tf = array_ops.symmetric_pad_or_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


if __name__ == '__main__':
  tf.test.main()
