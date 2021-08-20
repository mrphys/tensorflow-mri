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
"""Tests for module `image_ops`."""

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.utils import io_utils


class PeakSignalToNoiseRatioTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_utils.read_hdf5('tests/data/image_ops_data.h5')


  def test_psnr_2d_scalar(self):
    """Test 2D PSNR with scalar batch."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.psnr(img1, img2, 255, rank=2)

    self.assertAllClose(result, 22.73803845)


  def test_psnr_2d_trivial_batch(self):
    """Test 2D PSNR with trivial batch of size 1."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    img1 = tf.expand_dims(img1, 0)
    img2 = tf.expand_dims(img2, 0)

    result = image_ops.psnr(img1, img2, 255, rank=2)

    self.assertAllClose(result, [22.73803845])


  def test_psnr_2d_batch_multichannel(self):
    """Test 2D PSNR with multichannel batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']

    result = image_ops.psnr(img1, img2, 255)
    ref = [16.35598558,
           16.96981631,
           17.80788841,
           18.1842858,
           18.06558658,
           17.16817389]

    self.assertAllClose(result, ref)


  def test_psnr_invalid_rank(self):
    """Test PSNR with an invalid rank."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "`rank` must be >= 2"):
      image_ops.psnr(img1, img2, 255)

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "`rank` must be >= 2"):
      image_ops.psnr(img1, img2, 255)


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

    y_tf = image_ops.central_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_cropping_2(self):

    shape = [-1, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[2, 3], [6, 7], [10, 11], [14, 15]])

    y_tf = image_ops.central_crop(x_np, shape)

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

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_padding(self):

    shape = [4, 4]
    x_np = np.array([[1, 2], [3, 4]])
    y_np = np.array([[0, 0, 0, 0],
                     [0, 1, 2, 0],
                     [0, 3, 4, 0],
                     [0, 0, 0, 0]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_padding_cropping(self):

    shape = [1, 5]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[0, 4, 5, 6, 0]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


  def test_padding_cropping_2(self):

    shape = [1, -1]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[4, 5, 6]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


if __name__ == '__main__':
  tf.test.main()
