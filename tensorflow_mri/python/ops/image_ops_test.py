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

from absl.testing import parameterized

from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import test_util


class PeakSignalToNoiseRatioTest(test_util.TestCase):
  """Tests for PSNR op."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/image_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_2d_scalar(self):
    """Test 2D PSNR with scalar batch."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.psnr(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, 22.73803845)

    result = image_ops.psnr2d(img1, img2, max_val=255)
    self.assertAllClose(result, 22.73803845)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_2d_trivial_batch(self):
    """Test 2D PSNR with trivial batch of size 1."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    img1 = tf.expand_dims(img1, 0)
    img2 = tf.expand_dims(img2, 0)

    result = image_ops.psnr(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, [22.73803845])

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_2d_batch_multichannel(self):
    """Test 2D PSNR with multichannel batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    ref = [16.35598558,
           16.96981631,
           17.80788841,
           18.1842858,
           18.06558658,
           17.16817389]

    result = image_ops.psnr(img1, img2, max_val=255)
    self.assertAllClose(result, ref)

    # Test without specifying dynamic range, which should default to 255 for
    # `tf.uint8`.
    result = image_ops.psnr(img1, img2)
    self.assertAllClose(result, ref)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_2d_nd_batch(self):
    """Test 2D PSNR with N-D batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    img1 = tf.reshape(img1, (3, 2) + img1.shape[1:])
    img2 = tf.reshape(img2, (3, 2) + img2.shape[1:])
    ref = [[16.35598558, 16.96981631],
           [17.80788841, 18.18428580],
           [18.06558658, 17.16817389]]

    result = image_ops.psnr(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, ref)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_2d_batch_multichannel_float(self):
    """Test 2D PSNR with multichannel batch of floating point images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    ref = [16.35598558,
           16.96981631,
           17.80788841,
           18.1842858,
           18.06558658,
           17.16817389]

    img1 = tf.cast(img1, tf.float32) / 255.0
    img2 = tf.cast(img2, tf.float32) / 255.0

    result = image_ops.psnr(img1, img2, max_val=1)
    self.assertAllClose(result, ref)

    # Test without specifying dynamic range, which should default to 1 for
    # `tf.float32`.
    result = image_ops.psnr(img1, img2)
    self.assertAllClose(result, ref)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_3d_scalar(self):
    """Test 3D PSNR with scalar batch."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    img1 = img1[0, ...]
    img2 = img2[0, ...]

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.psnr(img1, img2, rank=3)
    self.assertAllClose(result, 32.3355765)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_3d_batch(self):
    """Test 3D PSNR with scalar batch."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    ref = [32.335575,
           31.898806,
           31.149742,
           34.818497,
           30.58971 ,
           32.17367 ]

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.psnr(img1, img2, max_val=255)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_3d_mdbatch(self):
    """Test 3D PSNR with multidimensional batch."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    ref = [[32.335575, 31.898806],
           [31.149742, 34.818497],
           [30.58971 , 32.17367 ]]

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    img1 = tf.reshape(img1, (3, 2) + img1.shape[1:])
    img2 = tf.reshape(img2, (3, 2) + img2.shape[1:])

    result = image_ops.psnr(img1, img2, max_val=255, rank=3)
    self.assertAllClose(result, ref, rtol=1e-3, atol=1e-3)

    result = image_ops.psnr3d(img1, img2, max_val=255)
    self.assertAllClose(result, ref, rtol=1e-3, atol=1e-3)

  @test_util.run_in_graph_and_eager_modes
  def test_psnr_3d_multichannel(self):
    """Test 3D PSNR with multichannel inputs."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    ref = [32.111702, 32.607716, 31.309875]

    img1 = tf.reshape(img1, (3, 2) + img1.shape[1:])
    img2 = tf.reshape(img2, (3, 2) + img2.shape[1:])

    img1 = tf.transpose(img1, [0, 2, 3, 4, 1])
    img2 = tf.transpose(img2, [0, 2, 3, 4, 1])

    result = image_ops.psnr(img1, img2, max_val=255, rank=3)
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)

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


class StructuralSimilarityTest(test_util.TestCase):
  """Tests for SSIM op."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/image_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_2d_scalar(self):
    """Test 2D SSIM with scalar batch."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.ssim(img1, img2, max_val=255, rank=2)

    self.assertAllClose(result, 0.5250339, rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_2d_trivial_batch(self):
    """Test 2D SSIM with trivial batch of size 1."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    img1 = tf.expand_dims(img1, 0)
    img2 = tf.expand_dims(img2, 0)

    result = image_ops.ssim(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, [0.5250339], rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_2d_batch_multichannel(self):
    """Test 2D SSIM with multichannel batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    ref = [0.250783,
           0.293936,
           0.33806 ,
           0.366984,
           0.38121 ,
           0.366342]

    result = image_ops.ssim(img1, img2, max_val=255)
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)

    # Test without specifying dynamic range, which should default to 255 for
    # `tf.uint8`.
    result = image_ops.ssim(img1, img2)
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_2d_nd_batch(self):
    """Test 2D SSIM with N-D batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    img1 = tf.reshape(img1, (3, 2) + img1.shape[1:])
    img2 = tf.reshape(img2, (3, 2) + img2.shape[1:])
    ref = [[0.250783, 0.293936],
           [0.33806 , 0.366984],
           [0.38121 , 0.366342]]

    result = image_ops.ssim(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_2d_batch_multichannel_float(self):
    """Test 2D SSIM with multichannel batch of floating point images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    ref = [0.250783,
           0.293936,
           0.33806 ,
           0.366984,
           0.38121 ,
           0.366342]

    img1 = tf.cast(img1, tf.float32) / 255.0
    img2 = tf.cast(img2, tf.float32) / 255.0

    result = image_ops.ssim(img1, img2, max_val=1)
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)

    # Test without specifying dynamic range, which should default to 1 for
    # `tf.float32`.
    result = image_ops.ssim(img1, img2)
    self.assertAllClose(result, ref, rtol=1e-4, atol=1e-4)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_3d_scalar(self):
    """Test 3D SSIM with scalar batch."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    img1 = img1[0, ...]
    img2 = img2[0, ...]

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.ssim(img1, img2, rank=3)
    self.assertAllClose(result, 0.93111473)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_3d_batch(self):
    """Test 3D SSIM with batch."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    ref = [0.93111473,
           0.90337730]

    img1 = img1[:2, ...]
    img2 = img2[:2, ...]

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.ssim(img1, img2, max_val=255)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_3d_mdbatch(self):
    """Test 3D SSIM with multidimensional batch."""
    img1 = self.data['psnr/3d/img1']
    img2 = self.data['psnr/3d/img2']

    ref = [[0.93111473, 0.90337730],
           [0.90820014, 0.92448730]]

    img1 = img1[:4, ...]
    img2 = img2[:4, ...]

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    img1 = tf.reshape(img1, (2, 2) + img1.shape[1:])
    img2 = tf.reshape(img2, (2, 2) + img2.shape[1:])

    result = image_ops.ssim(img1, img2, max_val=255, rank=3)
    self.assertAllClose(result, ref)

  @test_util.run_in_graph_and_eager_modes
  def test_ssim_3d_multichannel(self):
    """Test 3D SSIM with multichannel inputs."""
    # Does not work on CPU currently - GPU only.

    # img1 = self.data['psnr/3d/img1']
    # img2 = self.data['psnr/3d/img2']

    # ref = [[0.93111473, 0.90337730],
    #        [0.90820014, 0.92448730],
    #        [0.90630510, 0.92143655]]

    # img1 = tf.reshape(img1, (3, 2) + img1.shape[1:])
    # img2 = tf.reshape(img2, (3, 2) + img2.shape[1:])

    # img1 = tf.transpose(img1, [0, 2, 3, 4, 1])
    # img2 = tf.transpose(img2, [0, 2, 3, 4, 1])

    # result = image_ops.ssim(img1, img2, max_val=255, rank=3)
    # self.assertAllClose(result, tf.math.reduce_mean(ref, axis=1))

  def test_ssim_invalid_rank(self):
    """Test SSIM with an invalid rank."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "`rank` must be >= 2"):
      image_ops.ssim(img1, img2, 255)

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "`rank` must be >= 2"):
      image_ops.ssim(img1, img2, 255)


class MultiscaleStructuralSimilarityTest(test_util.TestCase):
  """Tests for MS-SSIM op."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/image_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_msssim_2d_scalar(self):
    """Test 2D MS-SSIM with scalar batch."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)

    result = image_ops.ssim_multiscale(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, 0.8270784)

    result = image_ops.ssim2d_multiscale(img1, img2, max_val=255)
    self.assertAllClose(result, 0.8270784)

  @test_util.run_in_graph_and_eager_modes
  def test_msssim_2d_trivial_batch(self):
    """Test 2D MS-SSIM with trivial batch of size 1."""
    img1 = self.data['psnr/2d/img1']
    img2 = self.data['psnr/2d/img2']

    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, -1)
    img1 = tf.expand_dims(img1, 0)
    img2 = tf.expand_dims(img2, 0)

    result = image_ops.ssim_multiscale(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, [0.8270784])

  @test_util.run_in_graph_and_eager_modes
  def test_msssim_2d_batch_multichannel(self):
    """Test 2D MS-SSIM with multichannel batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    ref = [0.47854424,
           0.60964876,
           0.71863150,
           0.76113180,
           0.77840980,
           0.71724670]

    result = image_ops.ssim_multiscale(img1, img2, max_val=255)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

    # Test without specifying dynamic range, which should default to 255 for
    # `tf.uint8`.
    result = image_ops.ssim_multiscale(img1, img2)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_msssim_2d_nd_batch(self):
    """Test 2D MS-SSIM with N-D batch of images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    img1 = tf.reshape(img1, (3, 2) + img1.shape[1:])
    img2 = tf.reshape(img2, (3, 2) + img2.shape[1:])
    ref = [[0.47854424, 0.60964876],
           [0.71863150, 0.76113180],
           [0.77840980, 0.71724670]]

    result = image_ops.ssim_multiscale(img1, img2, max_val=255, rank=2)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

    result = image_ops.ssim2d_multiscale(img1, img2, max_val=255)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_msssim_2d_batch_multichannel_float(self):
    """Test 2D MS-SSIM with multichannel batch of floating point images."""
    img1 = self.data['psnr/2d/batch/img1']
    img2 = self.data['psnr/2d/batch/img2']
    ref = [0.47854424,
           0.60964876,
           0.71863150,
           0.76113180,
           0.77840980,
           0.71724670]

    img1 = tf.cast(img1, tf.float32) / 255.0
    img2 = tf.cast(img2, tf.float32) / 255.0

    result = image_ops.ssim_multiscale(img1, img2, max_val=1)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

    # Test without specifying dynamic range, which should default to 1 for
    # `tf.float32`.
    result = image_ops.ssim_multiscale(img1, img2)
    self.assertAllClose(result, ref, rtol=1e-5, atol=1e-5)

  @test_util.run_in_graph_and_eager_modes
  def test_msssim_3d_scalar(self):
    """Test 3D MS-SSIM with scalar batch."""
    # Kills testing hardware.

    # img1 = self.data['psnr/3d/img1']
    # img2 = self.data['psnr/3d/img2']

    # def upsample_3d(img, scale):
    #   img = tf.repeat(img, scale, axis=1)
    #   img = tf.repeat(img, scale, axis=2)
    #   img = tf.repeat(img, scale, axis=3)
    #   return img
    # img1 = upsample_3d(img1, 3)
    # img2 = upsample_3d(img2, 3)

    # img1 = img1[0, ...]
    # img2 = img2[0, ...]

    # img1 = tf.expand_dims(img1, -1)
    # img2 = tf.expand_dims(img2, -1)

    # result = image_ops.ssim_multiscale(img1, img2, rank=3)

    # self.assertAllClose(result, 0.96301770)

  def test_msssim_input_size_error(self):
    """Test MS-SSIM with an invalid rank."""
    img1 = tf.zeros((4, 160, 160, 1))
    img2 = tf.zeros((4, 160, 160, 1))

    with self.assertRaisesRegex(
        tf.errors.InvalidArgumentError,
        "spatial dimensions must have size of at least 161"):
      image_ops.ssim_multiscale(img1, img2)

    img1 = tf.zeros((4, 161, 161, 1))
    img2 = tf.zeros((4, 161, 161, 1))

    image_ops.ssim_multiscale(img1, img2)


class CentralCropTest(test_util.TestCase):
  """Tests for central cropping operation."""
  # pylint: disable=missing-function-docstring

  @test_util.run_in_graph_and_eager_modes
  def test_cropping(self):
    """Test cropping."""
    shape = [2, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[6, 7], [10, 11]])

    y_tf = image_ops.central_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_cropping_unknown_dim(self):
    """Test cropping with an unknown dimension."""
    shape = [-1, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[2, 3], [6, 7], [10, 11], [14, 15]])

    y_tf = image_ops.central_crop(x_np, shape)

    self.assertAllEqual(y_tf, y_np)


class SymmetricPadOrCropTest(test_util.TestCase):
  """Tests for symmetric padding/cropping operation."""
  # pylint: disable=missing-function-docstring

  @test_util.run_in_graph_and_eager_modes
  def test_cropping(self):
    """Test cropping."""
    shape = [2, 2]
    x_np = np.array([[1, 2, 3, 4],
                     [5, 6, 7, 8],
                     [9, 10, 11, 12],
                     [13, 14, 15, 16]])
    y_np = np.array([[6, 7], [10, 11]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_padding(self):
    """Test padding."""
    shape = [4, 4]
    x_np = np.array([[1, 2], [3, 4]])
    y_np = np.array([[0, 0, 0, 0],
                     [0, 1, 2, 0],
                     [0, 3, 4, 0],
                     [0, 0, 0, 0]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_padding_non_default_mode(self):
    """Test padding."""
    shape = [7]
    x_np = np.array([1, 2, 3])
    y_np = np.array([3, 2, 1, 2, 3, 2, 1])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape,
                                             padding_mode='reflect')

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_padding_cropping(self):
    """Test combined cropping and padding."""
    shape = [1, 5]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[0, 4, 5, 6, 0]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_padding_cropping_unknown_dimension(self):
    """Test combined cropping and padding with an unknown dimension."""
    shape = [1, -1]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[4, 5, 6]])

    y_tf = image_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  def test_static_shape(self):
    """Test static shapes."""
    def get_fn(target_shape):
      return lambda x: image_ops.resize_with_crop_or_pad(x, target_shape)

    self._assert_static_shape(get_fn([1, -1]), [None, 3], [1, 3])
    self._assert_static_shape(get_fn([-1, -1]), [None, 3], [None, 3])
    self._assert_static_shape(get_fn([-1, 5]), [None, 3], [None, 5])
    self._assert_static_shape(get_fn([5, 5]), [None, None], [5, 5])
    self._assert_static_shape(get_fn([-1, -1]), [None, None], [None, None])
    self._assert_static_shape(
        get_fn([144, 144, 144, -1]), [None, None, None, 1], [144, 144, 144, 1])

  def _assert_static_shape(self, fn, input_shape, expected_output_shape):
    """Asserts that function returns the expected static shapes."""
    @tf.function
    def graph_fn(x):
      return fn(x)

    input_spec = tf.TensorSpec(shape=input_shape)
    concrete_fn = graph_fn.get_concrete_function(input_spec)

    self.assertAllEqual(concrete_fn.structured_outputs.shape,
                        expected_output_shape)


class TotalVariationTest(test_util.TestCase):
  """Tests for operation `total_variation`."""

  @test_util.run_in_graph_and_eager_modes
  def test_total_variation(self):
    """Test total variation."""
    # Example image.
    img = [[1, 2, 4, 4],
           [4, 7, 2, 1],
           [8, 2, 4, 3],
           [2, 2, 1, 6]]
    # The following are the sum of absolute differences between the pixels.
    # sum row dif = (4-1) + (8-4) + (8-2) + (7-2) + (7-2) + (2-2)
    #             + (4-2) + (4-2) + (4-1) + (4-1) + (3-1) + (6-3)
    #             = (3 + 4 + 6) + (5 + 5 + 0) + (2 + 2 + 3) + (3 + 2 + 3)
    #             = 13 + 10 + 7 + 8 = 38
    # sum col dif = (2-1) + (4-2) + (4-4) + (7-4) + (7-2) + (2-1)
    #             + (8-2) + (4-2) + (4-3) + (2-2) + (2-1) + (6-1)
    #             = (1 + 2 + 0) + (3 + 5 + 1) + (6 + 2 + 1) + 0 + 1 + 5 =
    #             = 3 + 9 + 9 + 6

    result = image_ops.total_variation(img)
    self.assertAllClose(result, 65)

    result = image_ops.total_variation(img, axis=0)
    self.assertAllClose(result, [13, 10, 7, 8])

    result = image_ops.total_variation(img, axis=1)
    self.assertAllClose(result, [3, 9, 9, 6])

    # Test with `keepdims=True`.
    result = image_ops.total_variation(img, axis=0, keepdims=True)
    self.assertAllClose(result, tf.reshape([13, 10, 7, 8], [1, 4]))

    result = image_ops.total_variation(img, axis=1, keepdims=True)
    self.assertAllClose(result, tf.reshape([3, 9, 9, 6], [4, 1]))

    # Test float by scaling pixel values. Total variation scales as well.
    result = image_ops.total_variation(1.25 * np.array(img))
    self.assertAllClose(result, 1.25 * 65)

    # Test complex image.
    result = image_ops.total_variation(tf.dtypes.complex(
        1.25 * np.array(img), 1.5 * np.array(img)))
    self.assertAllClose(result, np.sqrt((1.25 * 65) ** 2 + (1.5 * 65) ** 2))


class ExtractGlimpsesTest(test_util.TestCase):
  """Tests for the `extract_glimpses` operation."""

  @test_util.run_in_graph_and_eager_modes
  def test_extract_glimpses(self):
    """Test `extract_glimpses` operation."""
    images = tf.reshape(tf.range(40), [1, 4, 5, 2])
    sizes = [2, 3]
    offsets = [[2, 2], [0, 1]]
    expected = [[[24, 25, 26, 27, 28, 29, 34, 35, 36, 37, 38, 39],
                 [2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17]]]

    patches = image_ops.extract_glimpses(images, sizes, offsets)
    self.assertAllEqual(patches, expected)


class PhantomTest(test_util.TestCase):
  """Tests for `phantom` op."""

  @classmethod
  def setUpClass(cls):
    """Prepare tests."""
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/phantoms.h5')

  @parameterized.parameters('shepp_logan', 'modified_shepp_logan')
  @test_util.run_in_graph_and_eager_modes
  def test_shepp_logan(self, phantom_type):
    """Test 2D Shepp-Logan phantom against MATLAB results."""
    expected = self.data[phantom_type + '/2d']
    result = image_ops.phantom(phantom_type=phantom_type)
    self.assertAllClose(result, expected)

  @parameterized.parameters('kak_roberts', 'modified_kak_roberts')
  @test_util.run_in_graph_and_eager_modes
  def test_kak_roberts(self, phantom_type):
    """Test 3D Kak-Roberts phantom against saved results."""
    expected = self.data[phantom_type + '/3d']
    result = image_ops.phantom(phantom_type=phantom_type, shape=[128, 128, 128])
    self.assertAllClose(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_default_2d(self):
    """Test 2D default."""
    expected = self.data['modified_shepp_logan/2d']
    result = image_ops.phantom()
    self.assertAllClose(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_default_3d(self):
    """Test 3D default."""
    expected = self.data['modified_kak_roberts/3d']
    result = image_ops.phantom(shape=[128, 128, 128])
    self.assertAllClose(result, expected)

  @parameterized.product(rank=[2, 3],
                         dtype=[tf.float32, tf.complex64])
  @test_util.run_in_graph_and_eager_modes
  def test_parallel_imaging(self, rank, dtype): # pylint: disable=missing-param-doc
    """Test parallel imaging phantom."""
    image, sens = image_ops.phantom(shape=[64] * rank,
                                    num_coils=12,
                                    dtype=dtype,
                                    return_sensitivities=True)

    sens_ref = image_ops._birdcage_sensitivities([64] * rank, 12, dtype=dtype) # pylint: disable=protected-access

    image_ref = image_ops.phantom(shape=[64] * rank, dtype=dtype) * sens
    self.assertAllClose(image, image_ref)
    self.assertAllClose(sens, sens_ref)

  @parameterized.product(shape=[[32, 32], [128, 64], [32, 32, 32]],
                         num_coils=[4, 6],
                         birdcage_radius=[1.5, 1.3],
                         num_rings=[2])
  @test_util.run_in_graph_and_eager_modes
  def test_birdcage_sensitivities(self, # pylint: disable=missing-param-doc
                                  shape,
                                  num_coils,
                                  birdcage_radius,
                                  num_rings):
    """Test birdcage sensitivities."""
    tf_sens = image_ops._birdcage_sensitivities(shape, # pylint: disable=protected-access
                                                num_coils,
                                                birdcage_radius=birdcage_radius,
                                                num_rings=num_rings)

    np_sens = self._np_birdcage_sensitivities(
        [num_coils] + shape, r=birdcage_radius,
        nzz=np.ceil(num_coils / num_rings))

    self.assertAllClose(tf_sens, np_sens, rtol=1e-4, atol=1e-4)

  def _np_birdcage_sensitivities(self, shape, r=1.5, nzz=8, dtype=np.complex64): # pylint: disable=missing-param-doc
    """Simulate birdcage coil sensitivities.

    Implementation from:
    https://github.com/mikgroup/sigpy/blob/v0.1.23/sigpy/mri/sim.py
    """
    if len(shape) == 3:

      nc, ny, nx = shape
      c, y, x = np.mgrid[:nc, :ny, :nx]

      coilx = r * np.cos(c * (2 * np.pi / nc))
      coily = r * np.sin(c * (2 * np.pi / nc))
      coil_phs = -c * (2 * np.pi / nc)

      x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
      y_co = (y - ny / 2.0) / (ny / 2.0) - coily
      rr = np.sqrt(x_co ** 2 + y_co ** 2)
      phi = np.arctan2(x_co, -y_co) + coil_phs
      out = (1.0 / rr) * np.exp(1j * phi)

    elif len(shape) == 4:
      nc, nz, ny, nx = shape
      c, z, y, x = np.mgrid[:nc, :nz, :ny, :nx]

      coilx = r * np.cos(c * (2 * np.pi / nzz))
      coily = r * np.sin(c * (2 * np.pi / nzz))
      coilz = np.floor(c / nzz) - 0.5 * (np.ceil(nc / nzz) - 1)
      coil_phs = -(c + np.floor(c / nzz)) * (2 * np.pi / nzz)

      x_co = (x - nx / 2.0) / (nx / 2.0) - coilx
      y_co = (y - ny / 2.0) / (ny / 2.0) - coily
      z_co = (z - nz / 2.0) / (nz / 2.0) - coilz
      rr = (x_co**2 + y_co**2 + z_co**2)**0.5
      phi = np.arctan2(x_co, -y_co) + coil_phs
      out = (1 / rr) * np.exp(1j * phi)
    else:
      raise ValueError('Can only generate shape with length 3 or 4')

    rss = sum(abs(out) ** 2, 0)**0.5
    out /= rss

    return out.astype(dtype)


if __name__ == '__main__':
  tf.test.main()
