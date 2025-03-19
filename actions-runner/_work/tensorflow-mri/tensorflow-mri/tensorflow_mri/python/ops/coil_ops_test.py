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
"""Tests for module `coil_ops`."""

import itertools

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.ops import coil_ops
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import io_util
from tensorflow_mri.python.util import test_util

# Many tests on this file have high tolerance for numerical errors, likely due
# to issues with `tf.linalg.svd`. TODO: come up with a better solution.

class SensMapsTest(test_util.TestCase):
  """Tests for ops related to estimation of coil sensitivity maps."""

  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/coil_ops_data.h5')

  @test_util.run_in_graph_and_eager_modes
  def test_walsh(self):
    """Test Walsh's method."""
    # GPU results are close, but about 1-2% of values show deviations up to
    # 1e-3. This is probably related to TF issue:
    # https://github.com/tensorflow/tensorflow/issues/45756
    # In the meantime, we run these tests on the CPU only. Same applies to all
    # other tests in this class.
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        self.data['images'], method='walsh')

    self.assertAllClose(maps, self.data['maps/walsh'], rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_walsh_transposed(self):
    """Test Walsh's method with a transposed array."""
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        tf.transpose(self.data['images'], [2, 0, 1]),
        coil_axis=0, method='walsh')

    self.assertAllClose(maps, tf.transpose(self.data['maps/walsh'], [2, 0, 1]),
                        rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_inati(self):
    """Test Inati's method."""
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        self.data['images'], method='inati')

    self.assertAllClose(maps, self.data['maps/inati'], rtol=1e-4, atol=1e-4)

  # TODO(jmontalt): Look into accuracy issues and re-enable these tests.
  # @test_util.run_in_graph_and_eager_modes
  # def test_espirit(self):
  #   """Test ESPIRiT method."""
  #   with tf.device('/cpu:0'):
  #     maps = coil_ops.estimate_coil_sensitivities(
  #       self.data['kspace'], method='espirit')

  #   self.assertAllClose(maps, self.data['maps/espirit'], rtol=1e-2, atol=1e-2)

  # @test_util.run_in_graph_and_eager_modes
  # def test_espirit_transposed(self):
  #   """Test ESPIRiT method with a transposed array."""
  #   with tf.device('/cpu:0'):
  #     maps = coil_ops.estimate_coil_sensitivities(
  #       tf.transpose(self.data['kspace'], [2, 0, 1]),
  #       coil_axis=0, method='espirit')

  #   self.assertAllClose(
  #       maps, tf.transpose(self.data['maps/espirit'], [2, 0, 1, 3]),
  #       rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_walsh_3d(self):
    """Test Walsh method with 3D image."""
    with tf.device('/cpu:0'):
      image = image_ops.phantom(shape=[64, 64, 64], num_coils=4)
      # Currently only testing if it runs.
      maps = coil_ops.estimate_coil_sensitivities(image, # pylint: disable=unused-variable
                                                  coil_axis=0,
                                                  method='walsh')


class CoilCombineTest(test_util.TestCase):
  """Tests for coil combination op."""

  @parameterized.product(coil_axis=[0, -1],
                         keepdims=[True, False])
  @test_util.run_in_graph_and_eager_modes
  def test_sos(self, coil_axis, keepdims): # pylint: disable=missing-param-doc
    """Test sum of squares combination."""

    images = self._random_complex((20, 20, 8))

    combined = coil_ops.combine_coils(
      images, coil_axis=coil_axis, keepdims=keepdims)

    ref = tf.math.sqrt(
      tf.math.reduce_sum(images * tf.math.conj(images),
                         axis=coil_axis, keepdims=keepdims))

    self.assertAllEqual(combined.shape, ref.shape)
    self.assertAllClose(combined, ref)


  @parameterized.product(coil_axis=[0, -1],
                         keepdims=[True, False])
  @test_util.run_in_graph_and_eager_modes
  def test_adaptive(self, coil_axis, keepdims): # pylint: disable=missing-param-doc
    """Test adaptive combination."""

    images = self._random_complex((20, 20, 8))
    maps = self._random_complex((20, 20, 8))

    combined = coil_ops.combine_coils(
      images, maps=maps, coil_axis=coil_axis, keepdims=keepdims)

    ref = tf.math.reduce_sum(images * tf.math.conj(maps),
                             axis=coil_axis, keepdims=keepdims)

    ref /= tf.math.reduce_sum(maps * tf.math.conj(maps),
                              axis=coil_axis, keepdims=keepdims)

    self.assertAllEqual(combined.shape, ref.shape)
    self.assertAllClose(combined, ref)

  def setUp(self):
    super().setUp()
    tf.random.set_seed(0)

  def _random_complex(self, shape):
    return tf.dtypes.complex(
      tf.random.normal(shape),
      tf.random.normal(shape))


class CoilCompressionTest(test_util.TestCase):
  """Tests for coil compression op."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data = io_util.read_hdf5('tests/data/coil_ops_data.h5')

  # TODO(jmontalt): Look into SVD accuracy issues and re-enable these tests.
  # @test_util.run_in_graph_and_eager_modes
  # def test_coil_compression_svd(self):
  #   """Test SVD coil compression."""
  #   kspace = self.data['cc/kspace']
  #   result = self.data['cc/result/svd']

  #   cc_kspace = coil_ops.compress_coils(kspace)

  #   self.assertAllClose(cc_kspace, result, rtol=1e-2, atol=1e-2)

  # @test_util.run_in_graph_and_eager_modes
  # def test_coil_compression_svd_two_step(self):
  #   """Test SVD coil compression using two-step API."""
  #   kspace = self.data['cc/kspace']
  #   result = self.data['cc/result/svd']

  #   compressor = coil_ops.CoilCompressorSVD(out_coils=16)
  #   compressor = compressor.fit(kspace)
  #   cc_kspace = compressor.transform(kspace)
  #   self.assertAllClose(cc_kspace, result[..., :16], rtol=1e-2, atol=1e-2)

  # @test_util.run_in_graph_and_eager_modes
  # def test_coil_compression_svd_transposed(self):
  #   """Test SVD coil compression using two-step API."""
  #   kspace = self.data['cc/kspace']
  #   result = self.data['cc/result/svd']

  #   kspace = tf.transpose(kspace, [2, 0, 1])
  #   cc_kspace = coil_ops.compress_coils(kspace, coil_axis=0)
  #   cc_kspace = tf.transpose(cc_kspace, [1, 2, 0])

  #   self.assertAllClose(cc_kspace, result, rtol=1e-2, atol=1e-2)

  @test_util.run_in_graph_and_eager_modes
  def test_coil_compression_svd_basic(self):
    """Test coil compression using SVD method with basic arrays."""
    shape = (20, 20, 8)
    data = tf.dtypes.complex(
        tf.random.stateless_normal(shape, [32, 43]),
        tf.random.stateless_normal(shape, [321, 321]))

    params = {
      'out_coils': [None, 4],
      'variance_ratio': [None, 0.75]}

    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:
      with self.subTest(**p):
        if p['out_coils'] is not None and p['variance_ratio'] is not None:
          with self.assertRaisesRegex(
              ValueError,
              "Cannot specify both `out_coils` and `variance_ratio`"):
            coil_ops.compress_coils(data, **p)
          continue

        # Test op.
        compressed_data = coil_ops.compress_coils(data, **p)

        # Flatten input data.
        encoding_dims = tf.shape(data)[:-1]
        input_coils = tf.shape(data)[-1]
        data = tf.reshape(data, (-1, tf.shape(data)[-1]))
        samples = tf.shape(data)[0]

        # Calculate compression matrix.
        # This should be equivalent to TF line below. Not sure why
        # not. Giving up.
        # u, s, vh = np.linalg.svd(data, full_matrices=False)
        # v = vh.T.conj()
        s, u, v = tf.linalg.svd(data, full_matrices=False)
        matrix = tf.cond(samples > input_coils, lambda v=v: v, lambda u=u: u)

        out_coils = input_coils
        if p['variance_ratio'] and not p['out_coils']:
          variance = s ** 2 / 399.0
          out_coils = tf.math.count_nonzero(
              tf.math.cumsum(variance / tf.math.reduce_sum(variance), axis=0) <=
              p['variance_ratio'])
        if p['out_coils']:
          out_coils = p['out_coils']
        matrix = matrix[:, :out_coils]

        ref_data = tf.matmul(data, matrix)
        ref_data = tf.reshape(
            ref_data, tf.concat([encoding_dims, [out_coils]], 0))

        self.assertAllClose(compressed_data, ref_data)


if __name__ == '__main__':
  tf.test.main()
