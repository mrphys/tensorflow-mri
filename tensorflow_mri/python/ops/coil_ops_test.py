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
from tensorflow_mri.python.utils import io_utils
from tensorflow_mri.python.utils import test_utils


class SensMapsTest(test_utils.TestCase):
  """Tests for ops related to estimation of coil sensitivity maps."""

  @classmethod
  def setUpClass(cls):

    super().setUpClass()
    cls.data = io_utils.read_hdf5('tests/data/coil_ops_data.h5')

  @test_utils.run_in_graph_and_eager_modes
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

    self.assertAllClose(maps, self.data['maps/walsh'])

  @test_utils.run_in_graph_and_eager_modes
  def test_walsh_transposed(self):
    """Test Walsh's method with a transposed array."""
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        tf.transpose(self.data['images'], [2, 0, 1]),
        coil_axis=0, method='walsh')

    self.assertAllClose(maps, tf.transpose(self.data['maps/walsh'], [2, 0, 1]))

  @test_utils.run_in_graph_and_eager_modes
  def test_inati(self):
    """Test Inati's method."""
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        self.data['images'], method='inati')

    self.assertAllClose(maps, self.data['maps/inati'], rtol=1e-4, atol=1e-4)

  @test_utils.run_in_graph_and_eager_modes
  def test_espirit(self):
    """Test ESPIRiT method."""
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        self.data['kspace'], method='espirit')

    self.assertAllClose(maps, self.data['maps/espirit'])

  @test_utils.run_in_graph_and_eager_modes
  def test_espirit_transposed(self):
    """Test ESPIRiT method with a transposed array."""
    with tf.device('/cpu:0'):
      maps = coil_ops.estimate_coil_sensitivities(
        tf.transpose(self.data['kspace'], [2, 0, 1]),
        coil_axis=0, method='espirit')

    self.assertAllClose(
      maps, tf.transpose(self.data['maps/espirit'], [2, 0, 1, 3]))


class CoilCombineTest(test_utils.TestCase):
  """Tests for coil combination op."""

  @parameterized.product(coil_axis=[0, -1],
                         keepdims=[True, False])
  @test_utils.run_in_graph_and_eager_modes
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
  @test_utils.run_in_graph_and_eager_modes
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


class CoilCompressionTest(test_utils.TestCase):
  """Tests for coil compression op."""

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.data = io_utils.read_hdf5('tests/data/coil_ops_data.h5')

  @test_utils.run_in_graph_and_eager_modes
  def test_coil_compression_svd(self):
    """Test SVD coil compression."""
    kspace = self.data['cc/kspace']
    result = self.data['cc/result/svd']

    cc_kspace = coil_ops.compress_coils(kspace)

    self.assertAllClose(cc_kspace, result)

  @test_utils.run_in_graph_and_eager_modes
  def test_coil_compression_svd_two_step(self):
    """Test SVD coil compression using two-step API."""
    kspace = self.data['cc/kspace']
    result = self.data['cc/result/svd']

    matrix = coil_ops.coil_compression_matrix(kspace, num_output_coils=16)
    self.assertAllEqual(tf.shape(matrix), [32, 16])

    cc_kspace = coil_ops.compress_coils(kspace, matrix=matrix)
    self.assertAllClose(cc_kspace, result[..., :16])

  @test_utils.run_in_graph_and_eager_modes
  def test_coil_compression_svd_transposed(self):
    """Test SVD coil compression using two-step API."""
    kspace = self.data['cc/kspace']
    result = self.data['cc/result/svd']

    kspace = tf.transpose(kspace, [2, 0, 1])
    cc_kspace = coil_ops.compress_coils(kspace, coil_axis=0)
    cc_kspace = tf.transpose(cc_kspace, [1, 2, 0])

    self.assertAllClose(cc_kspace, result)

  @test_utils.run_in_graph_and_eager_modes
  def test_coil_compression_svd_basic(self):
    """Test coil compression using SVD method with basic arrays."""
    shape = (20, 20, 8)
    data = tf.dtypes.complex(
      tf.random.normal(shape),
      tf.random.normal(shape))

    params = {
      'num_output_coils': [None, 4],
      'tol': [None, 0.05]}

    values = itertools.product(*params.values())
    params = [dict(zip(params.keys(), v)) for v in values]

    for p in params:
      with self.subTest(**p):

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

        num_output_coils = input_coils
        if p['tol'] and not p['num_output_coils']:
          num_output_coils = tf.math.count_nonzero(
              tf.abs(s) / tf.abs(s[0]) > p['tol'])
        if p['num_output_coils']:
          num_output_coils = p['num_output_coils']
        matrix = matrix[:, :num_output_coils]

        ref_data = tf.matmul(data, matrix)
        ref_data = tf.reshape(
            ref_data, tf.concat([encoding_dims, [num_output_coils]], 0))

        self.assertAllClose(compressed_data, ref_data)


if __name__ == '__main__':
  tf.test.main()
