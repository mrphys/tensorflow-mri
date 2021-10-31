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
"""Tests for module `conv_blocks`."""

from absl.testing import parameterized
import tensorflow as tf

from tensorflow_mri.python.layers import preproc_layers
from tensorflow_mri.python.ops import image_ops
from tensorflow_mri.python.util import test_util


class KSpaceResamplingTest(test_util.TestCase):

  @parameterized.product(dens_algo=['geometric', 'radial', 'jackson', 'pipe'])
  @test_util.run_in_graph_and_eager_modes
  def test_radial_2d(self, dens_algo):
    """Test radial 2D configuration."""
    image_shape = [256, 256]
    image = image_ops.phantom(shape=image_shape)
    image = tf.expand_dims(image, -1)

    layer = preproc_layers.KSpaceResampling(image_shape=image_shape,
                                            trajectory='radial',
                                            views=403,
                                            angle_range='half',
                                            dens_algo=dens_algo)

    result_max = {
        'geometric': 1.4835564,
        'radial': 1.4835564,
        'jackson': 1.5735059,
        'pipe': 1.4818194
    }

    output = layer(image)
    self.assertAllEqual(output.shape, [256, 256, 1])
    self.assertDTypeEqual(output, 'float32')
    self.assertAllClose(tf.math.reduce_max(output), result_max[dens_algo],
                        rtol=1e-4, atol=1e-4)

  @parameterized.product(dens_algo=['geometric', 'radial', 'jackson', 'pipe'])
  def test_radial_2d_impulse(self, dens_algo):
    """Test radial 2D with impulse function."""
    image = tf.scatter_nd([[96, 96]], [1.0], [192, 192])
    image = tf.expand_dims(image, -1)

    layer = preproc_layers.KSpaceResampling(image_shape=[192, 192],
                                            trajectory='radial',
                                            views=89,
                                            angle_range='half',
                                            dens_algo=dens_algo)

    result_max = {
        'geometric': 1.0000054,
        'radial': 1.0000054,
        'jackson': 0.6805566,
        'pipe': 0.6819394
    }

    output = layer(image)
    self.assertAllEqual(output.shape, [192, 192, 1])
    self.assertDTypeEqual(output, 'float32')
    self.assertAllClose(output[96, 96, 0], result_max[dens_algo],
                        rtol=1e-4, atol=1e-4)


class ResizeWithCropOrPadTest(test_util.TestCase):

  @test_util.run_in_graph_and_eager_modes
  def test_output_shapes(self):
    """Test output shapes."""
    input1 = tf.keras.Input(shape=(64, 16, 4), batch_size=1)
    input2 = tf.keras.Input(shape=(2, 64, 16, 4), batch_size=1)
    input3 = tf.keras.Input(shape=(None, None, None, 1), batch_size=1)

    layer1 = preproc_layers.ResizeWithCropOrPad(shape=[32, 32])
    layer2 = preproc_layers.ResizeWithCropOrPad(shape=[64, 64, 64])

    output1 = layer1(input1)
    self.assertAllEqual(output1.shape, [1, 32, 32, 4])

    output2 = layer1(input2)
    self.assertAllEqual(output2.shape, [1, 2, 32, 32, 4])

    output3 = layer2(input3)
    self.assertAllEqual(output3.shape, [1, 64, 64, 64, 1])
  
  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        shape=[32, 32],
        name='resize_with_crop_or_pad',
        dtype='float32',
        trainable=True)

    layer1 = preproc_layers.ResizeWithCropOrPad(**config)
    self.assertEqual(layer1.get_config(), config)

    layer2 = preproc_layers.ResizeWithCropOrPad.from_config(layer1.get_config())
    self.assertAllEqual(layer1.get_config(), layer2.get_config())

  def _assert_static_shape(self, fn, input_shape, expected_output_shape):
    """Asserts that function returns the expected static shapes."""
    @tf.function
    def graph_fn(x):
      return fn(x)

    input_spec = tf.TensorSpec(shape=input_shape)
    concrete_fn = graph_fn.get_concrete_function(input_spec)

    self.assertAllEqual(concrete_fn.structured_outputs.shape,
                        expected_output_shape)


class TransposeTest(test_util.TestCase):

  @parameterized.product(perm=[[0, 2, 1], [1, 0, 2]],
                         conjugate=[True, False])
  @test_util.run_in_graph_and_eager_modes
  def test_result(self, perm, conjugate):
    """Test result shapes."""
    input1 = tf.random.stateless_normal([4, 4, 4], [234, 231])

    layer1 = preproc_layers.Transpose(perm=perm, conjugate=conjugate)

    output1 = layer1(input1)
    self.assertAllEqual(output1, tf.transpose(input1, perm=perm,
                                              conjugate=conjugate))

  def test_serialize_deserialize(self):
    """Test de/serialization."""
    config = dict(
        perm=[1, 0],
        conjugate=True,
        name='transpose',
        dtype='float32',
        trainable=True)

    layer1 = preproc_layers.Transpose(**config)
    self.assertEqual(layer1.get_config(), config)

    layer2 = preproc_layers.Transpose.from_config(layer1.get_config())
    self.assertAllEqual(layer1.get_config(), layer2.get_config())


if __name__ == '__main__':
  tf.test.main()
