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
# pylint: disable=missing-class-docstring

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.util import test_util


class CartesianProductTest(test_util.TestCase):
  """Tests for the `cartesian_product` op."""

  @test_util.run_in_graph_and_eager_modes
  def test_cartesian_product(self):
    """Test `cartesian_product` op."""
    vec1 = [1, 2, 3]
    vec2 = [4, 5]

    ref = [[1, 4],
           [1, 5],
           [2, 4],
           [2, 5],
           [3, 4],
           [3, 5]]

    result = array_ops.cartesian_product(vec1, vec2)
    self.assertAllEqual(result, ref)


class MeshgridTest(test_util.TestCase):
  """Tests for the `meshgrid` op."""

  @test_util.run_in_graph_and_eager_modes
  def test_meshgrid(self):
    """Test `meshgrid` op."""
    vec1 = [1, 2, 3]
    vec2 = [4, 5]

    ref = [[[1, 4], [1, 5]],
           [[2, 4], [2, 5]],
           [[3, 4], [3, 5]]]

    result = array_ops.meshgrid(vec1, vec2)
    self.assertAllEqual(result, ref)


class RavelMultiIndexTest(test_util.TestCase):
  """Tests for the `ravel_multi_index` op."""

  @test_util.run_in_graph_and_eager_modes
  def test_ravel_multi_index_2d(self):
    """Test multi-index ravelling (2D, 1D indices array)."""
    indices = [[0, 0], [0, 1], [2, 2], [3, 1]]
    expected = [0, 1, 10, 13]

    result = array_ops.ravel_multi_index(indices, [4, 4])
    self.assertAllEqual(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_ravel_multi_index_2d_scalar(self):
    """Test multi-index ravelling (2D, scalar index)."""
    indices = [2, 2]
    expected = 12

    result = array_ops.ravel_multi_index(indices, [4, 5])
    self.assertAllEqual(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_ravel_multi_index_2d_batch(self):
    """Test multi-index ravelling (2D, 2D indices array)."""
    indices = [[[0, 0], [0, 1], [2, 2], [3, 1]],
               [[0, 2], [2, 0], [1, 3], [3, 3]]]
    expected = [[0, 1, 10, 13],
                [2, 8, 7, 15]]

    result = array_ops.ravel_multi_index(indices, [4, 4])
    self.assertAllEqual(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_ravel_multi_index_3d(self):
    """Test multi-index ravelling (3D, 1D indices array)."""
    indices = [[0, 0, 0], [0, 1, 1], [2, 2, 3], [3, 1, 2]]
    expected = [0, 5, 43, 54]

    result = array_ops.ravel_multi_index(indices, [4, 4, 4])
    self.assertAllEqual(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def test_ravel_multi_index_3d_scalar(self):
    """Test multi-index ravelling (3D, scalar index)."""
    indices = [2, 2, 1]
    expected = 41

    result = array_ops.ravel_multi_index(indices, [4, 4, 4])
    self.assertAllEqual(result, expected)



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

    y_tf = array_ops.central_crop(x_np, shape)

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

    y_tf = array_ops.central_crop(x_np, shape)

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

    y_tf = array_ops.resize_with_crop_or_pad(x_np, shape)

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

    y_tf = array_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_padding_non_default_mode(self):
    """Test padding."""
    shape = [7]
    x_np = np.array([1, 2, 3])
    y_np = np.array([3, 2, 1, 2, 3, 2, 1])

    y_tf = array_ops.resize_with_crop_or_pad(x_np, shape,
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

    y_tf = array_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  @test_util.run_in_graph_and_eager_modes
  def test_padding_cropping_unknown_dimension(self):
    """Test combined cropping and padding with an unknown dimension."""
    shape = [1, -1]
    x_np = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    y_np = np.array([[4, 5, 6]])

    y_tf = array_ops.resize_with_crop_or_pad(x_np, shape)

    self.assertAllEqual(y_tf, y_np)

  def test_static_shape(self):
    """Test static shapes."""
    def get_fn(target_shape):
      return lambda x: array_ops.resize_with_crop_or_pad(x, target_shape)

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


class UpdateTensorTest(test_util.TestCase):

  @test_util.run_all_execution_modes
  @parameterized.named_parameters(
      # name, tensor, slices, value
      ("test0", [0, 0, 0], (slice(1, 2),), [2]),
      ("test1", [0, 0, 0], (slice(0, None, 2)), [2, 1]),
      ("test2", [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
       (slice(0, 2), slice(1, 3)), [[1, 2], [3, 4]])
  )
  def test_update_tensor(self, tensor, slices, value):
    with tf.device('/cpu:0'):
      expected = np.asarray(tensor)
      expected[slices] = value
      tensor = tf.constant(tensor)
      result = array_ops.update_tensor(tensor, slices, value)
      self.assertAllClose(expected, result)


if __name__ == '__main__':
  tf.test.main()
