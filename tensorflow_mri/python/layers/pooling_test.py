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

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for max pooling layers."""
# pylint: disable=missing-class-docstring,missing-function-docstring

from absl.testing import parameterized
from keras.testing_infra import test_combinations
import numpy as np
import tensorflow as tf

import tensorflow_mri as tfmri
from tensorflow_mri.python.util import test_util


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class AveragePoolingTest(test_util.TestCase):
  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_average_pooling_1d(self, dtype):
    # We need to disable training validation when type is complex and we are
    # executing in graph mode. This is because the MSE loss used in
    # `test_util.layer_test` returns a complex number which is not possible.
    validate_training = dtype == "float32" or tf.executing_eagerly()
    for padding in ["valid", "same"]:
      for stride in [1, 2]:
        test_util.layer_test(
            tfmri.layers.AveragePooling1D,
            kwargs={"strides": stride, "padding": padding, "dtype": dtype},
            input_shape=(3, 5, 4),
            input_dtype=dtype,
            expected_output_dtype=dtype,
            validate_training=validate_training
        )

    test_util.layer_test(
        tfmri.layers.AveragePooling1D,
        kwargs={"data_format": "channels_first", "dtype": dtype},
        input_shape=(3, 2, 6),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_average_pooling_2d(self, dtype):
    # We need to disable training validation when type is complex and we are
    # executing in graph mode. This is because the MSE loss used in
    # `test_util.layer_test` returns a complex number which is not possible.
    validate_training = dtype == "float32" or tf.executing_eagerly()
    test_util.layer_test(
        tfmri.layers.AveragePooling2D,
        kwargs={"strides": (2, 2),
                "padding": "same",
                "pool_size": (2, 2),
                "dtype": dtype},
        input_shape=(3, 5, 6, 4),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )
    test_util.layer_test(
        tfmri.layers.AveragePooling2D,
        kwargs={"strides": (2, 2),
                "padding": "valid",
                "pool_size": (3, 3),
                "dtype": dtype},
        input_shape=(3, 5, 6, 4),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )

    # This part of the test can only run on GPU but doesn't appear
    # to be properly assigned to a GPU when running in eager mode.
    if not tf.executing_eagerly():
      # Only runs on GPU with CUDA, channels_first is not supported on
      # CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      if tf.test.is_gpu_available(cuda_only=True):
        test_util.layer_test(
            tfmri.layers.AveragePooling2D,
            kwargs={
                "strides": (1, 1),
                "padding": "valid",
                "pool_size": (2, 2),
                "data_format": "channels_first",
                "dtype": dtype
            },
            input_shape=(3, 4, 5, 6),
            input_dtype=dtype,
            expected_output_dtype=dtype,
            validate_training=validate_training
        )

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_average_pooling_3d(self, dtype):
    # We need to disable training validation when type is complex and we are
    # executing in graph mode. This is because the MSE loss used in
    # `test_util.layer_test` returns a complex number which is not possible.
    validate_training = dtype == "float32" or tf.executing_eagerly()
    pool_size = (3, 3, 3)
    test_util.layer_test(
        tfmri.layers.AveragePooling3D,
        kwargs={"strides": 2,
                "padding": "valid",
                "pool_size": pool_size,
                "dtype": dtype},
        input_shape=(3, 11, 12, 10, 4),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )
    test_util.layer_test(
        tfmri.layers.AveragePooling3D,
        kwargs={
            "strides": 3,
            "padding": "valid",
            "data_format": "channels_first",
            "pool_size": pool_size,
            "dtype": dtype
        },
        input_shape=(3, 4, 11, 12, 10),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )


@test_combinations.generate(test_combinations.combine(mode=["graph", "eager"]))
class MaxPoolingTest(test_util.TestCase):
  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_max_pooling_1d(self, dtype):
    # We need to disable training validation when type is complex and we are
    # executing in graph mode. This is because the MSE loss used in
    # `test_util.layer_test` returns a complex number which is not possible.
    validate_training = dtype == "float32" or tf.executing_eagerly()
    for padding in ["valid", "same"]:
      for stride in [1, 2]:
        test_util.layer_test(
            tfmri.layers.MaxPooling1D,
            kwargs={"strides": stride, "padding": padding, "dtype": dtype},
            input_shape=(3, 5, 4),
            input_dtype=dtype,
            expected_output_dtype=dtype,
            validate_training=validate_training
        )
    test_util.layer_test(
        tfmri.layers.MaxPooling1D,
        kwargs={"data_format": "channels_first", "dtype": dtype},
        input_shape=(3, 2, 6),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_max_pooling_1d_result(self, dtype):
    if dtype == 'float32':
      inputs = np.array([[2., 3., 4., 1.],
                         [1., 8., 0., 2.],
                         [4., 1., 6., 3.],
                         [3., 7., 5., 4.]])[..., None]
      expected = np.array([[3., 4.],
                           [8., 2.],
                           [4., 6.],
                           [7., 5.]])[..., None]
    elif dtype == 'complex64':
      inputs = np.array(
          [[2. + 1.j, 3. + 2.j, 4. + 4.j, 1. + 5.j],
           [5. + 7.j, 8. + 0.j, 3. + 9.j, 2. + 2.j],
           [4. + 2.j, 1. + 1.j, 6. + 3.j, 3. + 4.j],
           [3. + 3.j, 7. + 3.j, 5. + 0.j, 4. + 8.j]])[..., None]
      expected = np.array([[3. + 2.j, 4. + 4.j],
                           [5. + 7.j, 3. + 9.j],
                           [4. + 2.j, 6. + 3.j],
                           [7. + 3.j, 4. + 8.j]])[..., None]
    layer = tfmri.layers.MaxPooling1D(dtype=dtype)
    output = layer(inputs)
    self.assertAllClose(expected, output)

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_max_pooling_2d(self, dtype):
    # We need to disable training validation when type is complex and we are
    # executing in graph mode. This is because the MSE loss used in
    # `test_util.layer_test` returns a complex number which is not possible.
    validate_training = dtype == "float32" or tf.executing_eagerly()
    pool_size = (3, 3)
    for strides in [(1, 1), (2, 2)]:
      test_util.layer_test(
          tfmri.layers.MaxPooling2D,
          kwargs={
              "strides": strides,
              "padding": "valid",
              "pool_size": pool_size,
              "dtype": dtype
          },
          input_shape=(3, 5, 6, 4),
          input_dtype=dtype,
          expected_output_dtype=dtype,
          validate_training=validate_training
      )

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_max_pooling_2d_result(self, dtype):
    if dtype == 'float32':
      inputs = np.array([[2., 3., 4., 1.],
                         [1., 8., 0., 2.],
                         [4., 1., 6., 3.],
                         [3., 7., 5., 4.]])[None, ..., None]
      expected = np.array([[8., 4.],
                           [7., 6.]])[None, ..., None]
    elif dtype == 'complex64':
      inputs = np.array(
          [[2. + 1.j, 3. + 2.j, 4. + 3.j, 1. + 7.j],
           [5. + 7.j, 8. + 0.j, 3. + 9.j, 2. + 2.j],
           [4. + 2.j, 1. + 1.j, 6. + 3.j, 3. + 4.j],
           [3. + 3.j, 7. + 3.j, 5. + 0.j, 4. + 8.j]])[None, ..., None]
      expected = np.array([[5. + 7.j, 3. + 9.j],
                           [7. + 3.j, 4. + 8.j]])[None, ..., None]
    layer = tfmri.layers.MaxPooling2D(dtype=dtype)
    output = layer(inputs)
    self.assertAllClose(expected, output)

  @parameterized.named_parameters(
      ("float32", "float32"),
      ("complex64", "complex64")
  )
  def test_max_pooling_3d(self, dtype):
    # We need to disable training validation when type is complex and we are
    # executing in graph mode. This is because the MSE loss used in
    # `test_util.layer_test` returns a complex number which is not possible.
    validate_training = dtype == "float32" or tf.executing_eagerly()
    pool_size = (3, 3, 3)
    test_util.layer_test(
        tfmri.layers.MaxPooling3D,
        kwargs={"strides": 2,
                "padding": "valid",
                "pool_size": pool_size,
                "dtype": dtype},
        input_shape=(3, 11, 12, 10, 4),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )
    test_util.layer_test(
        tfmri.layers.MaxPooling3D,
        kwargs={
            "strides": 3,
            "padding": "valid",
            "data_format": "channels_first",
            "pool_size": pool_size,
            "dtype": dtype
        },
        input_shape=(3, 4, 11, 12, 10),
        input_dtype=dtype,
        expected_output_dtype=dtype,
        validate_training=validate_training
    )

  @parameterized.named_parameters(
      ("valid", "valid"),
      ("same", "same")
  )
  def test_max_pooling_3d_result(self, padding):
    x = tf.random.stateless_uniform((3, 11, 12, 10, 4), [32, 12])
    real_layer = tfmri.layers.MaxPooling3D(
        padding=padding, dtype="float32")
    complex_layer = tfmri.layers.MaxPooling3D(
        padding=padding, dtype="complex64")
    real_output = real_layer(x)
    complex_output = complex_layer(tf.cast(x, tf.complex64))
    self.assertAllClose(real_output, tf.math.real(complex_output))


if __name__ == "__main__":
  tf.test.main()
