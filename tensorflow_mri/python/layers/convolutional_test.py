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
"""Tests for convolutional layers."""
# pylint: disable=missing-class-docstring,missing-function-docstring

# Using some private functions. This works as of TF/Keras 2.8.0.
from absl.testing import parameterized
import keras
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import test_util
import tensorflow_mri as tfmri


@test_combinations.run_all_keras_modes
class Conv1DTest(test_combinations.TestCase):

  def _run_test(self, kwargs, expected_output_shape):
    num_samples = 2
    stack_size = 3
    length = 7

    with self.cached_session():
      test_utils.layer_test(
          tfmri.layers.Conv1D,
          kwargs=kwargs,
          input_shape=(num_samples, length, stack_size),
          expected_output_shape=expected_output_shape)

  def _run_test_extra_batch_dim(self, kwargs, expected_output_shape):
    batch_shape = (2, 11)
    stack_size = 3
    length = 7

    with self.cached_session():
      if expected_output_shape is not None:
        expected_output_shape = (None,) + expected_output_shape

      test_utils.layer_test(
          tfmri.layers.Conv1D,
          kwargs=kwargs,
          input_shape=batch_shape + (length, stack_size),
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 5, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 7, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (None, 7, 2)),
      ('padding_same_dilation_3', {
          'padding': 'same',
          'dilation_rate': 3
      }, (None, 7, 2)),
      ('padding_causal', {
          'padding': 'causal'
      }, (None, 7, 2)),
      ('strides', {
          'strides': 2
      }, (None, 3, 2)),
      ('dilation_rate', {
          'dilation_rate': 2
      }, (None, 3, 2)),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 5, 6), True),
  )
  def test_conv1d(self, kwargs, expected_output_shape, requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = 3
    if not requires_gpu or tf.test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape)
      self._run_test_extra_batch_dim(kwargs, expected_output_shape)

  def test_conv1d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session():
      layer = tfmri.layers.Conv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv1d_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session():
      layer = tfmri.layers.Conv1D(**kwargs)
      layer.build((None, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv1d_recreate_conv(self):
    with self.cached_session():
      layer = tfmri.layers.Conv1D(filters=1,
                                  kernel_size=3,
                                  strides=1,
                                  dilation_rate=2,
                                  padding='causal')
      inpt1 = np.random.normal(size=[1, 2, 1])
      inpt2 = np.random.normal(size=[1, 1, 1])
      outp1_shape = layer(inpt1).shape
      _ = layer(inpt2).shape
      self.assertEqual(outp1_shape, layer(inpt1).shape)

  def test_conv1d_recreate_conv_unknown_dims(self):
    with self.cached_session():
      layer = tfmri.layers.Conv1D(filters=1,
                                  kernel_size=3,
                                  strides=1,
                                  dilation_rate=2,
                                  padding='causal')

      inpt1 = np.random.normal(size=[1, 9, 1]).astype(np.float32)
      inpt2 = np.random.normal(size=[1, 2, 1]).astype(np.float32)
      outp1_shape = layer(inpt1).shape

      @tf.function(input_signature=[
          tf.TensorSpec([1, None, 1])])
      def fn(inpt):
        return layer(inpt)

      fn(inpt2)
      self.assertEqual(outp1_shape, layer(inpt1).shape)

  def test_conv1d_invalid_output_shapes(self):
    kwargs = {'filters': 2, 'kernel_size': 20}
    with self.assertRaisesRegex(
        ValueError, r"""One of the dimensions in the output is <= 0"""):
      layer = tfmri.layers.Conv1D(**kwargs)
      layer.build((None, 5, 2))

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 5, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 7, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (None, 7, 2)),
      ('padding_same_dilation_3', {
          'padding': 'same',
          'dilation_rate': 3
      }, (None, 7, 2)),
      ('padding_causal', {
          'padding': 'causal'
      }, (None, 7, 2)),
      ('strides', {
          'strides': 2
      }, (None, 3, 2)),
      ('dilation_rate', {
          'dilation_rate': 2
      }, (None, 3, 2)),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 5, 6), True),
      ('activation', {
          'activation': 'exponential'
      }, (None, 5, 2)),
      ('regularizer', {
          'kernel_regularizer': 'l1'
      }, (None, 5, 2))
  )
  def test_conv1d_complex(
      self, kwargs, expected_output_shape, requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = 3
    kwargs['dtype'] = 'complex64'
    if not requires_gpu or tf.test.is_gpu_available(cuda_only=True):
      num_samples = 2
      stack_size = 3
      length = 7
      input_shape = (num_samples, length, stack_size)

      with self.cached_session():
        test_utils.layer_test(
            tfmri.layers.Conv1D,
            kwargs=kwargs,
            input_shape=input_shape,
            input_dtype='complex64',
            expected_output_shape=expected_output_shape,
            expected_output_dtype='complex64',
            validate_training=False)

      # Training validation in `layer_test` is performed with RMSprop, with
      # fails due to missing ops in core TF (as of TF 2.8). However, the layer
      # can be trained with other optimizers such as SGD or Adam.
      self._validate_training(kwargs, input_shape, 'complex64')

  def _validate_training(self, layer_kwargs, input_shape, input_dtype):  # pylint: disable=missing-function-docstring
    # Instantiate layer.
    layer = tfmri.layers.Conv1D(**layer_kwargs)

    # Create model and generate input/output data.
    x = tf.keras.Input(shape=input_shape[1:], dtype=input_dtype)
    model = tf.keras.Model(x, layer(x))
    input_data = 10 * np.random.random(input_shape)
    actual_output = model.predict(input_data)

    # Now compile and train on a batch.
    model.compile('adam', mse)
    model.train_on_batch(input_data, actual_output)

  def test_conv1d_complex_values(self):
    """Test that result of complex-valued convolution is correct."""
    input_data = tf.constant([[[ 3.+2.j],
                               [-5.+0.j],
                               [ 4.-3.j],
                               [ 0.-4.j],
                               [-5.-4.j]]], dtype=tf.complex64)  # (1, 5, 1)
    expected_result = [[[4. - 1.j], [11. - 12.j], [0. - 5.j]]]

    def kernel_initializer(shape, dtype):  # pylint: disable=unused-argument
      return [[[-1. + 1.j]], [[0 + 0.j]], [[1. + 1.j]]]

    def bias_initializer(shape, dtype):  # pylint: disable=unused-argument
      return [2 - 3.j]

    layer = tfmri.layers.Conv1D(1, 3,
                                kernel_initializer=kernel_initializer,
                                bias_initializer=bias_initializer,
                                dtype='complex64')
    layer.build((1, 5, 1))
    result = layer(input_data)

    self.assertAllClose(expected_result, result)


@test_combinations.run_all_keras_modes
class Conv2DTest(test_combinations.TestCase):

  def _run_test(self, kwargs, expected_output_shape, spatial_shape=(7, 6)):
    num_samples = 2
    stack_size = 3
    num_row, num_col = spatial_shape
    input_data = None
    # Generate valid input data.
    if None in spatial_shape:
      input_data_shape = (num_samples, num_row or 7, num_col or 6, stack_size)
      input_data = 10 * np.random.random(input_data_shape).astype(np.float32)

    with self.cached_session():
      test_utils.layer_test(
          tfmri.layers.Conv2D,
          kwargs=kwargs,
          input_shape=(num_samples, num_row, num_col, stack_size),
          input_data=input_data,
          expected_output_shape=expected_output_shape)

  def _run_test_extra_batch_dim(self,
                                kwargs,
                                expected_output_shape,
                                spatial_shape=(7, 6)):
    batch_shape = (2, 11)
    stack_size = 3
    num_row, num_col = spatial_shape
    input_data = None
    # Generate valid input data.
    if None in spatial_shape:
      input_data_shape = batch_shape + (num_row or 7, num_col or 6, stack_size)
      input_data = 10 * np.random.random(input_data_shape).astype(np.float32)

    with self.cached_session():
      if expected_output_shape is not None:
        expected_output_shape = (None,) + expected_output_shape
      test_utils.layer_test(
          tfmri.layers.Conv2D,
          kwargs=kwargs,
          input_shape=batch_shape + (num_row, num_col, stack_size),
          input_data=input_data,
          expected_output_shape=expected_output_shape)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 7, 6, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (None, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2)
      }, (None, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2)
      }, (None, 3, 2, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {
          'data_format': 'channels_first'
      }, None, True),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 5, 4, 6), True),
      ('dilation_2_unknown_width', {
          'dilation_rate': (2, 2)
      }, (None, None, 2, 2), False, (None, 6)),
      ('dilation_2_unknown_height', {
          'dilation_rate': (2, 2)
      }, (None, 3, None, 2), False, (7, None)),
  )
  def test_conv2d(self,
                  kwargs,
                  expected_output_shape=None,
                  requires_gpu=False,
                  spatial_shape=(7, 6)):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = (3, 3)
    if not requires_gpu or tf.test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape, spatial_shape)
      self._run_test_extra_batch_dim(kwargs, expected_output_shape,
                                     spatial_shape)

  def test_conv2d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session():
      layer = tfmri.layers.Conv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv2d_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session():
      layer = tfmri.layers.Conv2D(**kwargs)
      layer.build((None, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv2d_zero_kernel_size(self):
    kwargs = {'filters': 2, 'kernel_size': 0}
    with self.assertRaises(ValueError):
      tfmri.layers.Conv2D(**kwargs)

  def test_conv2d_invalid_output_shapes(self):
    kwargs = {'filters': 2, 'kernel_size': 20}
    with self.assertRaisesRegex(
        ValueError, r"""One of the dimensions in the output is <= 0"""):
      layer = tfmri.layers.Conv2D(**kwargs)
      layer.build((None, 5, 5, 2))

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 7, 6, 2)),
      ('padding_same_dilation_2', {
          'padding': 'same',
          'dilation_rate': 2
      }, (None, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2)
      }, (None, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2)
      }, (None, 3, 2, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {
          'data_format': 'channels_first'
      }, None, True),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 5, 4, 6), True),
      ('dilation_2_unknown_width', {
          'dilation_rate': (2, 2)
      }, (None, None, 2, 2), False, (None, 6)),
      ('dilation_2_unknown_height', {
          'dilation_rate': (2, 2)
      }, (None, 3, None, 2), False, (7, None)),
      ('activation', {
          'activation': 'exponential'
      }, (None, 5, 4, 2)),
      ('regularizer', {
          'kernel_regularizer': 'l1'
      }, (None, 5, 4, 2))
  )
  def test_conv2d_complex(self,
                          kwargs,
                          expected_output_shape,
                          requires_gpu=False,
                          spatial_shape=(7, 6)):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = 3
    kwargs['dtype'] = 'complex64'
    if not requires_gpu or tf.test.is_gpu_available(cuda_only=True):
      num_samples = 2
      stack_size = 3
      rows, cols = spatial_shape
      input_shape = (num_samples, rows, cols, stack_size)

      # Generate valid input data.
      if None in spatial_shape:
        input_data_shape = (num_samples, rows or 7, cols or 6, stack_size)
        input_data = 10 * np.random.random(input_data_shape).astype(np.float32)
      else:
        input_data = None

      with self.cached_session():
        test_utils.layer_test(
            tfmri.layers.Conv2D,
            kwargs=kwargs,
            input_shape=input_shape,
            input_dtype='complex64',
            input_data=input_data,
            expected_output_shape=expected_output_shape,
            expected_output_dtype='complex64',
            validate_training=False)

      # Training validation in `layer_test` is performed with RMSprop, with
      # fails due to missing ops in core TF (as of TF 2.8). However, the layer
      # can be trained with other optimizers such as SGD or Adam.
      self._validate_training(kwargs, input_shape, 'complex64',
                              input_data=input_data)

  def _validate_training(self, layer_kwargs, input_shape, input_dtype,  # pylint: disable=missing-function-docstring
                         input_data=None):
    # Instantiate layer.
    layer = tfmri.layers.Conv2D(**layer_kwargs)

    # Create model and generate input/output data.
    x = tf.keras.Input(shape=input_shape[1:], dtype=input_dtype)
    model = tf.keras.Model(x, layer(x))
    if input_data is None:
      input_data = 10 * np.random.random(input_shape)
    actual_output = model.predict(input_data)

    # Now compile and train on a batch.
    model.compile('adam', mse)
    model.train_on_batch(input_data, actual_output)


@test_combinations.run_all_keras_modes
class Conv3DTest(test_combinations.TestCase):

  def _run_test(self, kwargs, expected_output_shape, validate_training=True):
    num_samples = 2
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    with self.cached_session():
      test_utils.layer_test(
          tfmri.layers.Conv3D,
          kwargs=kwargs,
          input_shape=(num_samples, depth, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape,
          validate_training=validate_training)

  def _run_test_extra_batch_dim(self,
                                kwargs,
                                expected_output_shape,
                                validate_training=True):
    batch_shape = (2, 11)
    stack_size = 3
    num_row = 7
    num_col = 6
    depth = 5

    with self.cached_session():
      if expected_output_shape is not None:
        expected_output_shape = (None,) + expected_output_shape

      test_utils.layer_test(
          tfmri.layers.Conv3D,
          kwargs=kwargs,
          input_shape=batch_shape + (depth, num_row, num_col, stack_size),
          expected_output_shape=expected_output_shape,
          validate_training=validate_training)

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 3, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 5, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2, 2)
      }, (None, 2, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2, 2)
      }, (None, 1, 3, 2, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {
          'data_format': 'channels_first'
      }, None, True),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 3, 5, 4, 6), True),
  )
  def test_conv3d(self, kwargs, expected_output_shape=None, requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = (3, 3, 3)
    # train_on_batch currently fails with XLA enabled on GPUs
    test_training = 'groups' not in kwargs or not test_util.is_xla_enabled()
    if not requires_gpu or tf.test.is_gpu_available(cuda_only=True):
      self._run_test(kwargs, expected_output_shape, test_training)
      self._run_test_extra_batch_dim(kwargs, expected_output_shape,
                                     test_training)

  def test_conv3d_regularizers(self):
    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_regularizer': 'l2',
        'bias_regularizer': 'l2',
        'activity_regularizer': 'l2',
        'strides': 1
    }
    with self.cached_session():
      layer = tfmri.layers.Conv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(len(layer.losses), 2)
      self.assertEqual(len(layer.losses), 2)
      layer(keras.backend.variable(np.ones((1, 5, 5, 5, 2))))
      self.assertEqual(len(layer.losses), 3)

  def test_conv3d_constraints(self):
    k_constraint = lambda x: x
    b_constraint = lambda x: x

    kwargs = {
        'filters': 3,
        'kernel_size': 3,
        'padding': 'valid',
        'kernel_constraint': k_constraint,
        'bias_constraint': b_constraint,
        'strides': 1
    }
    with self.cached_session():
      layer = tfmri.layers.Conv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))
      self.assertEqual(layer.kernel.constraint, k_constraint)
      self.assertEqual(layer.bias.constraint, b_constraint)

  def test_conv3d_dynamic_shape(self):
    input_data = np.random.random((1, 3, 3, 3, 3)).astype(np.float32)
    with self.cached_session():
      # Won't raise error here.
      test_utils.layer_test(
          tfmri.layers.Conv3D,
          kwargs={
              'data_format': 'channels_last',
              'filters': 3,
              'kernel_size': 3
          },
          input_shape=(None, None, None, None, 3),
          input_data=input_data)
      if tf.test.is_gpu_available(cuda_only=True):
        test_utils.layer_test(
            tfmri.layers.Conv3D,
            kwargs={
                'data_format': 'channels_first',
                'filters': 3,
                'kernel_size': 3
            },
            input_shape=(None, 3, None, None, None),
            input_data=input_data)

  def test_conv3d_invalid_output_shapes(self):
    kwargs = {'filters': 2, 'kernel_size': 20}
    with self.assertRaisesRegex(
        ValueError, r"""One of the dimensions in the output is <= 0"""):
      layer = tfmri.layers.Conv3D(**kwargs)
      layer.build((None, 5, 5, 5, 2))

  @parameterized.named_parameters(
      ('padding_valid', {
          'padding': 'valid'
      }, (None, 3, 5, 4, 2)),
      ('padding_same', {
          'padding': 'same'
      }, (None, 5, 7, 6, 2)),
      ('strides', {
          'strides': (2, 2, 2)
      }, (None, 2, 3, 2, 2)),
      ('dilation_rate', {
          'dilation_rate': (2, 2, 2)
      }, (None, 1, 3, 2, 2)),
      # Only runs on GPU with CUDA, channels_first is not supported on CPU.
      # TODO(b/62340061): Support channels_first on CPU.
      ('data_format', {
          'data_format': 'channels_first'
      }, None, True),
      # Only runs on GPU with CUDA, groups are not supported on CPU.
      # https://github.com/tensorflow/tensorflow/issues/29005
      ('group', {
          'groups': 3,
          'filters': 6
      }, (None, 3, 5, 4, 6), True),
      ('activation', {
          'activation': 'exponential'
      }, (None, 3, 5, 4, 2)),
      ('regularizer', {
          'kernel_regularizer': 'l1'
      }, (None, 3, 5, 4, 2))
  )
  def test_conv3d_complex(self,
                          kwargs,
                          expected_output_shape,
                          requires_gpu=False):
    kwargs['filters'] = kwargs.get('filters', 2)
    kwargs['kernel_size'] = 3
    kwargs['dtype'] = 'complex64'
    if not requires_gpu or tf.test.is_gpu_available(cuda_only=True):
      num_samples = 2
      stack_size = 3
      rows = 7
      cols = 6
      depth = 5
      input_shape = (num_samples, depth, rows, cols, stack_size)

      with self.cached_session():
        test_utils.layer_test(
            tfmri.layers.Conv3D,
            kwargs=kwargs,
            input_shape=input_shape,
            input_dtype='complex64',
            expected_output_shape=expected_output_shape,
            expected_output_dtype='complex64',
            validate_training=False)

      # Training validation in `layer_test` is performed with RMSprop, with
      # fails due to missing ops in core TF (as of TF 2.8). However, the layer
      # can be trained with other optimizers such as SGD or Adam.
      self._validate_training(kwargs, input_shape, 'complex64')

  def _validate_training(self, layer_kwargs, input_shape, input_dtype):  # pylint: disable=missing-function-docstring
    # Instantiate layer.
    layer = tfmri.layers.Conv3D(**layer_kwargs)

    # Create model and generate input/output data.
    x = tf.keras.Input(shape=input_shape[1:], dtype=input_dtype)
    model = tf.keras.Model(x, layer(x))
    input_data = 10 * np.random.random(input_shape)
    actual_output = model.predict(input_data)

    # Now compile and train on a batch.
    model.compile('adam', mse)
    model.train_on_batch(input_data, actual_output)


@test_combinations.run_all_keras_modes(always_skip_v1=True)
class GroupedConvTest(test_combinations.TestCase):

  @parameterized.named_parameters(
      ('Conv1D', tfmri.layers.Conv1D),
      ('Conv2D', tfmri.layers.Conv2D),
      ('Conv3D', tfmri.layers.Conv3D),
  )
  def test_group_conv_incorrect_use(self, layer):
    with self.assertRaisesRegex(ValueError, 'The number of filters'):
      layer(16, 3, groups=3)
    with self.assertRaisesRegex(ValueError, 'The number of input channels'):
      layer(16, 3, groups=4).build((32, 12, 12, 3))

  @parameterized.named_parameters(
      ('Conv1D', tfmri.layers.Conv1D, (32, 12, 32)),
      ('Conv2D', tfmri.layers.Conv2D, (32, 12, 12, 32)),
      ('Conv3D', tfmri.layers.Conv3D, (32, 12, 12, 12, 32)),
  )
  def test_group_conv(self, layer_cls, input_shape):
    # pylint: disable=no-value-for-parameter,redundant-keyword-arg,unexpected-keyword-arg
    if tf.test.is_gpu_available(cuda_only=True):
      with test_utils.use_gpu():
        inputs = tf.random.uniform(shape=input_shape)

        layer = layer_cls(16, 3, groups=4, use_bias=False)
        layer.build(input_shape)

        input_slices = tf.split(inputs, 4, axis=-1)
        weight_slices = tf.split(layer.kernel, 4, axis=-1)
        expected_outputs = tf.concat([
            tf.nn.convolution(inputs, weights)
            for inputs, weights in zip(input_slices, weight_slices)
        ],
                                            axis=-1)
        self.assertAllClose(
            layer(inputs), expected_outputs, rtol=3e-5, atol=3e-5)

  def test_group_conv_depthwise(self):
    # pylint: disable=no-value-for-parameter,redundant-keyword-arg
    if tf.test.is_gpu_available(cuda_only=True):
      with test_utils.use_gpu():
        inputs = tf.random.uniform(shape=(3, 27, 27, 32))

        layer = tfmri.layers.Conv2D(32, 3, groups=32, use_bias=False)
        layer.build((3, 27, 27, 32))

        weights_dw = tf.reshape(layer.kernel, [3, 3, 32, 1])
        expected_outputs = tf.compat.v1.nn.depthwise_conv2d(
            inputs, weights_dw, strides=[1, 1, 1, 1], padding='VALID')

        self.assertAllClose(layer(inputs), expected_outputs, rtol=1e-5)



@test_combinations.run_all_keras_modes
class ConvSequentialTest(test_combinations.TestCase):

  def _run_test(self, conv_layer_cls, kwargs, input_shape1, input_shape2,
                expected_output_shape1, expected_output_shape2):
    kwargs['filters'] = 1
    kwargs['kernel_size'] = 3
    kwargs['dilation_rate'] = 2
    with self.cached_session():
      layer = conv_layer_cls(**kwargs)
      output1 = layer(np.zeros(input_shape1))
      self.assertEqual(output1.shape, expected_output_shape1)
      output2 = layer(np.zeros(input_shape2))
      self.assertEqual(output2.shape, expected_output_shape2)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'},
       (1, 8, 2), (1, 5, 2), (1, 4, 1), (1, 1, 1)),
      ('padding_same', {'padding': 'same'},
       (1, 8, 2), (1, 5, 2), (1, 8, 1), (1, 5, 1)),
      ('padding_causal', {'padding': 'causal'},
       (1, 8, 2), (1, 5, 2), (1, 8, 1), (1, 5, 1)),
  )
  def test_conv1d(self, kwargs, input_shape1, input_shape2,
                  expected_output_shape1, expected_output_shape2):
    self._run_test(tfmri.layers.Conv1D, kwargs, input_shape1, input_shape2,
                   expected_output_shape1, expected_output_shape2)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'},
       (1, 7, 6, 2), (1, 6, 5, 2), (1, 3, 2, 1), (1, 2, 1, 1)),
      ('padding_same', {'padding': 'same'},
       (1, 7, 6, 2), (1, 6, 5, 2), (1, 7, 6, 1), (1, 6, 5, 1)),
  )
  def test_conv2d(self, kwargs, input_shape1, input_shape2,
                  expected_output_shape1, expected_output_shape2):
    self._run_test(tfmri.layers.Conv2D, kwargs, input_shape1, input_shape2,
                   expected_output_shape1, expected_output_shape2)

  @parameterized.named_parameters(
      ('padding_valid', {'padding': 'valid'},
       (1, 5, 7, 6, 2), (1, 8, 6, 5, 2), (1, 1, 3, 2, 1), (1, 4, 2, 1, 1)),
      ('padding_same', {'padding': 'same'},
       (1, 5, 7, 6, 2), (1, 8, 6, 5, 2), (1, 5, 7, 6, 1), (1, 8, 6, 5, 1)),
  )
  def test_conv3d(self, kwargs, input_shape1, input_shape2,
                  expected_output_shape1, expected_output_shape2):
    self._run_test(tfmri.layers.Conv3D, kwargs, input_shape1, input_shape2,
                   expected_output_shape1, expected_output_shape2)

  def test_dynamic_shape(self):
    with self.cached_session():
      layer = tfmri.layers.Conv3D(2, 3)
      input_shape = (5, None, None, 2)
      inputs = keras.Input(shape=input_shape)
      x = layer(inputs)
      # Won't raise error here with None values in input shape (b/144282043).
      layer(x)


# Default MSE loss fails because it returns a complex value.
def mse(y_true, y_pred):
  value = tf.keras.losses.mean_squared_error(y_true, y_pred)
  if value.dtype.is_complex:
    value = tf.math.real(value)
  return value


if __name__ == '__main__':
  tf.test.main()
