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
"""Tests for module `signal_layers`."""

import keras
import numpy as np
import tensorflow as tf

import tensorflow_mri as tfmri
from tensorflow_mri.python.util import test_util


class DWTTest(test_util.TestCase):

  def test_dwt_1d(self):
    wavelet = 'haar'
    rng = np.random.default_rng()
    input_data = rng.normal(size=(1, 10, 1)).astype(np.float32)
    expected_output = tfmri.signal.dwt(input_data, wavelet, axes=(1,))

    self._test_layer(tfmri.layers.DWT1D,
                     {'wavelet': 'haar'},
                     input_data,
                     expected_output)

  def test_dwt_2d(self):
    wavelet = 'haar'
    rng = np.random.default_rng()
    input_data = rng.normal(size=(1, 10, 11, 1)).astype(np.float32)
    expected_output = tfmri.signal.dwt(input_data, wavelet, axes=(1, 2))

    self._test_layer(tfmri.layers.DWT2D,
                     {'wavelet': 'haar'},
                     input_data,
                     expected_output)

  def test_dwt_3d(self):
    wavelet = 'haar'
    rng = np.random.default_rng()
    input_data = rng.normal(size=(1, 10, 11, 8, 1)).astype(np.float32)
    expected_output = tfmri.signal.dwt(input_data, wavelet, axes=(1, 2, 3))

    self._test_layer(tfmri.layers.DWT3D,
                     {'wavelet': 'haar'},
                     input_data,
                     expected_output)

  def test_idwt_1d(self):
    wavelet = 'haar'
    rng = np.random.default_rng()
    input_data = {k: rng.normal(size=(1, 10, 1)).astype(np.float32)
                  for k in ['a', 'd']}
    expected_output = tfmri.signal.idwt(input_data, wavelet, axes=(1,))

    self._test_layer(tfmri.layers.IDWT1D,
                     {'wavelet': 'haar'},
                     input_data,
                     expected_output)

  def test_idwt_2d(self):
    wavelet = 'haar'
    rng = np.random.default_rng()
    input_data = {k: rng.normal(size=(1, 10, 11, 1)).astype(np.float32)
                  for k in ['aa', 'ad', 'da', 'dd']}
    expected_output = tfmri.signal.idwt(input_data, wavelet, axes=(1, 2))

    self._test_layer(tfmri.layers.IDWT2D,
                     {'wavelet': 'haar'},
                     input_data,
                     expected_output)

  def test_idwt_3d(self):
    wavelet = 'haar'
    rng = np.random.default_rng()
    input_data = {k: rng.normal(size=(1, 10, 11, 8, 1)).astype(np.float32)
                  for k in ['aaa', 'aad', 'ada', 'add',
                            'daa', 'dad', 'dda', 'ddd']}
    expected_output = tfmri.signal.idwt(input_data, wavelet, axes=(1, 2, 3))

    self._test_layer(tfmri.layers.IDWT3D,
                     {'wavelet': 'haar'},
                     input_data,
                     expected_output)


  def _test_layer(self,
                  layer_cls,
                  kwargs,
                  input_data,
                  expected_output):
    input_shape = _nested_shape(input_data)
    input_dtype = _nested_dtype(input_data)

    # Instantiation.
    kwargs = kwargs or {}
    layer = layer_cls(**kwargs)

    # Test weights.
    weights = layer.get_weights()
    layer.set_weights(weights)

    x = _nested_input(input_shape, input_dtype)
    y = layer(x)

    # Check shape inference.
    model = keras.Model(x, y)
    computed_output_shape = layer.compute_output_shape(input_shape)
    computed_output_signature = layer.compute_output_signature(
        _nested_tensor_spec(input_shape, input_dtype))
    computed_output_dtype = _nested_dtype(computed_output_signature)
    actual_output = model.predict(input_data)
    actual_output_shape = _nested_shape(actual_output)
    actual_output_dtype = _nested_dtype(actual_output)
    self.assertEqual(actual_output_shape, computed_output_shape)
    self.assertEqual(actual_output_dtype, computed_output_dtype)

    # Check output.
    if expected_output is not None:
      self.assertAllClose(expected_output, actual_output)

    # Test serialization and weight setting at model level.
    model_config = model.get_config()
    recovered_model = keras.Model.from_config(model_config)
    if model.weights:
      weights = model.get_weights()
      recovered_model.set_weights(weights)
      output = recovered_model.predict(input_data)
      self.assertAllClose(actual_output, output)

    # Validate training.
    model = keras.Model(x, layer(x))
    model.compile('rmsprop',
                  'mse',
                  weighted_metrics=['acc'],
                  run_eagerly=tf.executing_eagerly())
    model.train_on_batch(input_data, actual_output)


def _nested_shape(x):
  return tf.nest.map_structure(lambda x: tf.TensorShape(x.shape), x)

def _nested_dtype(x):
  return tf.nest.map_structure(lambda x: x.dtype, x)

def _nested_input(shape, dtype):
  return tf.nest.map_structure(
      lambda shape, dtype: keras.Input(shape=shape[1:], dtype=dtype),
      shape, dtype)

def _nested_tensor_spec(shape, dtype):
  return tf.nest.map_structure(
      lambda shape, dtype: tf.TensorSpec(shape=shape, dtype=dtype),
      shape, dtype)
