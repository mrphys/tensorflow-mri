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
"""Tests for Keras initializers."""
# pylint: disable=missing-function-docstring

from keras import backend
from keras import models
from keras.engine import input_layer
from keras.layers import core
import numpy as np
import tensorflow as tf
import tensorflow_mri as tfmri


def _compute_fans(shape):
  """Computes the number of input and output units for a weight shape.
  Args:
    shape: Integer shape tuple or TF tensor shape.
  Returns:
    A tuple of integer scalars (fan_in, fan_out).
  """
  if len(shape) < 1:  # Just to avoid errors for constants.
    fan_in = fan_out = 1
  elif len(shape) == 1:
    fan_in = fan_out = shape[0]
  elif len(shape) == 2:
    fan_in = shape[0]
    fan_out = shape[1]
  else:
    # Assuming convolution kernels (2D, 3D, or more).
    # kernel shape: (..., input_depth, depth)
    receptive_field_size = 1
    for dim in shape[:-2]:
      receptive_field_size *= dim
    fan_in = shape[-2] * receptive_field_size
    fan_out = shape[-1] * receptive_field_size
  return int(fan_in), int(fan_out)


@tf.__internal__.test.combinations.generate(
    tf.__internal__.test.combinations.combine(mode=['graph', 'eager']))
class KerasInitializersTest(tf.test.TestCase):
  """Tests for Keras initializers."""
  def _runner(self, init, shape, target_mean=None, target_std=None,  # pylint: disable=missing-function-docstring,unused-argument
              target_max=None, target_min=None, dtype=None):  # pylint: disable=unused-argument
    # The global seed is set so that we can get the same random streams between
    # eager and graph mode when stateful op is used.
    tf.random.set_seed(1337)
    variable = backend.variable(init(shape, dtype), dtype)
    output = backend.get_value(variable)

    # Test output dtype.
    self.assertDTypeEqual(output, dtype or backend.floatx())

    # Test serialization (assumes deterministic behavior).
    config = init.get_config()
    reconstructed_init = init.__class__.from_config(config)

    tf.random.set_seed(1337)
    variable = backend.variable(reconstructed_init(shape, dtype), dtype)
    output_2 = backend.get_value(variable)
    self.assertAllClose(output, output_2, atol=1e-4)

  def test_lecun_uniform(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(1. / fan_in)
      self._runner(
          tfmri.initializers.LecunUniform(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std)

  def test_lecun_uniform_complex(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, fan_out = _compute_fans(tensor_shape)
      std = np.sqrt(2. / (fan_in + fan_out))
      self._runner(
          tfmri.initializers.LecunUniform(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std,
          dtype='complex64')

  def test_glorot_uniform(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, fan_out = _compute_fans(tensor_shape)
      std = np.sqrt(2. / (fan_in + fan_out))
      self._runner(
          tfmri.initializers.GlorotUniform(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std)

  def test_glorot_uniform_complex(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, fan_out = _compute_fans(tensor_shape)
      std = np.sqrt(2. / (fan_in + fan_out))
      self._runner(
          tfmri.initializers.GlorotUniform(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std,
          dtype='complex64')

  def test_he_uniform(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(2. / fan_in)
      self._runner(
          tfmri.initializers.HeUniform(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std)

  def test_he_uniform_complex(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(2. / fan_in)
      self._runner(
          tfmri.initializers.HeUniform(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std,
          dtype='complex64')

  def test_lecun_normal(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(1. / fan_in)
      self._runner(
          tfmri.initializers.LecunNormal(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std)

  def test_lecun_normal_complex(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(1. / fan_in)
      self._runner(
          tfmri.initializers.LecunNormal(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std,
          dtype='complex64')

  def test_glorot_normal(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, fan_out = _compute_fans(tensor_shape)
      std = np.sqrt(2. / (fan_in + fan_out))
      self._runner(
          tfmri.initializers.GlorotNormal(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std)

  def test_glorot_normal_complex(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, fan_out = _compute_fans(tensor_shape)
      std = np.sqrt(2. / (fan_in + fan_out))
      self._runner(
          tfmri.initializers.GlorotNormal(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std,
          dtype='complex64')

  def test_he_normal(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(2. / fan_in)
      self._runner(
          tfmri.initializers.HeNormal(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std)

  def test_he_normal_complex(self):
    tensor_shape = (5, 6, 4, 2)
    with self.cached_session():
      fan_in, _ = _compute_fans(tensor_shape)
      std = np.sqrt(2. / fan_in)
      self._runner(
          tfmri.initializers.HeNormal(seed=123),
          tensor_shape,
          target_mean=0.,
          target_std=std,
          dtype='complex64')

  def test_tfmri_initializer_saving(self):

    inputs = input_layer.Input((10,))
    outputs = core.Dense(
        1, kernel_initializer=tfmri.initializers.GlorotUniform())(inputs)
    model = models.Model(inputs, outputs)
    model2 = model.from_config(model.get_config())
    self.assertIsInstance(model2.layers[1].kernel_initializer,
                          tfmri.initializers.GlorotUniform)

  def test_partition(self):
    with self.cached_session():
      partition_enabled_initializers = [
          tfmri.initializers.LecunUniform(),
          tfmri.initializers.GlorotUniform(),
          tfmri.initializers.HeUniform()
      ]
      for initializer in partition_enabled_initializers:
        got = initializer(
            shape=(4, 2), partition_shape=(2, 2), partition_offset=(0, 0))
        self.assertEqual(got.shape, (2, 2))

      partition_forbidden_initializers = []
      for initializer in partition_forbidden_initializers:
        with self.assertRaisesRegex(
            ValueError,
            "initializer doesn't support partition-related arguments"):
          initializer(
              shape=(4, 2), partition_shape=(2, 2), partition_offset=(0, 0))


if __name__ == '__main__':
  tf.test.main()
