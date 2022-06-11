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
"""Convolutional layers."""

import string

import tensorflow as tf

from tensorflow_mri.python.initializers import TFMRI_INITIALIZERS
from tensorflow_mri.python.util import api_util


EXTENSION_NOTE = string.Template("""

  .. note::
    This layer can be used as a drop-in replacement for
    `tf.keras.layers.${name}`_. However, this one also supports complex-valued
    pooling. Simply pass `dtype='complex64'` or `dtype='complex128'` to the
    layer constructor.

  .. _tf.keras.layers.${name}: https://www.tensorflow.org/api_docs/python/tf/keras/layers/${name}

""")


def complex_pool(base):
  """Adds complex-valued support to Keras pooling layers.

  We need the init method in the pooling layer to replace the `pool_function`
  attribute with a function that supports complex inputs.

  Args:
    base: The base class to be extended. Must be one of `MaxPool1D`,
      `MaxPool2D`, `MaxPool3D`, `AvgPool1D`, `AvgPool2D` and `AvgPool3D`.

  Returns:
    A subclass of `base` that supports complex-valued pooling.

  Raises:
    ValueError: If `base` is not one of the supported base classes.
  """
  if not issubclass(base, (tf.keras.layers.MaxPool1D,
                           tf.keras.layers.MaxPool2D,
                           tf.keras.layers.MaxPool3D)):
    raise ValueError(f'Unexpected base class: {base}')

  def call(self, inputs):  # pylint: disable=invalid-name
    if tf.as_dtype(self.dtype).is_complex:
      # For complex numbers the max is computed according to the magnitude
      # or absolute value of the complex input. To do this we rely on
      # `tf.nn.max_pool_with_argmax`, called on the magnitude, which
      # returns the indices of the maximum values as well as the values
      # themselves. Then we can use these indices to recover the full
      # complex values. Unfortunately, `tf.nn.max_pool_with_argmax` only
      # supports 2D inputs, so for the 1D and 3D implementations we need
      # to reshape the input to 2D.
      is_1d = isinstance(self, tf.keras.layers.MaxPool1D)
      is_3d = isinstance(self, tf.keras.layers.MaxPool3D)
      if self.data_format == "channels_first":
        # `tf.nn.max_pool_with_argmax` does not support `channels_first`,
        # so we must handle this case manually.
        # TODO(jmontalt): remove this when TF pooling is feature-complete.
        if is_1d:
          perm = (0, 2, 1)
          inv_perm = (0, 2, 1)
        elif is_3d:
          perm = (0, 2, 3, 4, 1)
          inv_perm = (0, 4, 1, 2, 3)
        else:
          perm = (0, 2, 3, 1)
          inv_perm = (0, 3, 1, 2)
        inputs = tf.transpose(inputs, perm)

      ksize = (1,) + self.pool_size + (1,)
      strides = (1,) + self.strides + (1,)

      # Reshape 1D and 3D inputs to 2D.
      if is_1d:
        pad_axis = 2
        inputs = tf.expand_dims(inputs, pad_axis)
        ksize = ksize + (1,)
        strides = strides + (1,)

      # Perform magnitude-based max-pooling.
      outputs, indices = tf.nn.max_pool_with_argmax(
          tf.math.abs(inputs),
          ksize=ksize,
          strides=strides,
          padding=self.padding.upper(),
          data_format="NHWC",
          include_batch_in_index=True)

      outputs = tf.reshape(
          tf.gather(tf.reshape(inputs, [-1]), indices),  # pylint: disable=no-value-for-parameter
          tf.shape(outputs))

      if is_1d:
        outputs = tf.squeeze(outputs, pad_axis)

      if self.data_format == "channels_first":
        outputs = tf.transpose(outputs, inv_perm)

      return outputs

    else:
      # For real values, we can just use the regular pooling function.
      return base.call(self, inputs)

  # Dynamically create a subclass of `base` with the same name as `base` and
  # with the overriden `convolution_op` method.
  subclass = type(base.__name__, (base,), {'call': call})

  # Copy docs from the base class, adding the extra note.
  docstring = base.__doc__
  doclines = docstring.split('\n')
  doclines[1:1] = EXTENSION_NOTE.substitute(name=base.__name__).splitlines()
  subclass.__doc__ = '\n'.join(doclines)

  return subclass


# Define the complex-valued pooling layers. We use a composition of three
# decorators:
#  1. `complex_pool`: Adds complex-valued support to a Keras pooling layer.
#  2. `register_keras_serializable`: Registers the new layer with the Keras
#     serialization framework.
#  3. `export`: Exports the new layer to the TFMRI API.
MaxPooling1D = api_util.export("layers.MaxPool1D", "layers.MaxPooling1D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_pool(tf.keras.layers.MaxPooling1D)))


MaxPooling2D = api_util.export("layers.MaxPool2D", "layers.MaxPooling2D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_pool(tf.keras.layers.MaxPooling2D)))


MaxPooling3D = api_util.export("layers.MaxPool3D", "layers.MaxPooling3D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_pool(tf.keras.layers.MaxPooling3D)))
