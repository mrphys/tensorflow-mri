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
    convolutions. Simply pass `dtype='complex64'` or `dtype='complex128'` to
    the layer constructor.

  .. _tf.keras.layers.${name}: https://www.tensorflow.org/api_docs/python/tf/keras/layers/${name}

""")


def complex_conv(base):
  """Adds complex-valued support to Keras conv layers.

  We need two override two methods in the conv layer:

  1. `__init__`: This allows us to replace some of the Keras initializers
      that do not support complex dtypes by the corresponding TFMRI versions
      with support initialization of complex-valued variables.
  2. `convolution_op`: This implements the complex-valued convolution.

  Args:
    base: The base class to be extended. Must be one of
      `tf.keras.layers.Conv1D`, `tf.keras.layers.Conv2D`, or
      `tf.keras.layers.Conv3D`.

  Returns:
    A subclass of `base` that supports complex-valued convolutions.

  Raises:
    ValueError: If `base` is not one of the supported base classes.
  """
  if not issubclass(base, (tf.keras.layers.Conv1D,
                           tf.keras.layers.Conv2D,
                           tf.keras.layers.Conv3D)):
    raise ValueError(
        f'Expected base class to be a subclass of '
        f'`tf.keras.layers.ConvND`, but got {base}.')

  def __init__(self, *args, **kwargs):  # pylint: disable=invalid-name
    # If the requested initializer is one of those provided by TFMRI, prefer
    # the TFMRI version.
    kernel_initializer = kwargs.get('kernel_initializer', 'glorot_uniform')
    if (isinstance(kernel_initializer, str) and
        kernel_initializer in TFMRI_INITIALIZERS):
      kwargs['kernel_initializer'] = TFMRI_INITIALIZERS[kernel_initializer]()

    bias_initializer = kwargs.get('bias_initializer', 'zeros')
    if (isinstance(bias_initializer, str) and
        bias_initializer in TFMRI_INITIALIZERS):
      kwargs['bias_initializer'] = TFMRI_INITIALIZERS[bias_initializer]()

    return base.__init__(self, *args, **kwargs)

  def convolution_op(self, inputs, kernel):
    # If dtype is not complex, fall back to original implementation.
    if not tf.as_dtype(self.dtype).is_complex:
      return base.convolution_op(self, inputs, kernel)

    inputs_real = tf.math.real(inputs)
    inputs_imag = tf.math.imag(inputs)
    kernel_real = tf.math.real(kernel)
    kernel_imag = tf.math.imag(kernel)

    outputs_real = (
        base.convolution_op(self, inputs_real, kernel_real) -
        base.convolution_op(self, inputs_imag, kernel_imag))

    outputs_imag = (
        base.convolution_op(self, inputs_real, kernel_imag) +
        base.convolution_op(self, inputs_imag, kernel_real))

    return tf.dtypes.complex(outputs_real, outputs_imag)

  # Dynamically create a subclass of `base` with the same name as `base` and
  # with the overriden `convolution_op` method.
  subclass = type(base.__name__, (base,), {'__init__': __init__,
                                           'convolution_op': convolution_op})

  # Copy docs from the base class, adding the extra note.
  docstring = base.__doc__
  doclines = docstring.split('\n')
  doclines[1:1] = EXTENSION_NOTE.substitute(name=base.__name__).splitlines()
  subclass.__doc__ = '\n'.join(doclines)

  return subclass


# Define the complex-valued conv layers. We use a composition of three
# decorators:
#  1. `complex_conv`: Adds complex-valued support to a Keras conv layer.
#  2. `register_keras_serializable`: Registers the new layer with the Keras
#     serialization framework.
#  3. `export`: Exports the new layer to the TFMRI API.
Conv1D = api_util.export("layers.Conv1D", "layers.Convolution1D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_conv(tf.keras.layers.Conv1D)))


Conv2D = api_util.export("layers.Conv2D", "layers.Convolution2D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_conv(tf.keras.layers.Conv2D)))


Conv3D = api_util.export("layers.Conv3D", "layers.Convolution3D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_conv(tf.keras.layers.Conv3D)))
