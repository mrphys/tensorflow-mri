# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Reshaping layers."""

import string

import tensorflow as tf

from tensorflow_mri.python.util import api_util


EXTENSION_NOTE = string.Template("""

  ```{note}
    This layer can be used as a drop-in replacement for
    `tf.keras.layers.${name}`. However, this one also supports complex-valued
    operations. Simply pass `dtype='complex64'` or `dtype='complex128'` to the
    layer constructor.
  ```

""")


def complex_reshape(base):
  """Adds complex-valued support to a Keras reshaping layer.

  We need the init method in the pooling layer to replace the `pool_function`
  attribute with a function that supports complex inputs.

  Args:
    base: The base class to be extended.

  Returns:
    A subclass of `base` that supports complex-valued pooling.

  Raises:
    ValueError: If `base` is not one of the supported base classes.
  """
  if issubclass(base, (tf.keras.layers.UpSampling1D,
                       tf.keras.layers.UpSampling2D,
                       tf.keras.layers.UpSampling3D)):
    def call(self, inputs):
      if tf.as_dtype(self.dtype).is_complex:
        return tf.dtypes.complex(
            base.call(self, tf.math.real(inputs)),
            base.call(self, tf.math.imag(inputs)))

      # For real values, we can just use the regular reshape function.
      return base.call(self, inputs)

  else:
    raise ValueError(f'Unexpected base class: {base}')

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
#  1. `complex_reshape`: Adds complex-valued support to a Keras reshape layer.
#  2. `register_keras_serializable`: Registers the new layer with the Keras
#     serialization framework.
#  3. `export`: Exports the new layer to the TFMRI API.
UpSampling1D = api_util.export("layers.UpSampling1D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_reshape(tf.keras.layers.UpSampling1D)))


UpSampling2D = api_util.export("layers.UpSampling2D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_reshape(tf.keras.layers.UpSampling2D)))


UpSampling3D = api_util.export("layers.UpSampling3D")(
    tf.keras.utils.register_keras_serializable(package='MRI')(
    complex_reshape(tf.keras.layers.UpSampling3D)))
