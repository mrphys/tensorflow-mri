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
"""Keras initializers."""

import inspect

import keras

from tensorflow_mri.python.initializers import initializers
from tensorflow_mri.python.util import api_util


TFMRI_INITIALIZERS = {
    'VarianceScaling': initializers.VarianceScaling,
    'GlorotNormal': initializers.GlorotNormal,
    'GlorotUniform': initializers.GlorotUniform,
    'HeNormal': initializers.HeNormal,
    'HeUniform': initializers.HeUniform,
    'LecunNormal': initializers.LecunNormal,
    'LecunUniform': initializers.LecunUniform,
    'variance_scaling': initializers.VarianceScaling,
    'glorot_normal': initializers.GlorotNormal,
    'glorot_uniform': initializers.GlorotUniform,
    'he_normal': initializers.HeNormal,
    'he_uniform': initializers.HeUniform,
    'lecun_normal': initializers.LecunNormal,
    'lecun_uniform': initializers.LecunUniform,
}


@api_util.export("initializers.serialize")
def serialize(initializer):
  """Serialize a Keras initializer.

  ```{note}
  This function is a drop-in replacement for `tf.keras.initializers.serialize`.
  ```

  Args:
    initializer: A Keras initializer.

  Returns:
    A serialized Keras initializer.
  """
  return keras.initializers.serialize(initializer)


@api_util.export("initializers.deserialize")
def deserialize(config, custom_objects=None):
  """Deserialize a Keras initializer.

  ```{note}
  This function is a drop-in replacement for
  `tf.keras.initializers.deserialize`. The only difference is that this function
  has built-in knowledge of TFMRI initializers. Where a TFMRI initializer exists
  that replaces the corresponding Keras initializer, this function prefers the
  TFMRI initializer.
  ```

  Args:
    config: A Keras initializer configuration.
    custom_objects: Optional dictionary mapping names (strings) to custom
      classes or functions to be considered during deserialization.

  Returns:
    A Keras initializer.
  """
  custom_objects = {**TFMRI_INITIALIZERS, **(custom_objects or {})}
  return keras.initializers.deserialize(config, custom_objects)


@api_util.export("initializers.get")
def get(identifier):
  """Retrieve a Keras initializer by the identifier.

  ```{note}
  This function is a drop-in replacement for
  `tf.keras.initializers.get`. The only difference is that this function
  has built-in knowledge of TFMRI initializers. Where a TFMRI initializer exists
  that replaces the corresponding Keras initializer, this function prefers the
  TFMRI initializer.
  ```

  The `identifier` may be the string name of a initializers function or class (
  case-sensitively).

  >>> identifier = 'Ones'
  >>> tfmri.initializers.deserialize(identifier)
  <...keras.initializers.initializers_v2.Ones...>

  You can also specify `config` of the initializer to this function by passing
  dict containing `class_name` and `config` as an identifier. Also note that the
  `class_name` must map to a `Initializer` class.

  >>> cfg = {'class_name': 'Ones', 'config': {}}
  >>> tfmri.initializers.deserialize(cfg)
  <...keras.initializers.initializers_v2.Ones...>

  In the case that the `identifier` is a class, this method will return a new
  instance of the class by its constructor.

  Args:
    identifier: A `str` or `dict` containing the initializer name or
      configuration.

  Returns:
    An initializer instance based on the input identifier.

  Raises:
    ValueError: If the input identifier is not a supported type or in a bad
      format.
  """
  if identifier is None:
    return None
  if isinstance(identifier, dict):
    return deserialize(identifier)
  elif isinstance(identifier, str):
    identifier = str(identifier)
    return deserialize(identifier)
  elif callable(identifier):
    if inspect.isclass(identifier):
      identifier = identifier()
    return identifier
  else:
    raise ValueError('Could not interpret initializer identifier: ' +
                     str(identifier))
