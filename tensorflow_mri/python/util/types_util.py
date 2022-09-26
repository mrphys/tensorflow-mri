# Copyright 2021 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Types utilities."""

import tensorflow as tf

SIGNED_INTEGER_TYPES = [tf.int8, tf.int16, tf.int32, tf.int64]
UNSIGNED_INTEGER_TYPES = [tf.uint8, tf.uint16, tf.uint32, tf.uint64]
INTEGER_TYPES = SIGNED_INTEGER_TYPES + UNSIGNED_INTEGER_TYPES

FLOATING_TYPES = [tf.float16, tf.float32, tf.float64]
COMPLEX_TYPES = [tf.complex64, tf.complex128]


def is_ref(x):
  """Evaluates if the object has reference semantics.

  An object is deemed "reference" if it is a `tf.Variable` instance or is
  derived from a `tf.Module` with `dtype` and `shape` properties.

  Args:
    x: Any object.

  Returns:
    is_ref: Python `bool` indicating input is has nonreference semantics, i.e.,
      is a `tf.Variable` or a `tf.Module` with `dtype` and `shape` properties.
  """
  return (
      isinstance(x, tf.Variable) or
      (isinstance(x, tf.Module) and hasattr(x, "dtype") and
       hasattr(x, "shape")))


def assert_not_ref_type(x, arg_name):
  if is_ref(x):
    raise TypeError(
        f"Argument {arg_name} cannot be reference type. Found: {type(x)}.")
