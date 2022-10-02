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


def convert_nonref_to_tensor(value, dtype=None, dtype_hint=None, name=None):
  """Converts the given `value` to a `Tensor` if input is nonreference type.

  This function converts Python objects of various types to `Tensor` objects
  except if the input has nonreference semantics. Reference semantics are
  characterized by `is_ref` and is any object which is a
  `tf.Variable` or instance of `tf.Module`. This function accepts any input
  which `tf.convert_to_tensor` would also.

  Note: This function diverges from default Numpy behavior for `float` and
    `string` types when `None` is present in a Python list or scalar. Rather
    than silently converting `None` values, an error will be thrown.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    dtype_hint: Optional element type for the returned tensor,
      used when dtype is None. In some cases, a caller may not have a
      dtype in mind when converting to a tensor, so dtype_hint
      can be used as a soft preference.  If the conversion to
      `dtype_hint` is not possible, this argument has no effect.
    name: Optional name to use if a new `Tensor` is created.

  Returns:
    tensor: A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value` to `dtype`.
    RuntimeError: If a registered conversion function returns an invalid value.
    ValueError: If the `value` is a tensor not of given `dtype` in graph mode.


  #### Examples:

  ```python

  x = tf.Variable(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> True

  x = tf.constant(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> True

  x = np.array(0.)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> False
  tf.is_tensor(y)
  # ==> True

  x = tfp.util.DeferredTensor(13.37, lambda x: x)
  y = convert_nonref_to_tensor(x)
  x is y
  # ==> True
  tf.is_tensor(y)
  # ==> False
  tf.equal(y, 13.37)
  # ==> True
  ```

  """
  # We explicitly do not use a tf.name_scope to avoid graph clutter.
  if value is None:
    return None
  if is_ref(value):
    if dtype is None:
      return value
    dtype_base = base_dtype(dtype)
    value_dtype_base = base_dtype(value.dtype)
    if dtype_base != value_dtype_base:
      raise TypeError(
          f"Argument `value` must be of dtype `{dtype_name(dtype_base)}` "
          f"Received: `{dtype_name(value_dtype_base)}`.")
    return value
  return tf.convert_to_tensor(
      value, dtype=dtype, dtype_hint=dtype_hint, name=name)


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


def base_dtype(dtype):
  """Returns a non-reference `dtype` based on this `dtype`."""
  dtype = tf.dtypes.as_dtype(dtype)
  if hasattr(dtype, "base_dtype"):
    return dtype.base_dtype
  return dtype


def dtype_name(dtype):
  """Returns the string name for this `dtype`."""
  dtype = tf.dtypes.as_dtype(dtype)
  if hasattr(dtype, "name"):
    return dtype.name
  if hasattr(dtype, "__name__"):
    return dtype.__name__
  return str(dtype)
