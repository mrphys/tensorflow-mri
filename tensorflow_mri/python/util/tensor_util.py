# Copyright 2021 University College London. All Rights Reserved.
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
"""Utilities for tensors."""

import tensorflow as tf


def cast_to_complex(tensor):
  """Casts a floating-point tensor to the corresponding complex dtype.

  In other words, cast `float32` to `complex64` and `float64` to `complex128`.
  If the tensor is already complex, it retains its type.

  Args:
    tensor: A `Tensor`.

  Returns:
    The cast tensor.
  """
  return tf.cast(tensor, get_complex_dtype(tensor.dtype))


def get_complex_dtype(dtype):
  """Returns the corresponding complex dtype for a given real dtype.

  Args:
    dtype: A `string` or a `DType`. Must be a floating or complex dtype.

  Returns:
    The complex dtype corresponding to the input dtype. If input dtype is
    already complex, the same type is returned.

  Raises:
    ValueError: If `dtype` has no complex equivalent.
  """
  dtypes = {
    tf.float32: tf.complex64,
    tf.float64: tf.complex128,
    tf.complex64: tf.complex64,
    tf.complex128: tf.complex128
  }
  # Canonicalize dtype. Convert strings, NumPy types, to tf.DType.
  dtype = tf.as_dtype(dtype)
  if dtype not in dtypes:
    raise ValueError(f"Data type {dtype} has no complex equivalent.")
  return dtypes[dtype]


def convert_shape_to_tensor(shape, name=None):
  """Convert a static shape to a tensor."""
  if isinstance(shape, (tuple, list)) and not shape:
    dtype = tf.dtypes.int32
  else:
    dtype = None
  return tf.convert_to_tensor(shape, dtype=dtype, name=name)


def convert_partial_shape_to_tensor(shape, name=None):
  """Convert a static shape to a tensor.

  The input shape may be partially known. Unknown dimensions are replaced by
  the special value `-1` in the output tensor.
  """
  shape = tf.TensorShape(shape).as_list() # Canonicalize.
  shape = [-1 if s is None else s for s in shape]
  return tf.convert_to_tensor(shape, dtype=tf.dtypes.int32, name=name)


def object_shape(tensor):
  """Returns the shape of a tensor or an object.

  Args:
    tensor: A `tf.Tensor` or an object with a `shape_tensor` method.

  Returns:
    The shape of the input object.
  """
  if hasattr(tensor, 'shape_tensor'):
    return tensor.shape_tensor()
  return tf.shape(tensor)


def maybe_get_static_value(tensor):
  """Returns the static value of a tensor, if possible.

  If the static value cannot be obtained, returns the input tensor.

  Args:
    tensor: A `tf.Tensor`.

  Returns:
    The static value of the input tensor, or the input tensor itself if its
    static value cannot be obtained.
  """
  static = tf.get_static_value(tensor)
  if static is not None:
    return static
  return tensor


def static_and_dynamic_shapes_from_shape(shape):
  """Returns static and dynamic shapes from tensor shape.

  Args:
    shape: This could be a 1D integer tensor, a tensor shape, a list,
      a tuple or any other valid representation of a tensor shape.

  Returns:
    A tuple of two objects:
      - A `tf.TensorShape` containing the static information that could be
        retrieved from `shape`. This might be a fully defined shape, a partially
        unknown shape or a fully unknown shape depending on the input. The
        information in this object is available during graph construction.
      - A `tf.Tensor` containing dynamic shape information. This always
        contains the full shape information, but it is only guaranteed to be
        available at run time.

  Raises:
    ValueError: If `shape` is not 1D.
  """
  static = tf.TensorShape(tf.get_static_value(shape, partial=True))
  dynamic = tf.convert_to_tensor(shape, tf.int32)
  if dynamic.shape.rank != 1:
    raise ValueError(f"Expected shape to be 1D, got {dynamic}.")
  return static, dynamic
