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
"""Utilities for linear operators."""

import tensorflow as tf
from tensorflow.python.ops.linalg import linear_operator_util


broadcast_matrix_batch_dims = linear_operator_util.broadcast_matrix_batch_dims
split_arg_into_blocks = linear_operator_util.split_arg_into_blocks
_reshape_for_efficiency = linear_operator_util._reshape_for_efficiency  # pylint: disable=protected-access


## Matrix operators.

def matrix_solve_ls_with_broadcast(matrix, rhs, adjoint=False, name=None):
  """Solve systems of linear equations."""
  with tf.name_scope(name or "MatrixSolveLSWithBroadcast"):
    matrix = tf.convert_to_tensor(matrix, name="matrix")
    rhs = tf.convert_to_tensor(rhs, name="rhs", dtype=matrix.dtype)

    # If either matrix/rhs has extra dims, we can reshape to get rid of them.
    matrix, rhs, reshape_inv, still_need_to_transpose = _reshape_for_efficiency(
        matrix, rhs, adjoint_a=adjoint)

    # This will broadcast by brute force if we still need to.
    matrix, rhs = broadcast_matrix_batch_dims([matrix, rhs])

    if adjoint and still_need_to_transpose:
      matrix = tf.linalg.adjoint(matrix)
    solution = tf.linalg.lstsq(matrix, rhs, fast=False)

    return reshape_inv(solution)


## Asserts.

def assert_no_entries_with_modulus_zero(x, message=None, name=None):
  """Returns `Op` that asserts Tensor `x` has no entries with modulus zero.

  Args:
    x: Numeric `Tensor`, real, integer, or complex.
    message: A string message to prepend to failure message.
    name: A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with modulus zero.
  """
  with tf.name_scope(name or "assert_no_entries_with_modulus_zero"):
    x = tf.convert_to_tensor(x, name="x")
    dtype = x.dtype.base_dtype
    should_be_nonzero = tf.math.abs(x)
    zero = tf.convert_to_tensor(0, dtype=dtype.real_dtype)
    return tf.debugging.assert_less(zero, should_be_nonzero, message=message)


def assert_zero_imag_part(x, message=None, name=None):
  """Returns `Op` that asserts Tensor `x` has no non-zero imaginary parts.

  Args:
    x: Numeric `Tensor`, real, integer, or complex.
    message: A string message to prepend to failure message.
    name: A name to give this `Op`.

  Returns:
    An `Op` that asserts `x` has no entries with non-zero imaginary part.
  """
  with tf.name_scope(name or "assert_zero_imag_part"):
    x = tf.convert_to_tensor(x, name="x")
    dtype = x.dtype.base_dtype

    if dtype.is_floating:
      return tf.no_op()

    zero = tf.convert_to_tensor(0, dtype=dtype.real_dtype)
    return tf.debugging.assert_equal(zero, tf.math.imag(x), message=message)


# Other utilities.

def prepare_inner_dims_for_broadcasting(tensor_a,
                                        tensor_b,
                                        batch_dims_a=0,
                                        batch_dims_b=0):
  """Prepares two tensors for broadcasting, separating batch from inner dims.

  Essentially, this function makes sure that both tensors have the same number
  of inner dimensions, so that inner dimensions can be broadcasted with inner
  dimensions, and batch dimensions are broadcasted with batch dimensions.

  For example, given the following tensors:
    - `tensor_a` with shape `(2, 3, 4, 5)`, with 2 batch dimensions.
    - `tensor_b` with shape `(2, 3, 2, 4, 5)`, with 2 batch dimensions.

  This function will return the following:
    - `tensor_a` with shape `(2, 3, 1, 4, 5)`.
    - `tensor_b` with shape `(2, 3, 2, 4, 5)`.

  i.e., the inner dimensions of `tensor_a` are expanded to match the inner
  dimensions of `tensor_b`.

  ```{note}
  This function does not check that the batch/inner dimensions of `tensor_a`
  and `tensor_b` are compatible for broadcasting. It simply makes sure that
  both tensors have the same number of inner dimensions.
  ```
  """
  # Number of inner dimensions (static).
  inner_dims_a = tensor_a.shape.rank - batch_dims_a
  inner_dims_b = tensor_b.shape.rank - batch_dims_b
  if inner_dims_a == inner_dims_b:
    return tensor_a, tensor_b

  # Get shapes of batch and inner dimensions for both tensors.
  shape_a, shape_b = tf.shape_n([tensor_a, tensor_b])
  batch_shape_a = shape_a[:batch_dims_a]
  batch_shape_b = shape_b[:batch_dims_b]
  inner_shape_a = shape_a[batch_dims_a:]
  inner_shape_b = shape_b[batch_dims_b:]

  # Number of inner dimensions (dynamic).
  if inner_dims_a > inner_dims_b:
    extra_dims = inner_dims_a - inner_dims_b
    new_shape_b = tf.concat([batch_shape_b, [1] * extra_dims, inner_shape_b], 0)
    tensor_b = tf.reshape(tensor_b, new_shape_b)
  else:  # inner_dims_a < inner_dims_b
    extra_dims = inner_dims_b - inner_dims_a
    new_shape_a = tf.concat([batch_shape_a, [1] * extra_dims, inner_shape_a], 0)
    tensor_a = tf.reshape(tensor_a, new_shape_a)

  return tensor_a, tensor_b
