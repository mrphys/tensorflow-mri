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


def matrix_solve_ls_with_broadcast(matrix, rhs, adjoint=False, name=None):
  """Solve systems of linear equations."""
  with tf.name_scope(name or "MatrixSolveWithBroadcast"):
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
