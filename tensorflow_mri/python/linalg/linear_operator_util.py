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

import string

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_algebra


def with_mri_extensions(op_cls):
  """Adds TensorFlow MRI extensions to a linear operator.

  Args:
    op_cls: A `LinearOperator` class.

  Returns:
    A patched `LinearOperator` class.
  """
  attrs = {
      "solve_ls": solve_ls,
      "_solve_ls": _solve_ls,
      "solvevec_ls": solvevec_ls,
      "_solvevec_ls": _solvevec_ls
  }

  for name, attr in attrs.items():
    if not hasattr(op_cls, name):
      setattr(op_cls, name, attr)

  if is_tf_builtin(op_cls):
    op_cls = update_docstring(op_cls)

  return op_cls


def update_docstring(op_cls):
  """Adds a notice to the docstring."""
  tf_builtin_compatibility_notice = string.Template("""
  ```{rubric} Compatibility with core TensorFlow
  ```
  This operator is a drop-in replacement for
  `tf.linalg.${class_name}` but has been patched by TensorFlow MRI
  to support additional functionality including `solve_ls` and `solvevec_ls`.
  """).substitute(class_name=op_cls.__name__)

  docstring = op_cls.__doc__
  doclines = docstring.split('\n')
  doclines += tf_builtin_compatibility_notice.split('\n')
  docstring = '\n'.join(doclines)
  op_cls.__doc__ = docstring

  return op_cls


def is_tf_builtin(op_cls):
  """Returns `True` if `op_cls` is a built-in linear operator."""
  return hasattr(tf.linalg, op_cls.__name__)


def solve_ls(self, rhs, adjoint=False, adjoint_arg=False, name="solve_ls"):
  """Solve the (batch) linear system $A X = B$ in the least-squares sense.

  Given $A$ represented by this linear operator with shape `[..., M, N]`,
  computes the least-squares solution $X$ to the batch of linear systems
  $A X = B$. For systems without an exact solution, returns a "best fit"
  solution in the least squares sense. For systems with multiple solutions,
  returns the solution with the smallest Euclidean norm.

  This is equivalent to solving for the normal equations $A^H A X = A^H B$.

  Args:
    rhs: A `tf.Tensor` with same `dtype` as this operator and shape
      `[..., M, K]`. `rhs` is treated like a [batch] matrix meaning for
      every set of leading dimensions, the last two dimensions define a
      matrix.
    adjoint: A boolean. If `True`, solve the system involving the adjoint
      of this operator, $A^H X = B$. Default is `False`.
    adjoint_arg: A boolean. If `True`, solve $A X = B^H$ where $B^H$ is the
      Hermitian transpose (transposition and complex conjugation). Default
      is `False`.
    name: A name scope to use for ops added by this method.

  Returns:
    A `tf.Tensor` with shape `[..., N, K]` and same `dtype` as `rhs`.
  """
  if isinstance(rhs, linear_operator.LinearOperator):
    left_operator = self.adjoint() if adjoint else self
    right_operator = rhs.adjoint() if adjoint_arg else rhs

    if (right_operator.range_dimension is not None and
        left_operator.domain_dimension is not None and
        right_operator.range_dimension != left_operator.domain_dimension):
      raise ValueError(
          "Operators are incompatible. Expected `rhs` to have dimension"
          " {} but got {}.".format(
              left_operator.domain_dimension, right_operator.range_dimension))
    with self._name_scope(name):  # pylint: disable=not-callable
      return linear_operator_algebra.solve_ls(left_operator, right_operator)

  with self._name_scope(name):  # pylint: disable=not-callable
    rhs = tf.convert_to_tensor(rhs, name="rhs")
    self._check_input_dtype(rhs)

    self_dim = -1 if adjoint else -2
    arg_dim = -1 if adjoint_arg else -2
    tf.compat.dimension_at_index(
        self.shape, self_dim).assert_is_compatible_with(
            rhs.shape[arg_dim])

    return self._solve_ls(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

def _solve_ls(self, rhs, adjoint=False, adjoint_arg=False):
  """Default implementation of `_solve_ls`."""
  raise NotImplementedError(
      f"solve_ls is not implemented for {self.__class__.__name__}.")

def solvevec_ls(self, rhs, adjoint=False, name="solvevec_ls"):
  """Solve the linear system $A x = b$ in the least-squares sense.

  Given $A$ represented by this linear operator with shape `[..., M, N]`,
  computes the least-squares solution $x$ to the linear system $A x = b$.
  For systems without an exact solution, returns a "best fit" solution in
  the least squares sense. For systems with multiple solutions, returns the
  solution with the smallest Euclidean norm.

  This is equivalent to solving for the normal equations $A^H A x = A^H b$.

  Args:
    rhs: A `tf.Tensor` with same `dtype` as this operator and shape
      `[..., M]`. `rhs` is treated like a [batch] matrix meaning for
      every set of leading dimensions, the last two dimensions define a
      matrix.
    adjoint: A boolean. If `True`, solve the system involving the adjoint
      of this operator, $A^H x = b$. Default is `False`.
    name: A name scope to use for ops added by this method.

  Returns:
    A `tf.Tensor` with shape `[..., N]` and same `dtype` as `rhs`.
  """
  with self._name_scope(name):  # pylint: disable=not-callable
    rhs = tf.convert_to_tensor(rhs, name="rhs")
    self._check_input_dtype(rhs)
    self_dim = -1 if adjoint else -2
    tf.compat.dimension_at_index(
        self.shape, self_dim).assert_is_compatible_with(rhs.shape[-1])

    return self._solvevec_ls(rhs, adjoint=adjoint)

def _solvevec_ls(self, rhs, adjoint=False):
  """Default implementation of `_solvevec_ls`."""
  rhs_mat = tf.expand_dims(rhs, axis=-1)
  solution_mat = self.solve_ls(rhs_mat, adjoint=adjoint)
  return tf.squeeze(solution_mat, axis=-1)
