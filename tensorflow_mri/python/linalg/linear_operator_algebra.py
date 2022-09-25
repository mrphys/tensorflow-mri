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
"""Linear operator algebra."""

import tensorflow as tf

from tensorflow.python.ops.linalg import linear_operator_algebra


RegisterAdjoint = linear_operator_algebra.RegisterAdjoint
RegisterInverse = linear_operator_algebra.RegisterInverse
RegisterMatmul = linear_operator_algebra.RegisterMatmul
RegisterSolve = linear_operator_algebra.RegisterSolve
_registered_function = linear_operator_algebra._registered_function  # pylint: disable=protected-access


_ADD = {}
_PSEUDO_INVERSE = {}
_LEASTSQ = {}


def _registered_add(type_a, type_b):
  """Get the Add function registered for classes a and b."""
  return _registered_function([type_a, type_b], _ADD)


def _registered_pseudo_inverse(type_a):
  """Get the PseudoInverse function registered for class a."""
  return _registered_function([type_a], _PSEUDO_INVERSE)


def _registered_lstsq(type_a):
  """Get the SolveLS function registered for class a."""
  return _registered_function([type_a], _LEASTSQ)


def add(lin_op_a, lin_op_b, name=None):
  """Compute lin_op_a.add(lin_op_b).

  Args:
    lin_op_a: The LinearOperator on the left.
    lin_op_b: The LinearOperator on the right.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the addition between `lin_op_a` and
      `lin_op_b`.

  Raises:
    NotImplementedError: If no add method is defined between types of
      `lin_op_a` and `lin_op_b`.
  """
  add_fn = _registered_add(type(lin_op_a), type(lin_op_b))
  if add_fn is None:
    raise NotImplementedError("No add registered for {}.add({})".format(
        type(lin_op_a), type(lin_op_b)))

  with tf.name_scope(name or "Add"):
    return add_fn(lin_op_a, lin_op_b)


def pseudo_inverse(lin_op_a, name=None):
  """Get the Pseudo-Inverse associated to lin_op_a.

  Args:
    lin_op_a: The LinearOperator to decompose.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the inverse of `lin_op_a`.

  Raises:
    NotImplementedError: If no Pseudo-Inverse method is defined for the
      LinearOperator type of `lin_op_a`.
  """
  pseudo_inverse_fn = _registered_pseudo_inverse(type(lin_op_a))
  if pseudo_inverse_fn is None:
    raise ValueError("No pseudo-inverse registered for {}".format(
        type(lin_op_a)))

  with tf.name_scope(name or "PseudoInverse"):
    return pseudo_inverse_fn(lin_op_a)


def lstsq(lin_op_a, lin_op_b, name=None):
  """Compute lin_op_a.lstsq(lin_op_b).

  Args:
    lin_op_a: The LinearOperator on the left.
    lin_op_b: The LinearOperator on the right.
    name: Name to use for this operation.

  Returns:
    A LinearOperator that represents the lstsq between `lin_op_a` and
      `lin_op_b`.

  Raises:
    NotImplementedError: If no lstsq method is defined between types of
      `lin_op_a` and `lin_op_b`.
  """
  solve_fn = _registered_lstsq(type(lin_op_a), type(lin_op_b))
  if solve_fn is None:
    raise ValueError("No solve registered for {}.solve({})".format(
        type(lin_op_a), type(lin_op_b)))

  with tf.name_scope(name or "SolveLS"):
    return solve_fn(lin_op_a, lin_op_b)


class RegisterAdd:
  """Decorator to register an Add implementation function.

  Usage:

  @linear_operator_algebra.RegisterAdd(
    lin_op.LinearOperatorFullMatrix,
    lin_op.LinearOperatorFullMatrix)
  def _add_full_matrix(a, b):
    # Return the new full matrix.
  """

  def __init__(self, lin_op_cls_a, lin_op_cls_b):
    """Initialize the LinearOperator registrar.

    Args:
      lin_op_cls_a: the class of the LinearOperator to multiply.
      lin_op_cls_b: the class of the second LinearOperator to multiply.
    """
    self._key = (lin_op_cls_a, lin_op_cls_b)

  def __call__(self, add_fn):
    """Perform the Add registration.

    Args:
      add_fn: The function to use for the Add.

    Returns:
      add_fn

    Raises:
      TypeError: if add_fn is not a callable.
      ValueError: if an Add function has already been registered for
        the given argument classes.
    """
    if not callable(add_fn):
      raise TypeError(
          "add_fn must be callable, received: {}".format(add_fn))
    if self._key in _ADD:
      raise ValueError("Add({}, {}) has already been registered.".format(
          self._key[0].__name__,
          self._key[1].__name__))
    _ADD[self._key] = add_fn
    return add_fn
