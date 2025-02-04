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
"""Composition of linear operators."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


LinearOperatorComposition = api_util.export(
    "linalg.LinearOperatorComposition")(
        doc_util.no_linkcode(
            linear_operator.make_linear_operator(
                tf.linalg.LinearOperatorComposition)))


tf.linalg.LinearOperatorComposition = LinearOperatorComposition


def combined_non_singular_hint(*operators):
  """Returns a hint for the non-singularity of a composition of operators.

  Args:
    *operators: A list of `LinearOperator` objects.

  Returns:
    A boolean, or `None`. Whether the composition of the operators is expected
    to be non-singular.
  """
  # If all operators are non-singular, so is the composition.
  if all(o.is_non_singular is True for o in operators):
    return True

  # If any operator is singular, then the composition is singular too.
  if any(o.is_non_singular is False for o in operators):
    return False

  # In all other cases, we don't know.
  return None


def combined_self_adjoint_hint(*operators, commuting=False):
  """Returns a hint for the self-adjointness of a composition of operators.

  Args:
    *operators: A list of `LinearOperator` objects.

  Returns:
    A boolean, or `None`. Whether the composition of the operators is expected
    to be self-adjoint.
  """
  if commuting:  # The operators commute.
    # If all operators are self-adjoint, then the composition is self-adjoint.
    if all(o.is_self_adjoint is True for o in operators):
      return True

    # If only one operator isn't self-adjoint, then the composition is not
    # self-adjoint.
    self_adjoint_operators = [
        o for o in operators if o.is_self_adjoint is True]
    non_self_adjoint_operators = [
        o for o in operators if o.is_self_adjoint is False]
    if (len(self_adjoint_operators) == len(operators) - 1 and
        len(non_self_adjoint_operators) == 1):
      return False

    # In all other cases, we don't know.
    return None

  # If commutative property is not guaranteed, we don't know anything about
  # the self-adjointness of the output.
  return None


def combined_positive_definite_hint(*operators, commuting=False):
  """Returns a hint for the positive-definiteness of a composition of operators.

  Args:
    *operators: A list of `LinearOperator` objects.

  Returns:
    A boolean, or `None`. Whether the composition of the operators is expected
    to be positive-definite.
  """
  # If all operators are positive-definite, its composition has positive
  # eigenvalues.
  eigvals_are_positive = all(o.is_positive_definite is True for o in operators)

  # Check if the output is expected to be self-adjoint.
  is_self_adjoint = combined_self_adjoint_hint(*operators, commuting=commuting)

  # If their composition is self-adjoint and the
  # eigenvalues are positive, then the composition is positive-definite.
  if eigvals_are_positive is True and is_self_adjoint is True:
    return True

  # Otherwise, we don't know.
  return None


def combined_square_hint(*operators):
  """Returns a hint for the squareness of a composition of operators.

  Args:
    *operators: A list of `LinearOperator` objects.

  Returns:
    A boolean, or `None`. Whether the composition of the operators is expected
    to be square.
  """
  # If all operators are square, so is the composition.
  if all(o.is_square is True for o in operators):
    return True

  # If all operators are square except one which is not, then the sum is
  # not square.
  square_operators = [
      o for o in operators if o.is_square is True]
  non_square_operators = [
      o for o in operators if o.is_square is False]
  if (len(square_operators) == len(operators) - 1 and
      len(non_square_operators) == 1):
    return False

  # In all other cases, we don't know.
  return None


def check_hint(expected, received, name):
  """Checks that a hint is consistent with its expected value.

  Args:
    expected: A boolean, or `None`. The expected value of the hint.
    received: A boolean, or `None`. The received value of the hint.
    name: A string. The name of the hint.

  Raises:
    ValueError: If `expected` and `value` are not consistent.
  """
  if expected is not None and received is not None and expected != received:
    raise ValueError(
        f"Inconsistent {name} hint: expected {expected} based on input "
        f"operators, but got {received}")
  return received if received is not None else expected
