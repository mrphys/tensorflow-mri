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
"""Composition of linear operators."""

import tensorflow as tf

from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import api_util


@api_util.export("linalg.LinearOperatorComposition")
class LinearOperatorComposition(linear_operator.LinearOperatorMixin,  # pylint: disable=abstract-method
                                tf.linalg.LinearOperatorComposition):
  """Composes one or more linear operators.

  `LinearOperatorComposition` is initialized with a list of operators
  :math:`A_1, A_2, ..., A_J` and represents their composition
  :math:`A_1 A_2 ... A_J`.

  .. note:
    Similar to `tf.linalg.LinearOperatorComposition`_, but with imaging
    extensions.

  Args:
    operators: A `list` of `LinearOperator` objects, each with the same `dtype`
      and composable shape.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form :math:`x^H A x` has positive real part for all
      nonzero :math:`x`. Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.  Default is the individual
      operators names joined with `_o_`.

  .. _tf.linalg.LinearOperatorComposition: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperatorComposition
  """
  def _transform(self, x, adjoint=False):
    # pylint: disable=protected-access
    if adjoint:
      transform_order_list = self.operators
    else:
      transform_order_list = list(reversed(self.operators))

    result = transform_order_list[0]._transform(x, adjoint=adjoint)
    for operator in transform_order_list[1:]:
      result = operator._transform(result, adjoint=adjoint)
    return result

  def _domain_shape(self):
    return self.operators[-1].domain_shape

  def _range_shape(self):
    return self.operators[0].range_shape

  def _batch_shape(self):
    return array_ops.broadcast_static_shapes(
        *[operator.batch_shape for operator in self.operators])

  def _domain_shape_tensor(self):
    return self.operators[-1].domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operators[0].range_shape_tensor()

  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shapes(
        *[operator.batch_shape_tensor() for operator in self.operators])
