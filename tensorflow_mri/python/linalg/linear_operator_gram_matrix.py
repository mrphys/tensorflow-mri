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
"""Gram matrix of a linear operator."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_addition
from tensorflow_mri.python.linalg import linear_operator_composition
from tensorflow_mri.python.linalg import linear_operator_identity
from tensorflow_mri.python.util import api_util


@api_util.export("linalg.LinearOperatorGramMatrix")
class LinearOperatorGramMatrix(linear_operator.LinearOperator):  # pylint: disable=abstract-method
  r"""Linear operator representing the Gram matrix of an operator.

  If $A$ is a `LinearOperator`, this operator is equivalent to
  $A^H A$.

  The Gram matrix of $A$ appears in the normal equation
  $A^H A x = A^H b$ associated with the least squares problem
  ${\mathop{\mathrm{argmin}}_x} {\left \| Ax-b \right \|_2^2}$.

  This operator is self-adjoint and positive definite. Therefore, linear systems
  defined by this linear operator can be solved using the conjugate gradient
  method.

  This operator supports the optional addition of a regularization parameter
  $\lambda$ and a transform matrix $T$. If these are provided,
  this operator becomes $A^H A + \lambda T^H T$. This appears
  in the regularized normal equation
  $\left ( A^H A + \lambda T^H T \right ) x = A^H b + \lambda T^H T x_0$,
  associated with the regularized least squares problem
  ${\mathop{\mathrm{argmin}}_x} {\left \| Ax-b \right \|_2^2 + \lambda \left \| T(x-x_0) \right \|_2^2}$.

  Args:
    operator: A `tfmri.linalg.LinearOperator`. The operator $A$ whose Gram
      matrix is represented by this linear operator.
    reg_parameter: A `Tensor` of shape `[B1, ..., Bb]` and real dtype.
      The regularization parameter $\lambda$. Defaults to 0.
    reg_operator: A `tfmri.linalg.LinearOperator`. The regularization transform
      $T$. Defaults to the identity.
    gram_operator: A `tfmri.linalg.LinearOperator`. The Gram matrix
      $A^H A$. This may be optionally provided to use a specialized
      Gram matrix implementation. Defaults to `None`.
    is_non_singular: Expect that this operator is non-singular.
    is_self_adjoint: Expect that this operator is equal to its Hermitian
      transpose.
    is_positive_definite: Expect that this operator is positive definite,
      meaning the quadratic form $x^H A x$ has positive real part for all
      nonzero $x$.  Note that we do not require the operator to be
      self-adjoint to be positive-definite.
    is_square: Expect that this operator acts like square [batch] matrices.
    name: A name for this `LinearOperator`.
  """
  def __init__(self,
               operator,
               reg_parameter=None,
               reg_operator=None,
               gram_operator=None,
               is_non_singular=None,
               is_self_adjoint=True,
               is_positive_definite=True,
               is_square=True,
               name=None):
    parameters = dict(
        operator=operator,
        reg_parameter=reg_parameter,
        reg_operator=reg_operator,
        is_non_singular=is_non_singular,
        is_self_adjoint=is_self_adjoint,
        is_positive_definite=is_positive_definite,
        is_square=is_square,
        name=name)
    self._operator = operator
    self._reg_parameter = reg_parameter
    self._reg_operator = reg_operator
    self._gram_operator = gram_operator
    if gram_operator is not None:
      self._composed = gram_operator
    else:
      self._composed = linear_operator_composition.LinearOperatorComposition(
          operators=[self._operator.H, self._operator])

    if not is_self_adjoint:
      raise ValueError("A Gram matrix is always self-adjoint.")
    if not is_positive_definite:
      raise ValueError("A Gram matrix is always positive-definite.")
    if not is_square:
      raise ValueError("A Gram matrix is always square.")

    if self._reg_parameter is not None:
      reg_operator_gm = linear_operator_identity.LinearOperatorScaledIdentity(
          domain_shape=self._operator.domain_shape,
          multiplier=tf.cast(self._reg_parameter, self._operator.dtype))
      if self._reg_operator is not None:
        reg_operator_gm = linear_operator_composition.LinearOperatorComposition(
            operators=[reg_operator_gm,
                       self._reg_operator.H,
                       self._reg_operator])
      self._composed = linear_operator_addition.LinearOperatorAddition(
          operators=[self._composed, reg_operator_gm])

    super().__init__(operator.dtype,
                     is_non_singular=is_non_singular,
                     is_self_adjoint=is_self_adjoint,
                     is_positive_definite=is_positive_definite,
                     is_square=is_square,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):
    return self._composed.transform(x, adjoint=adjoint)

  def _domain_shape(self):
    return self.operator.domain_shape

  def _range_shape(self):
    return self.operator.domain_shape

  def _batch_shape(self):
    return self.operator.batch_shape

  def _domain_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _range_shape_tensor(self):
    return self.operator.domain_shape_tensor()

  def _batch_shape_tensor(self):
    return self.operator.batch_shape_tensor()

  @property
  def operator(self):
    return self._operator
