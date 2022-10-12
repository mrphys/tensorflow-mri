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
"""Tests for module `linear_operator`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import functools

import numpy as np
import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_identity_nd
from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.linalg import linear_operator_test_util
from tensorflow_mri.python.util import test_util


FullMatrix = tf.linalg.LinearOperatorFullMatrix
MakeND = linear_operator_nd.LinearOperatorMakeND


rng = np.random.RandomState(0)


class SquareLinearOperatorMakeNDTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  """Tests for `LinearOperatorMakeND`."""
  domain_shape = (3, 2)
  range_shape = (2, 3)
  batch_shape = (2, 1)

  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)

    if ensure_self_adjoint_and_pd:
      matrix = linear_operator_test_util.random_positive_definite_matrix(
          shape, dtype, force_well_conditioned=True)
    else:
      matrix = linear_operator_test_util.random_normal(shape=shape, dtype=dtype)

    if use_placeholder:
      matrix = tf.compat.v1.placeholder_with_default(matrix, shape=None)

    operator = MakeND(
        FullMatrix(matrix,
            is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
            is_positive_definite=True if ensure_self_adjoint_and_pd else None,
            is_square=True),
        [shape[-2]], [shape[-1]],
        is_self_adjoint=True if ensure_self_adjoint_and_pd else None,
        is_positive_definite=True if ensure_self_adjoint_and_pd else None,
        is_square=True)

    return operator, matrix

  def operator_and_operator_nd(self,
                               range_shape=range_shape,
                               domain_shape=domain_shape,
                               batch_shape=batch_shape):
    range_dimension = functools.reduce(lambda x, y: x * y, range_shape)
    domain_dimension = functools.reduce(lambda x, y: x * y, domain_shape)

    matrix = tf.random.uniform(
        batch_shape + (range_dimension, domain_dimension))

    operator = FullMatrix(matrix)
    operator_nd = MakeND(
        FullMatrix(matrix), range_shape, domain_shape)

    return operator, operator_nd

  def random_input(self, domain_shape=domain_shape, batch_shape=batch_shape):
    x_nd = tf.random.normal(batch_shape + domain_shape)
    x = tf.reshape(x_nd, batch_shape + (-1,))
    return x, x_nd

  def random_rhs(self, range_shape=range_shape, batch_shape=batch_shape):
    rhs_nd = tf.random.normal(batch_shape + range_shape)
    rhs = tf.reshape(rhs_nd, batch_shape + (-1,))
    return rhs, rhs_nd

  def test_is_nd_operator(self):
    _, operator_nd = self.operator_and_operator_nd()
    self.assertIsInstance(operator_nd, linear_operator_nd.LinearOperatorND)

  def test_name(self):
    _, operator_nd = self.operator_and_operator_nd()
    self.assertEqual("LinearOperatorFullMatrixND", operator_nd.name)

  def test_static_shapes(self):
    operator, operator_nd = self.operator_and_operator_nd()
    self.assertIsInstance(operator_nd.domain_shape, tf.TensorShape)
    self.assertIsInstance(operator_nd.range_shape, tf.TensorShape)
    self.assertIsInstance(operator_nd.batch_shape, tf.TensorShape)
    self.assertIsInstance(operator_nd.shape, tf.TensorShape)
    self.assertEqual(self.domain_shape, operator_nd.domain_shape)
    self.assertEqual(self.range_shape, operator_nd.range_shape)
    self.assertEqual(self.batch_shape, operator_nd.batch_shape)
    self.assertEqual(operator.shape, operator_nd.shape)

  def test_dynamic_shapes(self):
    operator, operator_nd = self.operator_and_operator_nd()
    self.assertIsInstance(operator_nd.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(operator_nd.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(operator_nd.batch_shape_tensor(), tf.Tensor)
    self.assertIsInstance(operator_nd.shape_tensor(), tf.Tensor)
    self.assertAllEqual(self.domain_shape, self.evaluate(
        operator_nd.domain_shape_tensor()))
    self.assertAllEqual(self.range_shape, self.evaluate(
        operator_nd.range_shape_tensor()))
    self.assertAllEqual(self.batch_shape, self.evaluate(
        operator_nd.batch_shape_tensor()))
    self.assertAllEqual(self.evaluate(operator.shape_tensor()),
                        self.evaluate(operator_nd.shape_tensor()))

  def test_operator_wrong_type(self):
    class Cat():
      def say_hello(self):
        return "meow"

    with self.assertRaisesRegex(TypeError, "must be a LinearOperator"):
      MakeND(Cat(), (2, 3), (3, 2))

  def test_nd_operator_returns_itself(self):
    operator = linear_operator_identity_nd.LinearOperatorIdentityND(
        domain_shape=(2, 3))
    operator_nd = MakeND(operator, (2, 3), (3, 2))
    self.assertIs(operator, operator_nd)

  def test_incompatible_domain_shape_raises(self):
    operator, _ = self.operator_and_operator_nd()
    with self.assertRaisesRegex(
        ValueError, "domain_shape must have the same number of elements"):
      MakeND(
          operator, self.range_shape, (5, 3))

  def test_incompatible_range_shape_raises(self):
    operator, _ = self.operator_and_operator_nd()
    with self.assertRaisesRegex(
        ValueError, "range_shape must have the same number of elements"):
      MakeND(
          operator, (5, 3), self.domain_shape)

  def test_matvec(self):
    operator, operator_nd = self.operator_and_operator_nd()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(operator.matvec(x),
                        operator_nd.matvec(x))
    self.assertAllClose(operator.matvec(rhs, adjoint=True),
                        operator_nd.matvec(rhs, adjoint=True))

  def test_matmul(self):
    operator, operator_nd = self.operator_and_operator_nd()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(
        operator.matmul(x[..., tf.newaxis]),
        operator_nd.matmul(x[..., tf.newaxis]))
    self.assertAllClose(
        operator.matmul(x[..., tf.newaxis, :], adjoint_arg=True),
        operator_nd.matmul(x[..., tf.newaxis, :], adjoint_arg=True))
    self.assertAllClose(
        operator.matmul(rhs[..., tf.newaxis], adjoint=True),
        operator_nd.matmul(rhs[..., tf.newaxis], adjoint=True))
    self.assertAllClose(
        operator.matmul(rhs[..., tf.newaxis, :], adjoint=True, adjoint_arg=True,),
        operator_nd.matmul(rhs[..., tf.newaxis, :], adjoint=True, adjoint_arg=True))

  def test_solvevec(self):
    operator, operator_nd = self.operator_and_operator_nd()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(operator.solvevec(rhs),
                        operator_nd.solvevec(rhs))
    self.assertAllClose(operator.solvevec(x, adjoint=True),
                        operator_nd.solvevec(x, adjoint=True))

  def test_solve(self):
    operator, operator_nd = self.operator_and_operator_nd()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(
        operator.solve(rhs[..., tf.newaxis]),
        operator_nd.solve(rhs[..., tf.newaxis]))
    self.assertAllClose(
        operator.solve(rhs[..., tf.newaxis, :], adjoint_arg=True),
        operator_nd.solve(rhs[..., tf.newaxis, :], adjoint_arg=True))
    self.assertAllClose(
        operator.solve(x[..., tf.newaxis], adjoint=True),
        operator_nd.solve(x[..., tf.newaxis], adjoint=True))
    self.assertAllClose(
        operator.solve(x[..., tf.newaxis, :], adjoint=True, adjoint_arg=True,),
        operator_nd.solve(x[..., tf.newaxis, :], adjoint=True, adjoint_arg=True))

  def test_matvec_nd(self):
    range_shape, domain_shape, batch_shape = (
        self.range_shape, self.domain_shape, self.batch_shape)
    batch_shape = self.batch_shape
    operator, operator_nd = self.operator_and_operator_nd()
    x, x_nd = self.random_input()
    rhs, rhs_nd = self.random_rhs()

    self.assertAllClose(
        tf.reshape(operator.matvec(x), batch_shape + range_shape),
        operator_nd.matvec_nd(x_nd))

    self.assertAllClose(
        tf.reshape(operator.matvec(rhs, adjoint=True), batch_shape + domain_shape),
        operator_nd.matvec_nd(rhs_nd, adjoint=True))

  def test_solvevec_nd(self):
    range_shape, domain_shape, batch_shape = (
        self.range_shape, self.domain_shape, self.batch_shape)
    batch_shape = self.batch_shape
    operator, operator_nd = self.operator_and_operator_nd()
    x, x_nd = self.random_input()
    rhs, rhs_nd = self.random_rhs()

    self.assertAllClose(
        tf.reshape(operator.solvevec(rhs), batch_shape + domain_shape),
        operator_nd.solvevec_nd(rhs_nd))

    self.assertAllClose(
        tf.reshape(operator.solvevec(x, adjoint=True), batch_shape + range_shape),
        operator_nd.solvevec_nd(x_nd, adjoint=True))


class NonSquareLinearOperatorMakeNDTest(
    linear_operator_test_util.NonSquareLinearOperatorDerivedClassTest):
  """Tests for `LinearOperatorMakeND`."""
  def operator_and_matrix(self, build_info, dtype, use_placeholder,
                          ensure_self_adjoint_and_pd=False):
    shape = list(build_info.shape)

    matrix = linear_operator_test_util.random_normal(shape=shape, dtype=dtype)

    if use_placeholder:
      matrix = tf.compat.v1.placeholder_with_default(matrix, shape=None)

    operator = MakeND(FullMatrix(matrix), [shape[-2]], [shape[-1]])

    return operator, matrix


linear_operator_test_util.add_tests(SquareLinearOperatorMakeNDTest)
linear_operator_test_util.add_tests(NonSquareLinearOperatorMakeNDTest)


if __name__ == "__main__":
  tf.test.main()
