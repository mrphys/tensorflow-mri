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
"""Tests for module `linear_operator`."""
# pylint: disable=missing-class-docstring,missing-function-docstring

import functools

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.util import test_util


FullMatrix = tf.linalg.LinearOperatorFullMatrix


class ConvertToLinearOperatorND(test_util.TestCase):
  """Tests for `convert_to_nd_operator`."""
  domain_shape = (3, 2)
  range_shape = (2, 3)
  batch_shape = (2, 1)

  def operators(self,
                range_shape=range_shape,
                domain_shape=domain_shape,
                batch_shape=batch_shape):
    range_dimension = functools.reduce(lambda x, y: x * y, range_shape)
    domain_dimension = functools.reduce(lambda x, y: x * y, domain_shape)

    matrix = tf.random.uniform(
        batch_shape + (range_dimension, domain_dimension))

    linop = FullMatrix(matrix)
    linop_nd = linear_operator_nd.convert_to_nd_operator(
        FullMatrix(matrix), range_shape, domain_shape)

    return linop, linop_nd

  def random_input(self, domain_shape=domain_shape, batch_shape=batch_shape):
    x_nd = tf.random.normal(batch_shape + domain_shape)
    x = tf.reshape(x_nd, batch_shape + (-1,))
    return x, x_nd

  def random_rhs(self, range_shape=range_shape, batch_shape=batch_shape):
    rhs_nd = tf.random.normal(batch_shape + range_shape)
    rhs = tf.reshape(rhs_nd, batch_shape + (-1,))
    return rhs, rhs_nd

  def test_is_nd_operator(self):
    _, linop_nd = self.operators()
    self.assertIsInstance(
        linop_nd, linear_operator_nd.LinearOperatorND)

  def test_bases(self):
    linop, linop_nd = self.operators()
    # self.assertEqual(
    #     linop.__class__.__bases__, (linear_operator.LinearOperator,))
    self.assertEqual(
        linop_nd.__class__.__bases__, (linear_operator_nd.LinearOperatorND,))

  def test_static_shapes(self):
    linop, linop_nd = self.operators()
    self.assertIsInstance(linop_nd.domain_shape, tf.TensorShape)
    self.assertIsInstance(linop_nd.range_shape, tf.TensorShape)
    self.assertIsInstance(linop_nd.batch_shape, tf.TensorShape)
    self.assertIsInstance(linop_nd.shape, tf.TensorShape)
    self.assertEqual(self.domain_shape, linop_nd.domain_shape)
    self.assertEqual(self.range_shape, linop_nd.range_shape)
    self.assertEqual(self.batch_shape, linop_nd.batch_shape)
    self.assertEqual(linop.shape, linop_nd.shape)

  def test_dynamic_shapes(self):
    linop, linop_nd = self.operators()
    self.assertIsInstance(linop_nd.domain_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop_nd.range_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop_nd.batch_shape_tensor(), tf.Tensor)
    self.assertIsInstance(linop_nd.shape_tensor(), tf.Tensor)
    self.assertAllEqual(self.domain_shape, self.evaluate(
        linop_nd.domain_shape_tensor()))
    self.assertAllEqual(self.range_shape, self.evaluate(
        linop_nd.range_shape_tensor()))
    self.assertAllEqual(self.batch_shape, self.evaluate(
        linop_nd.batch_shape_tensor()))
    self.assertAllEqual(self.evaluate(linop.shape_tensor()),
                        self.evaluate(linop_nd.shape_tensor()))

  def test_incompatible_domain_shape_raises(self):
    linop, _ = self.operators()
    with self.assertRaisesRegex(
        ValueError, "domain_shape must have the same number of elements"):
      linear_operator_nd.convert_to_nd_operator(
          linop, self.range_shape, (5, 3))

  def test_incompatible_range_shape_raises(self):
    linop, _ = self.operators()
    with self.assertRaisesRegex(
        ValueError, "range_shape must have the same number of elements"):
      linear_operator_nd.convert_to_nd_operator(
          linop, (5, 3), self.domain_shape)

  def test_matvec(self):
    linop, linop_nd = self.operators()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(linop.matvec(x),
                        linop_nd.matvec(x))
    self.assertAllClose(linop.matvec(rhs, adjoint=True),
                        linop_nd.matvec(rhs, adjoint=True))

  def test_matmul(self):
    linop, linop_nd = self.operators()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(
        linop.matmul(x[..., tf.newaxis]),
        linop_nd.matmul(x[..., tf.newaxis]))
    self.assertAllClose(
        linop.matmul(x[..., tf.newaxis, :], adjoint_arg=True),
        linop_nd.matmul(x[..., tf.newaxis, :], adjoint_arg=True))
    self.assertAllClose(
        linop.matmul(rhs[..., tf.newaxis], adjoint=True),
        linop_nd.matmul(rhs[..., tf.newaxis], adjoint=True))
    self.assertAllClose(
        linop.matmul(rhs[..., tf.newaxis, :], adjoint=True, adjoint_arg=True,),
        linop_nd.matmul(rhs[..., tf.newaxis, :], adjoint=True, adjoint_arg=True))

  def test_solvevec(self):
    linop, linop_nd = self.operators()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(linop.solvevec(rhs),
                        linop_nd.solvevec(rhs))
    self.assertAllClose(linop.solvevec(x, adjoint=True),
                        linop_nd.solvevec(x, adjoint=True))

  def test_solve(self):
    linop, linop_nd = self.operators()
    x, _ = self.random_input()
    rhs, _ = self.random_rhs()
    self.assertAllClose(
        linop.solve(rhs[..., tf.newaxis]),
        linop_nd.solve(rhs[..., tf.newaxis]))
    self.assertAllClose(
        linop.solve(rhs[..., tf.newaxis, :], adjoint_arg=True),
        linop_nd.solve(rhs[..., tf.newaxis, :], adjoint_arg=True))
    self.assertAllClose(
        linop.solve(x[..., tf.newaxis], adjoint=True),
        linop_nd.solve(x[..., tf.newaxis], adjoint=True))
    self.assertAllClose(
        linop.solve(x[..., tf.newaxis, :], adjoint=True, adjoint_arg=True,),
        linop_nd.solve(x[..., tf.newaxis, :], adjoint=True, adjoint_arg=True))

  def test_matvec_nd(self):
    range_shape, domain_shape, batch_shape = (
        self.range_shape, self.domain_shape, self.batch_shape)
    batch_shape = self.batch_shape
    linop, linop_nd = self.operators()
    x, x_nd = self.random_input()
    rhs, rhs_nd = self.random_rhs()

    self.assertAllClose(
        tf.reshape(linop.matvec(x), batch_shape + range_shape),
        linop_nd.matvec_nd(x_nd))

    self.assertAllClose(
        tf.reshape(linop.matvec(rhs, adjoint=True), batch_shape + domain_shape),
        linop_nd.matvec_nd(rhs_nd, adjoint=True))

  def test_solvevec_nd(self):
    range_shape, domain_shape, batch_shape = (
        self.range_shape, self.domain_shape, self.batch_shape)
    batch_shape = self.batch_shape
    linop, linop_nd = self.operators()
    x, x_nd = self.random_input()
    rhs, rhs_nd = self.random_rhs()

    self.assertAllClose(
        tf.reshape(linop.solvevec(rhs), batch_shape + domain_shape),
        linop_nd.solvevec_nd(rhs_nd))

    self.assertAllClose(
        tf.reshape(linop.solvevec(x, adjoint=True), batch_shape + range_shape),
        linop_nd.solvevec_nd(x_nd, adjoint=True))


if __name__ == "__main__":
  tf.test.main()
