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
"""Utilities for testing linear operators."""

import itertools

import tensorflow as tf

from tensorflow.python.framework import test_util
from tensorflow.python.ops.linalg import linear_operator_test_util


def add_tests(test_cls):
  # Call original add_tests.
  linear_operator_test_util.add_tests(test_cls)

  test_name_dict = {
      "pseudo_inverse": _test_pseudo_inverse
  }
  optional_tests = []
  tests_with_adjoint_args = []

  for name, test_template_fn in test_name_dict.items():
    if name in test_cls.skip_these_tests():
      continue
    if name in optional_tests and name not in test_cls.optional_tests():
      continue

    for dtype, use_placeholder, shape_info in itertools.product(
        test_cls.dtypes_to_test(),
        test_cls.use_placeholder_options(),
        test_cls.operator_shapes_infos()):
      base_test_name = "_".join([
          "test", name, "_shape={},dtype={},use_placeholder={}".format(
              shape_info.shape, dtype, use_placeholder)])
      if name in tests_with_adjoint_args:
        for adjoint in test_cls.adjoint_options():
          for adjoint_arg in test_cls.adjoint_arg_options():
            test_name = base_test_name + ",adjoint={},adjoint_arg={}".format(
                adjoint, adjoint_arg)
            if hasattr(test_cls, test_name):
              raise RuntimeError("Test %s defined more than once" % test_name)
            setattr(
                test_cls,
                test_name,
                test_util.run_deprecated_v1(
                    test_template_fn(  # pylint: disable=too-many-function-args
                        use_placeholder, shape_info, dtype, adjoint,
                        adjoint_arg, test_cls.use_blockwise_arg())))
      else:
        if hasattr(test_cls, base_test_name):
          raise RuntimeError("Test %s defined more than once" % base_test_name)
        setattr(
            test_cls,
            base_test_name,
            test_util.run_deprecated_v1(test_template_fn(
                use_placeholder, shape_info, dtype)))


SquareLinearOperatorDerivedClassTest = (
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest)


def _test_pseudo_inverse(use_placeholder, shapes_info, dtype):
  def test_pseudo_inverse(self):
    with self.session(graph=tf.Graph()) as sess:
      sess.graph.seed = 87654321
      operator, mat = self.operator_and_matrix(
          shapes_info, dtype, use_placeholder=use_placeholder)
      op_inverse_v, mat_inverse_v = sess.run([
          operator.pseudo_inverse().to_dense(), tf.linalg.pinv(mat)])
      self.assertAC(op_inverse_v, mat_inverse_v, check_dtype=True)
  return test_pseudo_inverse
