# Copyright 2022 University College London. All Rights Reserved.
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
"""Tests for data consistency layers."""

import tensorflow as tf
from tensorflow.python.ops.linalg import linear_operator

from tensorflow_mri.python.layers import data_consistency
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import test_util


class LeastSquaresGradientDescentStepTest(test_util.TestCase):
  def test_general(self):
    @linear_operator.make_composite_tensor
    class LinearOperatorScalarMultiply(linear_operator.LinearOperator):
      def __init__(self, scale):
        parameters = {'scale': scale}
        self.scale = tf.convert_to_tensor(scale)
        super().__init__(dtype=self.scale.dtype, parameters=parameters)

      def _transform(self, x, adjoint=False):
        if adjoint:
          return x * tf.math.conj(self.scale)
        else:
          return x * self.scale
      
      def _domain_shape(self):
        return tf.TensorShape([2])

      def _range_shape(self):
        return self._domain_shape()

    operator = LinearOperatorScalarMultiply(2.0 + 1.0j)
    layer = data_consistency.LeastSquaresGradientDescentStep(operator)

    inputs = [3, 3], [1, 1] 
    result = layer(inputs)
    print(result)
