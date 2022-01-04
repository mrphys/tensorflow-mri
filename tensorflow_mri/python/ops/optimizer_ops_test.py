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
"""Tests for module `optimizer_ops`."""

import tensorflow as tf

from tensorflow_mri.python.ops import convex_ops
from tensorflow_mri.python.ops import optimizer_ops
from tensorflow_mri.python.util import test_util


class ADMMTest(test_util.TestCase):

  def test_lasso(self):
    operator = tf.linalg.LinearOperatorFullMatrix([[1., -10],
                                                   [1., 10.],
                                                   [1., 0.]])
    rhs = tf.convert_to_tensor([2., 2., 2.])
    lambda_ = 1.0

    f = convex_ops.ConvexFunctionLeastSquares(operator, rhs)
    g = convex_ops.ConvexFunctionL1Norm(scale=lambda_)

    result = optimizer_ops.admm_minimize(f, g)
    print(result)



if __name__ == '__main__':
  tf.test.main()
