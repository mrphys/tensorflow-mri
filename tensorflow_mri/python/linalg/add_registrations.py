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
"""Registrations for LinearOperator.add."""

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_addition
from tensorflow_mri.python.linalg import linear_operator_algebra


# By default, use a LinearOperatorAddition to delay the computation.
@linear_operator_algebra.RegisterAdd(
    linear_operator.LinearOperator, linear_operator.LinearOperator)
def _add_linear_operator(linop_a, linop_b):
  """Generic add of two `LinearOperator`s."""
  # Set all hints to `None`. `LinearOperatorAddition` will figure them out
  # automatically, if possible.
  return linear_operator_addition.LinearOperatorAddition(
      operators=[linop_a, linop_b],
      is_non_singular=None,
      is_self_adjoint=None,
      is_positive_definite=None,
      is_square=None
  )
