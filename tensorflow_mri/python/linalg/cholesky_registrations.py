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
"""Registrations for LinearOperator.cholesky."""

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.linalg import linear_operator_diag_nd
from tensorflow_mri.python.linalg import linear_operator_identity_nd


@linear_operator_algebra.RegisterCholesky(
    linear_operator_identity_nd.LinearOperatorIdentityND)
def _cholesky_identity_nd(linop):
  return linear_operator_identity_nd.LinearOperatorIdentityND(
      domain_shape=linop.domain_shape_tensor(),
      batch_shape=linop.batch_shape,
      dtype=linop.dtype,
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_identity_nd.LinearOperatorScaledIdentityND)
def _cholesky_scaled_identity_nd(linop):
  return linear_operator_identity_nd.LinearOperatorScaledIdentityND(
      domain_shape=linop.domain_shape_tensor(),
      multiplier=tf.math.sqrt(linop.multiplier),
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)


@linear_operator_algebra.RegisterCholesky(
    linear_operator_diag_nd.LinearOperatorDiagND)
def _cholesky_diag_nd(linop):
  return linear_operator_diag_nd.LinearOperatorDiagND(
      tf.math.sqrt(linop.diag),
      batch_dims=linop.batch_shape.rank,
      is_non_singular=True,
      is_self_adjoint=True,
      is_positive_definite=True,
      is_square=True)
