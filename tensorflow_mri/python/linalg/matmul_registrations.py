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
"""Registrations for LinearOperator.matmul."""

from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.linalg import linear_operator_composition
from tensorflow_mri.python.linalg import linear_operator_diag_nd
from tensorflow_mri.python.linalg import linear_operator_identity_nd
from tensorflow_mri.python.linalg import linear_operator_nd
from tensorflow_mri.python.linalg import linear_operator_util


# IdentityND

@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity_nd.LinearOperatorIdentityND,
    linear_operator_nd.LinearOperatorND)
def _matmul_linear_operator_identity_nd_left(identity, linop):
  del identity
  return linop


@linear_operator_algebra.RegisterMatmul(
    linear_operator_nd.LinearOperatorND,
    linear_operator_identity_nd.LinearOperatorIdentityND)
def _matmul_linear_operator_identity_nd_right(linop, identity):
  del identity
  return linop


@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity_nd.LinearOperatorScaledIdentityND,
    linear_operator_identity_nd.LinearOperatorScaledIdentityND)
def _matmul_linear_operator_scaled_identity_nd(linop_a, linop_b):
  return linear_operator_identity_nd.LinearOperatorScaledIdentityND(
      domain_shape=linop_a.domain_shape_tensor(),
      multiplier=linop_a.multiplier * linop_b.multiplier,
      is_non_singular=linear_operator_composition.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=linear_operator_composition.combined_self_adjoint_hint(
          linop_a, linop_b, commuting=True),
      is_positive_definite=(
          linear_operator_composition.combined_positive_definite_hint(
              linop_a, linop_b, commuting=True)),
      is_square=True)


# DiagND

@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag_nd.LinearOperatorDiagND,
    linear_operator_diag_nd.LinearOperatorDiagND)
def _matmul_linear_operator_diag_nd(linop_a, linop_b):
  batch_dims_a, batch_dims_b = (
      linop_a.batch_shape.rank, linop_b.batch_shape.rank)
  diag_a, diag_b = linear_operator_util.prepare_inner_dims_for_broadcasting(
      linop_a.diag,
      linop_b.diag,
      batch_dims_a=batch_dims_a,
      batch_dims_b=batch_dims_b)
  return linear_operator_diag_nd.LinearOperatorDiagND(
      diag=diag_a * diag_b,
      batch_dims=max(batch_dims_a, batch_dims_b),
      is_non_singular=linear_operator_composition.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=linear_operator_composition.combined_self_adjoint_hint(
          linop_a, linop_b, commuting=True),
      is_positive_definite=(
          linear_operator_composition.combined_positive_definite_hint(
              linop_a, linop_b, commuting=True)),
      is_square=True)


@linear_operator_algebra.RegisterMatmul(
    linear_operator_diag_nd.LinearOperatorDiagND,
    linear_operator_identity_nd.LinearOperatorScaledIdentityND)
def _matmul_linear_operator_diag_scaled_identity_nd_right(
    linop_diag, linop_scaled_identity):
  batch_dims_a, batch_dims_b = (
      linop_diag.batch_shape.rank, linop_scaled_identity.batch_shape.rank)
  diag_a, diag_b = linear_operator_util.prepare_inner_dims_for_broadcasting(
      linop_diag.diag,
      linop_scaled_identity.multiplier,
      batch_dims_a=batch_dims_a,
      batch_dims_b=batch_dims_b)
  return linear_operator_diag_nd.LinearOperatorDiagND(
      diag=diag_a * diag_b,
      batch_dims=max(batch_dims_a, batch_dims_b),
      is_non_singular=linear_operator_composition.combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=linear_operator_composition.combined_self_adjoint_hint(
          linop_diag, linop_scaled_identity, commuting=True),
      is_positive_definite=(
          linear_operator_composition.combined_positive_definite_hint(
              linop_diag, linop_scaled_identity, commuting=True)),
      is_square=True)


@linear_operator_algebra.RegisterMatmul(
    linear_operator_identity_nd.LinearOperatorScaledIdentityND,
    linear_operator_diag_nd.LinearOperatorDiagND)
def _matmul_linear_operator_diag_scaled_identity_nd_left(
    linop_scaled_identity, linop_diag):
  batch_dims_a, batch_dims_b = (
      linop_scaled_identity.batch_shape.rank, linop_diag.batch_shape.rank)
  diag_a, diag_b = linear_operator_util.prepare_inner_dims_for_broadcasting(
      linop_scaled_identity.multiplier,
      linop_diag.diag,
      batch_dims_a=batch_dims_a,
      batch_dims_b=batch_dims_b)
  return linear_operator_diag_nd.LinearOperatorDiagND(
      diag=diag_a * diag_b,
      batch_dims=max(batch_dims_a, batch_dims_b),
      is_non_singular=linear_operator_composition.combined_non_singular_hint(
          linop_diag, linop_scaled_identity),
      is_self_adjoint=linear_operator_composition.combined_self_adjoint_hint(
          linop_diag, linop_scaled_identity, commuting=True),
      is_positive_definite=(
          linear_operator_composition.combined_positive_definite_hint(
              linop_diag, linop_scaled_identity, commuting=True)),
      is_square=True)
