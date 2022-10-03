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
"""Registrations for LinearOperator.solve."""

from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.linalg import linear_operator_composition
from tensorflow_mri.python.linalg import linear_operator_diag_nd
from tensorflow_mri.python.linalg import linear_operator_identity_nd
from tensorflow_mri.python.linalg import linear_operator_nd


# IdentityND

@linear_operator_algebra.RegisterSolve(
    linear_operator_identity_nd.LinearOperatorIdentityND,
    linear_operator_nd.LinearOperatorND)
def _solve_linear_operator_identity_nd_left(identity, linop):
  del identity
  return linop


@linear_operator_algebra.RegisterSolve(
    linear_operator_nd.LinearOperatorND,
    linear_operator_identity_nd.LinearOperatorIdentityND)
def _solve_linear_operator_identity_nd_right(linop, identity):
  del identity
  return linop.inverse()


@linear_operator_algebra.RegisterSolve(
    linear_operator_identity_nd.LinearOperatorScaledIdentityND,
    linear_operator_identity_nd.LinearOperatorScaledIdentityND)
def _solve_linear_operator_scaled_identity_nd(linop_a, linop_b):
  return linear_operator_identity_nd.LinearOperatorScaledIdentityND(
      domain_shape=linop_a.domain_shape_tensor(),
      multiplier=linop_b.multiplier / linop_a.multiplier,
      is_non_singular=linear_operator_composition.combined_non_singular_hint(
          linop_a, linop_b),
      is_self_adjoint=linear_operator_composition.combined_self_adjoint_hint(
          linop_a, linop_b, commuting=True),
      is_positive_definite=(
          linear_operator_composition.combined_positive_definite_hint(
              linop_a, linop_b, commuting=True)),
      is_square=True)


# DiagND

# @linear_operator_algebra.RegisterSolve(
#     linear_operator_diag.LinearOperatorDiag,
#     linear_operator_diag.LinearOperatorDiag)
# def _solve_linear_operator_diag(linop_a, linop_b):
#   return linear_operator_diag.LinearOperatorDiag(
#       diag=linop_b.diag / linop_a.diag,
#       is_non_singular=registrations_util.combined_non_singular_hint(
#           linop_a, linop_b),
#       is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
#           linop_a, linop_b),
#       is_positive_definite=(
#           registrations_util.combined_commuting_positive_definite_hint(
#               linop_a, linop_b)),
#       is_square=True)


# @linear_operator_algebra.RegisterSolve(
#     linear_operator_diag.LinearOperatorDiag,
#     linear_operator_identity_nd.LinearOperatorScaledIdentity)
# def _solve_linear_operator_diag_scaled_identity_right(
#     linop_diag, linop_scaled_identity):
#   return linear_operator_diag.LinearOperatorDiag(
#       diag=linop_scaled_identity.multiplier / linop_diag.diag,
#       is_non_singular=registrations_util.combined_non_singular_hint(
#           linop_diag, linop_scaled_identity),
#       is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
#           linop_diag, linop_scaled_identity),
#       is_positive_definite=(
#           registrations_util.combined_commuting_positive_definite_hint(
#               linop_diag, linop_scaled_identity)),
#       is_square=True)


# @linear_operator_algebra.RegisterSolve(
#     linear_operator_identity_nd.LinearOperatorScaledIdentity,
#     linear_operator_diag.LinearOperatorDiag)
# def _solve_linear_operator_diag_scaled_identity_left(
#     linop_scaled_identity, linop_diag):
#   return linear_operator_diag.LinearOperatorDiag(
#       diag=linop_diag.diag / linop_scaled_identity.multiplier,
#       is_non_singular=registrations_util.combined_non_singular_hint(
#           linop_diag, linop_scaled_identity),
#       is_self_adjoint=registrations_util.combined_commuting_self_adjoint_hint(
#           linop_diag, linop_scaled_identity),
#       is_positive_definite=(
#           registrations_util.combined_commuting_positive_definite_hint(
#               linop_diag, linop_scaled_identity)),
#       is_square=True)
