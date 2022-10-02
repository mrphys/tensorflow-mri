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
"""Registrations for LinearOperator.inverse."""

from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.linalg import linear_operator_coils
from tensorflow_mri.python.linalg import linear_operator_diag_nd
from tensorflow_mri.python.linalg import linear_operator_fft
from tensorflow_mri.python.linalg import linear_operator_mask
from tensorflow_mri.python.linalg import linear_operator_nufft


@linear_operator_algebra.RegisterInverse(
    linear_operator_coils.LinearOperatorCoils)
def _inverse_coils(linop):
  raise ValueError(
      f"{linop.name} is not invertible. If you wish to compute the "
      f"Moore-Penrose pseudo-inverse, use `linop.pseudo_inverse()` "
      f"instead.")


@linear_operator_algebra.RegisterInverse(
    linear_operator_diag_nd.LinearOperatorDiagND)
def _inverse_diag_nd(linop):
  return linear_operator_diag_nd.LinearOperatorDiagND(
      1. / linop.diag,
      batch_dims=linop.batch_shape.rank,
      is_non_singular=linop.is_non_singular,
      is_self_adjoint=linop.is_self_adjoint,
      is_positive_definite=linop.is_positive_definite,
      is_square=True)


@linear_operator_algebra.RegisterInverse(
    linear_operator_fft.LinearOperatorFFT)
def _inverse_fft(linop):
  return linop.adjoint()


@linear_operator_algebra.RegisterInverse(
    linear_operator_mask.LinearOperatorMask)
def _inverse_mask(linop):
  raise ValueError(
      f"{linop.name} is not invertible. If you wish to compute the "
      f"Moore-Penrose pseudo-inverse, use `linop.pseudo_inverse()` "
      f"instead.")


@linear_operator_algebra.RegisterInverse(
    linear_operator_nufft.LinearOperatorNUFFT)
def _inverse_nufft(linop):
  raise ValueError(
      f"{linop.name} is not invertible. If you wish to compute the "
      f"Moore-Penrose pseudo-inverse, use `linop.pseudo_inverse()` "
      f"instead.")
