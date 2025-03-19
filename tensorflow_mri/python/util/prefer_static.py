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
"""Operators that use static values if possible."""
# pylint: disable=redefined-builtin,wildcard-import,unused-wildcard-import

import tensorflow as tf
from tensorflow_probability.python.internal.prefer_static import *

from tensorflow_mri.python.ops import convex_ops


def batch_shape(operator):
  """Returns the batch shape of an operator.

  Returns the static batch shape of the operator if fully known, otherwise
  returns the dynamic batch shape.

  Args:
    operator: A `tf.linalg.LinearOperator` or a `tfmri.convex.ConvexFunction`.

  Returns:
    A `tf.TensorShape` or a 1D integer `tf.Tensor` representing the batch shape
    of the operator.

  Raises:
    ValueError: If `operator` is not a `tf.linalg.LinearOperator` or a
      `tfmri.convex.ConvexFunction`.
  """
  if not isinstance(operator, (tf.linalg.LinearOperator,
                               convex_ops.ConvexFunction)):
    raise ValueError(
        f"Input must be a `tf.linalg.LinearOperator` or a "
        f"`tfmri.convex.ConvexFunction`, but got: {type(operator)}")

  if operator.batch_shape.is_fully_defined():
    return operator.batch_shape

  return operator.batch_shape_tensor()


def domain_dimension(operator):
  """Retrieves the domain dimension of an operator.

  Args:
    operator: A `tf.linalg.LinearOperator` or a `tfmri.convex.ConvexFunction`.

  Returns:
    An int or scalar integer `tf.Tensor` representing the range dimension of the
    operator.

  Raises:
    ValueError: If `operator` is not a `tf.linalg.LinearOperator` or a
    `tfmri.convex.ConvexFunction`.
  """
  if not isinstance(operator, (tf.linalg.LinearOperator,
                               convex_ops.ConvexFunction)):
    raise ValueError(f"Input must be a `tf.linalg.LinearOperator` or a "
                     f"`tfmri.convex.ConvexFunction`, "
                     f"but got: {type(operator)}")

  dimension = operator.domain_dimension
  if isinstance(dimension, tf.compat.v1.Dimension):
    dimension = dimension.value
  if dimension is not None:
    return dimension

  return operator.domain_dimension_tensor()


def range_dimension(operator):
  """Retrieves the range dimension of an operator.

  Args:
    operator: A `tf.linalg.LinearOperator`.

  Returns:
    An int or scalar integer `tf.Tensor` representing the range dimension of the
    operator.

  Raises:
    ValueError: If `operator` is not a `tf.linalg.LinearOperator`.
  """
  if not isinstance(operator, tf.linalg.LinearOperator):
    raise ValueError(f"Input must be a `tf.linalg.LinearOperator`, "
                     f"but got: {type(operator)}")

  dimension = operator.range_dimension
  if isinstance(dimension, tf.compat.v1.Dimension):
    dimension = dimension.value
  if dimension is not None:
    return dimension

  return operator.range_dimension_tensor()
