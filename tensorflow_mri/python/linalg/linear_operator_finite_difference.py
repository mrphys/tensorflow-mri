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
"""Finite difference linear operator."""


import tensorflow as tf

from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import check_util
from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.util import tensor_util


@api_util.export("linalg.LinearOperatorFiniteDifference")
class LinearOperatorFiniteDifference(linear_operator.LinearOperator):  # pylint: disable=abstract-method
  """Linear operator representing a finite difference matrix.

  Args:
    domain_shape: A 1D `tf.Tensor` or a `list` of `int`. The domain shape of
      this linear operator.
    axis: An `int`. The axis along which the finite difference is taken.
      Defaults to -1.
    dtype: A `tf.dtypes.DType`. The data type for this operator. Defaults to
      `float32`.
    name: A `str`. A name for this operator.
  """
  def __init__(self,
               domain_shape,
               axis=-1,
               dtype=tf.dtypes.float32,
               name="LinearOperatorFiniteDifference"):

    parameters = dict(
        domain_shape=domain_shape,
        axis=axis,
        dtype=dtype,
        name=name
    )

    # Compute the static and dynamic shapes and save them for later use.
    self._domain_shape_static, self._domain_shape_dynamic = (
        tensor_util.static_and_dynamic_shapes_from_shape(domain_shape))

    # Validate axis and canonicalize to negative. This ensures the correct
    # axis is selected in the presence of batch dimensions.
    self.axis = check_util.validate_static_axes(
        axis, self._domain_shape_static.rank,
        min_length=1,
        max_length=1,
        canonicalize="negative",
        scalar_to_list=False)

    # Compute range shape statically. The range has one less element along
    # the difference axis than the domain.
    range_shape_static = self._domain_shape_static.as_list()
    if range_shape_static[self.axis] is not None:
      range_shape_static[self.axis] -= 1
    range_shape_static = tf.TensorShape(range_shape_static)
    self._range_shape_static = range_shape_static

    # Now compute dynamic range shape. First concatenate the leading axes with
    # the updated difference dimension. Then, iff the difference axis is not
    # the last one, concatenate the trailing axes.
    range_shape_dynamic = self._domain_shape_dynamic
    range_shape_dynamic = tf.concat([
        range_shape_dynamic[:self.axis],
        [range_shape_dynamic[self.axis] - 1]], 0)
    if self.axis != -1:
      range_shape_dynamic = tf.concat([
          range_shape_dynamic,
          range_shape_dynamic[self.axis + 1:]], 0)
    self._range_shape_dynamic = range_shape_dynamic

    super().__init__(dtype,
                     is_non_singular=None,
                     is_self_adjoint=None,
                     is_positive_definite=None,
                     is_square=None,
                     name=name,
                     parameters=parameters)

  def _transform(self, x, adjoint=False):

    if adjoint:
      paddings1 = [[0, 0]] * x.shape.rank
      paddings2 = [[0, 0]] * x.shape.rank
      paddings1[self.axis] = [1, 0]
      paddings2[self.axis] = [0, 1]
      x1 = tf.pad(x, paddings1) # pylint: disable=no-value-for-parameter
      x2 = tf.pad(x, paddings2) # pylint: disable=no-value-for-parameter
      x = x1 - x2
    else:
      slice1 = [slice(None)] * x.shape.rank
      slice2 = [slice(None)] * x.shape.rank
      slice1[self.axis] = slice(1, None)
      slice2[self.axis] = slice(None, -1)
      x1 = x[tuple(slice1)]
      x2 = x[tuple(slice2)]
      x = x1 - x2

    return x

  def _domain_shape(self):
    return self._domain_shape_static

  def _range_shape(self):
    return self._range_shape_static

  def _domain_shape_tensor(self):
    return self._domain_shape_dynamic

  def _range_shape_tensor(self):
    return self._range_shape_dynamic
