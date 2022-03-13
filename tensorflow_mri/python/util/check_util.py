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
"""Utilities for argument validation."""

import tensorflow as tf


def validate_enum(value, valid_values, name=None):
  """Validates that value is in a list of valid values.

  Args:
    value: The value to validate.
    valid_values: The list of valid values.
    name: The name of the argument being validated. This is only used to format
      error messages.

  Returns:
    A valid enum value.

  Raises:
    ValueError: If `value` is not in the list of valid values.
  """
  if value not in valid_values:
    raise ValueError(
      f"Argument `{name}` must be one of {valid_values}, but received value: "
      f"{value}")
  return value


def validate_type(value, type_, name=None):
  """Validates that value is of the specified type.

  Args:
    value: The value to validate.
    type_: The requested type.
    name: The name of the argument being validated. This is only used to format
      error messages.

  Returns:
    A valid value of type `type_`.

  Raises:
    TypeError: If `value` does not have type `type_`.
  """
  if not isinstance(value, type_):
    raise TypeError(
      f"Argument `{name}` must be of type {type_}, "
      f"but received type: {type(value)}")
  return value


def validate_list(value,
                  element_type=None,
                  length=None,
                  broadcast_scalars=True,
                  allow_tuples=True,
                  name=None):
  """Validates that value is a list with the specified characteristics.

  Args:
    value: The value to validate.
    element_type: A `type` or tuple of `type`s. The expected type for elements
      of the input list. Can be a tuple to allow more than one type. If `None`,
      the element type is not enforced.
    length: An `int`. The expected length of the list. If `None`, the length is
      not enforced.
    broadcast_scalars: A `boolean`. If `True`, scalar inputs are converted to
      lists of length `length`, if `length` is not `None`, or length 1
      otherwise. If `False`, an error is raised on scalar inputs.
    allow_tuples: A `boolean`. If `True`, inputs of type `tuple` are accepted
      and converted to `list`s. If `False`, an error is raised on tuple inputs.
    name: A `string`. The name of the argument being validated. This is only
      used to format error messages.

  Returns:
    A valid `list`.

  Raises:
    TypeError: When `value` does not meet the type requirements.
    ValueError: When `value` does not meet the length requirements.
  """
  # Handle tuples.
  if allow_tuples and isinstance(value, tuple):
    value = list(value)

  # Handle scalars.
  if broadcast_scalars:
    if ((element_type is not None and isinstance(value, element_type)) or
        (element_type is None and not isinstance(value, list))):
      value = [value] * (length if length is not None else 1)

  # We've handled tuples and scalars. If not a list by now, this is an error.
  if not isinstance(value, list):
    raise TypeError(
      f"Argument `{name}` must be a `list`, but received type: {type(value)}")

  # It's a list! Now check the length.
  if length is not None and not len(value) == length:
    raise ValueError(
      f"Argument `{name}` must be a `list` of length {length}, but received a "
      f"`list` of length {len(value)}")

  # It's a list with the correct length! Check element types.
  if element_type is not None:
    if not isinstance(element_type, (list, tuple)):
      element_types = (element_type,)
    else:
      element_types = element_type
    for element in value:
      if type(element) not in element_types:
        raise TypeError(
          f"Argument `{name}` must be a `list` of elements of type "
          f"`{element_type}`, but received type: `{type(element)}`")

  return value


def validate_axis(value,
                  rank=None,
                  min_length=None,
                  max_length=None,
                  canonicalize=False,
                  must_be_unique=True,
                  scalar_to_list=True):
  """Validates that value is a valid list of axes.

  Args:
    value: The value to check.
    rank: The rank of the tensor.
    min_length: The minimum number of axes.
    max_length: The maximum number of axes.
    canonicalize: Must be `"positive"`, `"negative"` or `None`.
    must_be_unique: If `True`, repeated axes are not allowed.
    scalar_to_list: If `True`, scalar inputs are converted to a list of length
      1.

  Returns:
    A valid `list` of axes.

  Raises:
    ValueError: If `value` is not valid.
  """
  scalar = isinstance(value, int)
  if scalar:
    value = [value]

  # Convert other iterables to list.
  value = list(value)

  if must_be_unique:
    if len(set(value)) != len(value):
      raise ValueError(
          f"Axes must be unique: {value}")

  if min_length is not None and len(value) < min_length:
    raise ValueError(
        f"Expected at least {min_length} axes, but got {len(value)}: {value}")
  if max_length is not None and len(value) > max_length:
    raise ValueError(
        f"Expected at most {max_length} axes, but got {len(value)}: {value}")

  if rank is not None:
    # These checks depend on the rank being known.
    for v in value:
      if v >= rank or v < -rank: # pylint: disable=invalid-unary-operand-type
        raise ValueError(
            f"Axis {v} is out of range for a tensor of rank {rank}")

    if canonicalize == "positive":
      value = [v + rank if v < 0 else v for v in value]
    elif canonicalize == "negative":
      value = [v - rank if v >= 0 else v for v in value]
    elif canonicalize is not None:
      raise ValueError(f"Invalid value of `canonicalize`: {canonicalize}")

  if scalar and not scalar_to_list:
    value = value[0]

  return value


def validate_tensor_dtype(tensor, dtypes, name):
  """Validates that a tensor has one of the specified types.

  Args:
    tensor: The tensor to validate.
    dtypes: A `tf.DType`, a list of `tf.DTypes`, or one of the following
      `strings`: `"bool"`, `"complex"`, `"floating"`, `"integer"`,
      `"numpy_compatible"`, `"quantized"` or `"unsigned"`.
    name: The name of the argument being validated. This is only used to format
      error messages.

  Returns:
    A tensor with asserted data type.

  Raises:
    TypeError: If `tensor` is not a `Tensor` or does not have a valid type.
  """
  is_attr = {
    'bool': tensor.dtype.is_bool,
    'complex': tensor.dtype.is_complex,
    'floating': tensor.dtype.is_floating,
    'integer': tensor.dtype.is_integer,
    'numpy_compatible': tensor.dtype.is_numpy_compatible,
    'quantized': tensor.dtype.is_quantized,
    'unsigned': tensor.dtype.is_unsigned
  }
  if isinstance(dtypes, str):
    if is_attr[dtypes]:
      return tensor
    raise TypeError(
      f"Argument `{name}` must have a {dtypes} data type, "
      f"but received data type: {tensor.dtype.name}")
  if isinstance(dtypes, tf.DType):
    dtypes = (dtypes,)
  if tensor.dtype not in dtypes:
    raise TypeError(
      f"Argument `{name}` must have data type {(dt.name for dt in dtypes)}, "
      f"but received data type: {tensor.dtype.name}")
  return tensor


def validate_rank(value, name=None, accept_none=True):
  """Validates that `value` is a valid rank.

  Args:
    value: The value to check.
    name: The name of the parameter. Only used to format error messages.
    accept_none: If `True`, `None` is accepted as a valid value.

  Returns:
    The value.

  Raises:
    TypeError: If `value` has an invalid type.
    ValueError: If `value` is not a valid rank.
  """
  if value is None:
    if accept_none:
      return None
    raise ValueError(f'Argument `{name}` must be specified.')
  if not isinstance(value, int):
    raise TypeError(
        f'Argument `{name}` must be an integer, but got {value}.')
  if value < 0:
    raise ValueError(
        f'Argument `{name}` must be non-negative, but got {value}.')
  return value


def verify_compatible_trajectory(kspace, traj):
  """Verifies that a trajectory is compatible with the given k-space.

  Args:
    kspace: A `Tensor`.
    traj: A `Tensor`.

  Returns:
    A tuple containing valid `kspace` and `traj` tensors.

  Raises:
    TypeError: If `kspace` and `traj` have incompatible dtypes.
    ValueError: If `kspace` and `traj` do not have the same number of samples
      or have incompatible batch shapes.
  """
  kspace = tf.convert_to_tensor(kspace, name='kspace')
  traj = tf.convert_to_tensor(traj, name='traj')

  # Check dtype.
  if traj.dtype != kspace.dtype.real_dtype:
    raise TypeError(
        f"kspace and trajectory have incompatible dtypes: "
        f"{kspace.dtype} and {traj.dtype}")

  # Check number of samples (static).
  if not kspace.shape[-1:].is_compatible_with(traj.shape[-2:-1]):
    raise ValueError(
        f"kspace and trajectory must have the same number of samples, but got "
        f"{kspace.shape[-1]} and {traj.shape[-2]}, respectively")
  # Check number of samples (dynamic).
  kspace_shape, traj_shape = tf.shape(kspace), tf.shape(traj)
  checks = [
      tf.debugging.assert_equal(
          kspace_shape[-1], traj_shape[-2],
          message="kspace and trajectory must have the same number of samples")
  ]
  with tf.control_dependencies(checks):
    kspace, traj = tf.identity_n([kspace, traj])

  # Check batch shapes (static).
  try:
    tf.broadcast_static_shape(kspace.shape[:-1], traj.shape[:-2])
  except ValueError as err:
    raise ValueError(
        f"kspace and trajectory have incompatible batch shapes, "
        f"got {kspace.shape[:-1]} and {traj.shape[:-2]}, respectively") from err
  # TODO(jmontalt): Check batch shapes (dynamic).

  return kspace, traj
