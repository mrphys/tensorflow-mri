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
    ValueError: If `value` does not have type `type_`.
  """
  if not isinstance(value, type_):
    raise ValueError(
      f"Argument `{name}` must have type {type_}, "
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
    broadcast_scalars: A `bool`. If `True`, scalar inputs are converted to lists
      of length `length`, if `length` is not `None`, or length 1 otherwise. If
      `False`, an error is raised on scalar inputs.
    allow_tuples: A `bool`. If `True`, inputs of type `tuple` are accepted and
      converted to `list`s. If `False`, an error is raised on tuple inputs.
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
