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
"""Utilities for system configuration."""

import distutils.util
import os


_ENABLE_ASSISTANT = True


def is_op_library_enabled():
  """Checks whether the op library is enabled.

  Returns `True` unless the environment variable `TFMRI_DISABLE_OP_LIBRARY` has
  been set to a true value (as defined by `distutils.util.strtobool`).
  """
  str_value = os.getenv("TFMRI_DISABLE_OP_LIBRARY", '0')
  bool_value = distutils.util.strtobool(str_value)
  return not bool_value


def is_assistant_enabled():
  """Check whether the TensorFlow MRI assistant is enabled.

  See also `enable_assistant` and `disable_assistant`.

  Note that the assistant is enabled by default.
  """
  global _ENABLE_ASSISTANT
  return _ENABLE_ASSISTANT


def enable_assistant():
  """Enable the TensorFlow MRI assistant."""
  global _ENABLE_ASSISTANT
  _ENABLE_ASSISTANT = True


def disable_assistant():
  """Disable the TensorFlow MRI assistant."""
  global _ENABLE_ASSISTANT
  _ENABLE_ASSISTANT = False
