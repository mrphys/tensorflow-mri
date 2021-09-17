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


def is_op_library_enabled():
  """Checks whether the op library is enabled.

  Returns `True` unless the environment variable `TFMR_DISABLE_OP_LIBRARY` has
  been set to a true value (as defined by `distutils.util.strtobool`).
  """
  str_value = os.getenv("TFMR_DISABLE_OP_LIBRARY", '0')
  bool_value = distutils.util.strtobool(str_value)
  return not bool_value
