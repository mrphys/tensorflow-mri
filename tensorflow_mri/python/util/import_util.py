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
"""Import utilities."""

import importlib
import sys


def lazy_import(name):
  """Imports a module lazily.

  Args:
    name: Name of the module to import.

  Returns:
    The module object.
  """
  # https://docs.python.org/3/library/importlib.html#implementing-lazy-imports
  spec = importlib.util.find_spec(name)
  loader = importlib.util.LazyLoader(spec.loader)
  spec.loader = loader
  module = importlib.util.module_from_spec(spec)
  sys.modules[name] = module
  loader.exec_module(module)
  return module
