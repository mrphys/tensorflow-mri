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
"""Utilities to export symbols to the API."""

import importlib
import sys


_API_SYMBOLS = dict()

_API_ATTR = '_api_names'

_NAMESPACES = [
    'plot',
    'recon'
]

_NAMESPACE_DOCSTRINGS = {
    'plot': "Plotting utilities.",
    'recon': "Image reconstruction."
}


def get_api_symbols():
  """Returns a live reference to the global API symbols dictionary."""
  return _API_SYMBOLS


def get_namespaces():
  """Returns a list of TFMRI namespaces."""
  return _NAMESPACES


def get_symbol_from_name(name):
  """Get API symbol from its name.

  Args:
    name: Name of the symbol.

  Returns:
    API symbol.
  """
  return _API_SYMBOLS.get(name)


def get_canonical_name_for_symbol(symbol):
  """Get canonical name for the API symbol.

  Args:
    symbol: API function or class.

  Returns:
    Canonical name for the API symbol.
  """
  if not hasattr(symbol, '__dict__'):
    return None
  if _API_ATTR not in symbol.__dict__:
    return None

  api_names = getattr(symbol, '_api_names')
  # Canonical name is the first name in the list.
  canonical_name = api_names[0]

  return canonical_name


def export(*names):
  """Returns a decorator to export a symbol to the API.

  Args:
    *names: List of API names under which the object should be exported.

  Returns:
    A decorator to export a symbol to the API.
  """
  def decorator(symbol):
    """Decorator to export a symbol to the API.

    Args:
      symbol: Symbol to decorate.

    Returns:
      The input symbol with the _api_names attribute set.

    Raises:
      ValueError: If the name is invalid or the symbol is already exported.
    """
    setattr(symbol, _API_ATTR, names)
    for name in names:
      # API name must have format "namespace.name".
      if name.count('.') != 1:
        raise ValueError(f"Invalid API name: {name}")
      # API namespace must be one of the supported ones.
      namespace, _ = name.split('.')
      if namespace not in _NAMESPACES:
        raise ValueError(f"Invalid API namespace: {namespace}")
      # API name must be unique.
      if name in _API_SYMBOLS:
        raise ValueError(
            f"Name {name} already used for exported symbol {symbol}")
      # Add symbol to the API symbols table.
      _API_SYMBOLS[name] = symbol
    return symbol
  return decorator


class APILoader(importlib.abc.Loader):  # pylint: disable=abstract-method
  """Loader for the public API."""
  def exec_module(self, module):
    """Executes the module.

    Args:
      module: module.
    """
    # Import public API.
    for name, symbol in _API_SYMBOLS.items():
      name = name.split('.')[-1]
      setattr(module, name, symbol)


def import_namespace(namespace):
  """Imports a namespace.

  Args:
    namespace: Namespace to import.

  Returns:
    The imported module.
  """
  spec = importlib.machinery.ModuleSpec(
      f'tensorflow_mri.{namespace}', APILoader())
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module
  spec.loader.exec_module(module)
  module.__doc__ = _NAMESPACE_DOCSTRINGS[namespace]
  return module
