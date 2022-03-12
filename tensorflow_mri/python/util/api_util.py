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


_API_SYMBOLS = dict()

_API_ATTR = '_api_names'


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
    """
    setattr(symbol, _API_ATTR, names)
    for name in names:
      _API_SYMBOLS[name] = symbol
    return symbol
  return decorator
