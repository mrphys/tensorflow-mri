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
"""Creates the public API of TensorFlow MRI."""

import inspect
import pathlib
import string
import sys

SCRIPT_PATH = pathlib.Path(__file__).resolve()
BUILD_PATH = SCRIPT_PATH.parent
ROOT_PATH = BUILD_PATH.parent.parent
API_PATH = ROOT_PATH / 'tensorflow_mri/_api'
INIT_PATH = ROOT_PATH / 'tensorflow_mri/__init__.py'

sys.path.insert(0, str(ROOT_PATH))
from tensorflow_mri.python.util import api_util as api_util

INIT_TEMPLATE = string.Template(
'''# This file was automatically generated by ${script_path}.
# Do not edit.
"""TensorFlow MRI."""
import os as _os
import sys as _sys

from tensorflow_mri.__about__ import *

# TODO(jmontalt): Remove these imports on release 1.0.0.
from tensorflow_mri.python.ops.array_ops import *
from tensorflow_mri.python.ops.coil_ops import *
from tensorflow_mri.python.ops.convex_ops import *
from tensorflow_mri.python.ops.fft_ops import *
from tensorflow_mri.python.ops.geom_ops import *
from tensorflow_mri.python.ops.image_ops import *
from tensorflow_mri.python.ops.linalg_ops import *
from tensorflow_mri.python.ops.math_ops import *
from tensorflow_mri.python.ops.optimizer_ops import *
from tensorflow_mri.python.ops.recon_ops import *
from tensorflow_mri.python.ops.signal_ops import *
from tensorflow_mri.python.ops.traj_ops import *

from tensorflow_mri import python

# Import submodules.
${submodule_imports}

# Make sure directory containing top level submodules is in
# the __path__ so that "from tensorflow_mri.foo import bar" works.
# We're using callbacks, but there's nothing special about that.
_API_MODULE = _sys.modules[__name__].callbacks
_tfmri_api_dir = _os.path.dirname(_os.path.dirname(_API_MODULE.__file__))
_current_module = _sys.modules[__name__]

if not hasattr(_current_module, '__path__'):
  __path__ = [_tfmri_api_dir]
elif _tfmri_api_dir not in __path__:
  __path__.append(_tfmri_api_dir)
''')


SUBMODULE_TEMPLATE = string.Template(
'''# This file was automatically generated by ${script_path}.
# Do not edit.
"""${docstring}"""

${symbol_imports}
''')

SUBMODULE_IMPORT_TEMPLATE = "from tensorflow_mri._api import {submodule_name}"
SYMBOL_IMPORT_TEMPLATE = "from {module_name} import {symbol_name} as {symbol_alias}"

# Update the top-level __init__.py file.
submodule_imports = [
    SUBMODULE_IMPORT_TEMPLATE.format(submodule_name=submodule_name)
    for submodule_name in api_util.get_submodule_names()]
submodule_imports = '\n'.join(submodule_imports)

init_contents = INIT_TEMPLATE.substitute(
    script_path=SCRIPT_PATH.relative_to(ROOT_PATH),
    submodule_imports=submodule_imports)

with open(INIT_PATH, 'w') as f:
  f.write(init_contents)

# Now generate the individual submodule APIs.
for submodule_name in api_util.get_submodule_names():

  docstring = api_util.get_docstring_for_submodule(submodule_name)
  symbols = api_util.get_symbols_in_submodule(submodule_name)

  symbol_imports = []
  for api_name, symbol in symbols.items():
    symbol_alias = api_name.split('.')[-1]
    module_name = inspect.getmodule(symbol).__name__
    symbol_name = symbol.__name__
    symbol_imports.append(
        SYMBOL_IMPORT_TEMPLATE.format(module_name=module_name,
                                      symbol_name=symbol_name,
                                      symbol_alias=symbol_alias))
  symbol_imports = '\n'.join(symbol_imports)

  submodule_path = API_PATH / submodule_name
  submodule_path.mkdir(parents=True, exist_ok=True)
  submodule_path = submodule_path / '__init__.py'

  out = SUBMODULE_TEMPLATE.substitute(
      script_path=SCRIPT_PATH.relative_to(ROOT_PATH),
      docstring=docstring,
      symbol_imports=symbol_imports)

  with open(submodule_path, 'w') as f:
    f.write(out)
