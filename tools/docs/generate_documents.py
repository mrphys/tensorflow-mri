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
"""RST document generator."""

import dataclasses
import inspect
import os
import string
import sys
import typing

DOCS_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(DOCS_PATH, '..', '..')

sys.path.insert(0, ROOT_PATH)

from tensorflow_mri.python.util import api_util


MODULE_DOC_TEMPLATE = string.Template(
"""tfmri.${module}
======${underline}

.. automodule:: tensorflow_mri.${module}

Classes
-------

.. autosummary::
    :toctree: tfmri/${module}
    :template: ${module}/class.rst
    :nosignatures:

    ${classes}

Functions
---------

.. autosummary::
    :toctree: tfmri/${module}
    :template: ${module}/function.rst
    :nosignatures:

    ${functions}
""")


@dataclasses.dataclass
class Module:
  """A module."""
  classes: typing.List[str] = dataclasses.field(default_factory=list)
  functions: typing.List[str] = dataclasses.field(default_factory=list)

modules = {namespace: Module() for namespace in api_util.get_namespaces()}

for name, symbol in api_util.get_api_symbols().items():
  namespace, name = name.split('.', maxsplit=1)

  if inspect.isclass(symbol):
    modules[namespace].classes.append(name)
  elif inspect.isfunction(symbol):
    modules[namespace].functions.append(name)

for name, module in modules.items():
  classes = '\n    '.join(sorted(module.classes))
  functions = '\n    '.join(sorted(module.functions))

  filename = os.path.join(DOCS_PATH, f'{name}.rst')
  with open(filename, 'w') as f:
    f.write(MODULE_DOC_TEMPLATE.substitute(
        module=name,
        underline='=' * len(name),
        classes=classes,
        functions=functions))
