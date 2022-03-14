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
API_DOCS_PATH = os.path.join(DOCS_PATH, 'api_docs')

sys.path.insert(0, ROOT_PATH)

from tensorflow_mri.python.util import api_util

# Create API docs directory.
os.makedirs(os.path.join(API_DOCS_PATH, 'tfmri'), exist_ok=True)

INDEX_TEMPLATE = string.Template(
"""TensorFlow MRI |release|
========================

.. image:: ../assets/view_on_github.svg
   :target: https://github.com/mrphys/tensorflow-mri
   :scale: 70%
   :alt: View on GitHub

|

.. include:: ../../README.rst
   :start-after: start-intro
   :end-before: end-intro

Installation
------------

.. include:: ../../README.rst
   :start-after: start-install
   :end-before: end-install

FAQ
---

.. include:: ../../README.rst
   :start-after: start-faq
   :end-before: end-faq

.. toctree::
   :maxdepth: 3
   :caption: API Documentation
   :hidden:

   ops
   ${namespaces}

.. meta::
   :google-site-verification: 8PySedj6KJ0kc5qC1CbO6_9blFB9Nho3SgXvbRzyVOU
""")

MODULE_DOC_TEMPLATE = string.Template(
"""tfmri.${module}
======${underline}

.. automodule:: tensorflow_mri.${module}

Classes
-------

.. autosummary::
    :toctree: ${module}
    :template: ${module}/class.rst
    :nosignatures:

    ${classes}

Functions
---------

.. autosummary::
    :toctree: ${module}
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

# Write namespace templates.
for name, module in modules.items():
  classes = '\n    '.join(sorted(module.classes))
  functions = '\n    '.join(sorted(module.functions))

  filename = os.path.join(API_DOCS_PATH, f'tfmri/{name}.rst')
  with open(filename, 'w') as f:
    f.write(MODULE_DOC_TEMPLATE.substitute(
        module=name,
        underline='=' * len(name),
        classes=classes,
        functions=functions))

# Write index.rst.
with open(os.path.join(DOCS_PATH, 'index.rst'), 'w') as f:
  namespaces = api_util.get_namespaces()
  namespaces = ['api_docs/tfmri/' + namespace for namespace in namespaces]
  f.write(INDEX_TEMPLATE.substitute(
      namespaces='\n   '.join(sorted(namespaces))))
