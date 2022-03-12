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
"""Autodoc template generator."""

import os
import string


CLASS_TEMPLATE = string.Template(
"""${module}.{{ objname | escape | underline }}${underline}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
    :members:
    :show-inheritance:
""")

FUNCTION_TEMPLATE = string.Template(
"""${module}.{{ objname | escape | underline }}${underline}

.. currentmodule:: {{ module }}

.. auto{{ objtype }}:: {{ objname }}
""")

MODULES = {
    'callbacks',
    'io',
    'layers',
    'linalg',
    'losses',
    'metrics',
    'plot',
    'ops',
    'recon',
    'summary'
}

TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '_templates')

for module in MODULES:
  # Create directory for this module.
  os.makedirs(os.path.join(TEMPLATE_PATH, module), exist_ok=True)

  # Special treatment for module `ops`, which maps to the `tfmri` parent module.
  module_path = module
  if module == 'ops':
    module = 'tfmri'
  else:
    module = f'tfmri.{module}'

  # Substitute the templates for this module.
  class_template = CLASS_TEMPLATE.substitute(
      module=module, underline='=' * (len(module) + 1))
  function_template = FUNCTION_TEMPLATE.substitute(
      module=module, underline='=' * (len(module) + 1))

  # Write template files.
  with open(os.path.join(TEMPLATE_PATH, module_path, 'class.rst'), 'w') as f:
    f.write(class_template)
  with open(os.path.join(TEMPLATE_PATH, module_path, 'function.rst'), 'w') as f:
    f.write(function_template)
