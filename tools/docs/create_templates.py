# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
import sys

DOCS_PATH = os.path.dirname(os.path.realpath(__file__))
ROOT_PATH = os.path.join(DOCS_PATH, '..', '..')

sys.path.insert(0, ROOT_PATH)

from tensorflow_mri.python.util import api_util


CLASS_TEMPLATE = string.Template(
"""# ${module}.{{ objname }}

```{currentmodule} {{ module }}
```

```{auto{{ objtype }}} {{ objname }}
---
members:
show-inheritance:
---
```
""")

FUNCTION_TEMPLATE = string.Template(
"""# ${module}.{{ objname }}

```{currentmodule} {{ module }}
```

```{auto{{ objtype }}} {{ objname }}
```
""")

NAMESPACES = api_util.get_submodule_names()

TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), '_templates')

for namespace in NAMESPACES:
  # Create directory for this namespace.
  os.makedirs(os.path.join(TEMPLATE_PATH, namespace), exist_ok=True)

  # Special treatment for namespace `ops`, which maps to the `tfmri` parent
  # module.
  if namespace == 'ops':
    module = 'tfmri'
  else:
    module = f'tfmri.{namespace}'

  # Substitute the templates for this module.
  class_template = CLASS_TEMPLATE.substitute(module=module)
  function_template = FUNCTION_TEMPLATE.substitute(module=module)

  # Write template files.
  with open(os.path.join(TEMPLATE_PATH, namespace, 'class.md'), 'w') as f:
    f.write(class_template)
  with open(os.path.join(TEMPLATE_PATH, namespace, 'function.md'), 'w') as f:
    f.write(function_template)
