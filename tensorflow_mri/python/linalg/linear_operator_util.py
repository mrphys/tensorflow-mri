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
"""Utilities for linear operators."""

import string


def patch_operator(operator_class):
  """Patches a built-in linear operator.

  Args:
    operator_class: A `LinearOperator` class.

  Returns:
    A patched `LinearOperator` class.
  """
  class_name = operator_class.__name__
  docstring = operator_class.__doc__

  patch_notice = string.Template("""
  ```{note}
  This operator is a drop-in replacement for
  `tf.linalg.${class_name}` but has been patched by TensorFlow MRI
  to support additional functionality.
  ```
  """).substitute(class_name=class_name)

  doclines = docstring.split('\n')
  doclines[2:2] = patch_notice.split('\n')
  docstring = '\n'.join(doclines)

  operator_class.__doc__ = docstring
  return operator_class
