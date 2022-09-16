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
"""Utilities for N-D linear operators."""

import string

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_algebra


def with_mri_extensions_nd(op_cls):
  """Adds TensorFlow MRI extensions to an N-D linear operator.

  Args:
    op_cls: A `LinearOperator` class.

  Returns:
    A patched `LinearOperator` class.
  """
  attrs = {
      "solve_ls": solve_ls,
      "_solve_ls": _solve_ls,
      "solvevec_ls": solvevec_ls,
      "_solvevec_ls": _solvevec_ls
  }

  for name, attr in attrs.items():
    if not hasattr(op_cls, name):
      setattr(op_cls, name, attr)

  if is_tf_builtin(op_cls):
    op_cls = update_docstring(op_cls)

  return op_cls


def update_docstring(op_cls):
  """Adds a notice to the docstring."""
  tf_builtin_compatibility_notice = string.Template("""
  ```{rubric} Compatibility with core TensorFlow
  ```
  This operator is a drop-in replacement for
  `tf.linalg.${class_name}` but has been patched by TensorFlow MRI
  to support additional functionality including `solve_ls` and `solvevec_ls`.
  """).substitute(class_name=op_cls.__name__)

  docstring = op_cls.__doc__
  doclines = docstring.split('\n')
  doclines += tf_builtin_compatibility_notice.split('\n')
  docstring = '\n'.join(doclines)
  op_cls.__doc__ = docstring

  return op_cls


def is_tf_builtin(op_cls):
  """Returns `True` if `op_cls` is a built-in linear operator."""
  return hasattr(tf.linalg, op_cls.__name__)
