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
"""Utilities for documentation."""

import inspect

import tensorflow as tf


def get_nd_layer_signature(base):
  signature = inspect.signature(base.__init__)
  parameters = signature.parameters
  parameters = [v for k, v in parameters.items() if k not in ('self', 'rank')]
  signature = signature.replace(parameters=parameters)
  return signature


def custom_linkcode(func):
  """Returns a decorator to specify a custom linkcode function.

  Args:
    func: A callable with signature `func(obj) -> str` that returns the
      linkcode for the given object.

  Returns:
    A callable that can be used to decorate a class or function.
  """
  def decorator(obj):
    """Specifies a source code link for object."""
    obj.__linkcode__ = func(obj)
    if inspect.isclass(obj):
      for name, member in inspect.getmembers(obj):
        if inspect.isfunction(member) and name[0] != '_':
          member.__linkcode__ = func(member)
    return obj
  return decorator


def tf_linkcode(obj):
  """Indicates that an object's source code can be found in core TF."""
  return custom_linkcode(get_tf_linkcode)(obj)


def no_linkcode(obj):
  """Indicates that an object's source code is not available."""
  return custom_linkcode(lambda _: None)(obj)


def get_tf_linkcode(obj):
  """Returns the linkcode for the specified TensorFlow symbol."""
  # Get the file name of the current object.
  file = inspect.getsourcefile(obj)
  # If no file, we're done. This happens for C++ ops.
  if file is None:
    return None
  # When using TF's deprecation decorators, `getsourcefile` returns the
  # `deprecation.py` file where the decorators are defined instead of the
  # file where the object is defined. This should probably be fixed on the
  # decorators themselves. For now, we just don't add the link for deprecated
  # objects.
  if 'deprecation' in file:
    return None
  # Crop anything before `tensorflow_mri\python`. This path is system
  # dependent and we don't care about it.
  index = file.index('tensorflow/python')
  file = file[index:]

  # Base URL.
  url = 'https://github.com/tensorflow/tensorflow'
  # Add version blob.
  url += '/blob/v' + tf.__version__
  # Add file.
  url += '/' + file

  # Try to add line numbers. This will not work when the class is defined
  # dynamically. In that case we point to the file, but no line number.
  try:
    lines, start = inspect.getsourcelines(obj)
    stop = start + len(lines) - 1
  except OSError:
    # Could not get source lines.
    return url

  # Add line numbers.
  url += '#L' + str(start) + '-L' + str(stop)

  return url


def get_tf_url(api_name):
  """Returns the URL for the specified TensorFlow API name."""
  api_name = api_name.replace('.', '/')
  return f"https://www.tensorflow.org/api_docs/python/{api_name}"
