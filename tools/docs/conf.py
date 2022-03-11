# Copyright 2021 University College London. All Rights Reserved.
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
"""Configuration file for the Sphinx documentation builder.

This file only contains a selection of the most common options. For a full
list see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

from os import path
import inspect
import operator
import packaging.version
import re
import sys
import types

import conf_helper


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
sys.path.insert(0, path.abspath('../..'))


# -- Project information -----------------------------------------------------

ROOT = path.abspath(path.join(path.dirname(__file__), '../..'))

ABOUT = {}
with open(path.join(ROOT, "tensorflow_mri/__about__.py")) as f:
  exec(f.read(), ABOUT)
_version = packaging.version.Version(ABOUT['__version__'])

project = ABOUT['__title__']
copyright = ABOUT['__copyright__']
author = ABOUT['__author__']
release = _version.public
version = '.'.join(map(str, (_version.major, _version.minor)))


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
  'sphinx.ext.autodoc',
  'sphinx.ext.napoleon',
  'sphinx.ext.autosummary',
  'sphinx.ext.linkcode',
  'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# Do not add full qualification to objects' signatures.
add_module_names = False


# -- Options for HTML output -------------------------------------------------

html_title = 'TensorFlow MRI Documentation'

html_favicon = '../assets/tfmr_icon.svg'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../assets']


html_theme_options = {
  'sidebar_hide_name': True,
  'light_logo': 'tfmr_logo.svg',
  'dark_logo': 'tmfr_logo_dark.svg',
  'light_css_variables': {
    'color-brand-primary': '#128091',
    'color-brand-content': '#128091'
  },
  'dark_css_variables': {
    'color-brand-primary': '#18A8BE',
    'color-brand-content': '#18A8BE'
  }
}


autosummary_filename_map = conf_helper.AutosummaryFilenameMap()


import tensorflow_mri as tfmri


def linkcode_resolve(domain, info):
  """Find the GitHub URL where an object is defined.
  
  Args:
    domain: The language domain. This is always `py`.
    info: A `dict` with keys `module` and `fullname`.
  
  Returns:
    The GitHub URL to the object, or `None` if not relevant.
  """  
  # Obtain fully-qualified name of object.
  qualname = info['module'] + '.' + info['fullname']
  # Remove the `tensorflow_mri` bit.
  qualname = qualname.split('.', maxsplit=1)[-1]

  # Get the object.
  obj = operator.attrgetter(qualname)(tfmri)
  # We only add links to classes (type `type`) and functions
  # (type `types.FunctionType`).
  if not isinstance(obj, (type, types.FunctionType)):
    return None

  # Get the file name of the current object.
  file = inspect.getsourcefile(obj)
  # If no file, we're done. This happens for C++ ops.
  if file is None:
    return None
  # Crop anything before `tensorflow_mri\python`. This path is system
  # dependent and we don't care about it.
  index = file.index('tensorflow_mri/python')
  file = file[index:]

  # Get first and last line numbers.
  lines, start = inspect.getsourcelines(obj)
  stop = start + len(lines) - 1

  # Base URL.
  url = 'https://github.com/mrphys/tensorflow-mri'
  # Add version blob.
  url += '/blob/v' + release
  # Add file.
  url += '/' + file
  # Add line numbers.
  url += '#L' + str(start) + '-L' + str(stop)

  return url


INT_PATTERN = re.compile(r"``int``")
INT_REPL = r"`int <https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex>`_"
FLOAT_PATTERN = re.compile(r"``float``")
FLOAT_REPL = r"`float <https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex>`_"
COMPLEX_PATTERN = re.compile(r"``complex``")
COMPLEX_REPL = r"`complex <https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex>`_"
STR_PATTERN = re.compile(r"``str``")
STR_REPL = r"`str <https://docs.python.org/3/library/stdtypes.html#str>`_"
BOOL_PATTERN = re.compile(r"``bool``")
BOOL_REPL = r"`boolean <https://docs.python.org/3/library/stdtypes.html#boolean-values>`_"
NDARRAY_PATTERN = re.compile(r"``np.ndarray``")
NDARRAY_REPL = r"`np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_"
TENSOR_PATTERN = re.compile(r"``tf.Tensor``")
TENSOR_REPL = r"`tf.Tensor <https://www.tensorflow.org/api_docs/python/tf/Tensor>`_"
TENSORSHAPE_PATTERN = re.compile(r"``tf.TensorShape``")
TENSORSHAPE_REPL = r"`tf.TensorShape <https://www.tensorflow.org/api_docs/python/tf/TensorShape>`_"
DTYPE_PATTERN = re.compile(r"``tf.DType``")
DTYPE_REPL = r"`tf.dtypes.DType <https://www.tensorflow.org/api_docs/python/tf/dtypes/DType>`_"
LINK_PATTERN = re.compile(r"``(?P<link_text>[\w\.]+)``_")
LINK_REPL = r"`\g<link_text>`_"


def process_docstring(app, what, name, obj, options, lines):  # pylint: disable=missing-param-doc,unused-argument
  """Process autodoc docstrings."""
  # Replace markdown literal markers (`) by ReST literal markers (``).
  myst = '\n'.join(lines)
  text = myst.replace('`', '``')
  text = text.replace(':math:``', ':math:`')
  # Add links to some common types.
  text = INT_PATTERN.sub(INT_REPL, text)
  text = FLOAT_PATTERN.sub(FLOAT_REPL, text)
  text = COMPLEX_PATTERN.sub(COMPLEX_REPL, text)
  text = BOOL_PATTERN.sub(BOOL_REPL, text)
  text = STR_PATTERN.sub(STR_REPL, text)
  text = NDARRAY_PATTERN.sub(NDARRAY_REPL, text)
  text = TENSOR_PATTERN.sub(TENSOR_REPL, text)
  text = TENSORSHAPE_PATTERN.sub(TENSORSHAPE_REPL, text)
  text = DTYPE_PATTERN.sub(DTYPE_REPL, text)
  # Correct double quotes.
  text = LINK_PATTERN.sub(LINK_REPL, text)
  lines[:] = text.splitlines()


def setup(app):
  app.connect('autodoc-process-docstring', process_docstring)
