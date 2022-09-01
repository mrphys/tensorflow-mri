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
sys.path.insert(0, path.abspath('extensions'))


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
  'myst_nb',
  'myst_autodoc',
  'myst_autosummary',
  'myst_napoleon',
  'sphinx.ext.linkcode',
  'sphinx_sitemap'
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'templates']

# Do not add full qualification to objects' signatures.
add_module_names = False

# For classes, list the class documentation but not `__init__`.
autoclass_content = 'class'

# -- Options for HTML output -------------------------------------------------

html_title = 'TensorFlow MRI Documentation'
html_logo = '../assets/tfmri_logo.svg'
html_favicon = '../assets/tfmri_icon.svg'

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_book_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['../assets']

# https://sphinx-book-theme.readthedocs.io/en/latest/tutorials/get-started.html
html_theme_options = {
    'repository_url': 'https://github.com/mrphys/tensorflow-mri',
    'use_repository_button': True,
    'logo_only': True,
    'launch_buttons': {
        'colab_url': "https://colab.research.google.com/"
    },
    'path_to_docs': 'tools/docs'
}

html_css_files = [
    'https://fonts.googleapis.com/css?family=Roboto|Roboto+Mono',
]

# Additional files to copy to output directory.
html_extra_path = ['robots.txt']

# For sitemap generation.
html_baseurl = 'https://mrphys.github.io/tensorflow-mri/'
sitemap_url_scheme = '{link}'

# For autosummary generation.
autosummary_generate = True
autosummary_filename_map = conf_helper.AutosummaryFilenameMap()

# -- Options for MyST ----------------------------------------------------------
# https://myst-nb.readthedocs.io/en/latest/authoring/jupyter-notebooks.html
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "substitution"
]

# https://myst-nb.readthedocs.io/en/latest/authoring/basics.html
source_suffix = [
    '.rst',
    '.md',
    '.ipynb'
]

# https://myst-parser.readthedocs.io/en/latest/syntax/optional.html#substitutions-with-jinja2
myst_substitutions = {
    'release': release
}

# Do not execute notebooks.
# https://myst-nb.readthedocs.io/en/latest/computation/execute.html
nb_execution_mode = "off"


import tensorflow_mri as tfmri


def linkcode_resolve(domain, info):
  """Find the GitHub URL where an object is defined.

  Args:
    domain: The language domain. This is always `py`.
    info: A `dict` with keys `module` and `fullname`.

  Returns:
    The GitHub URL to the object, or `None` if not relevant.
  """
  custom_ops = {'nufft', 'spiral_waveform'}
  if info['fullname'] in custom_ops:
    # Can't provide link to source for custom ops.
    return None

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
  # When using TF's deprecation decorators, `getsourcefile` returns the
  # `deprecation.py` file where the decorators are defined instead of the
  # file where the object is defined. This should probably be fixed on the
  # decorators themselves. For now, we just don't add the link for deprecated
  # objects.
  if 'deprecation' in file:
    return None
  # Crop anything before `tensorflow_mri\python`. This path is system
  # dependent and we don't care about it.
  index = file.index('tensorflow_mri/python')
  file = file[index:]

  # Base URL.
  url = 'https://github.com/mrphys/tensorflow-mri'
  # Add version blob.
  url += '/blob/v' + release
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


# -- Hyperlinks --------------------------------------------------------------
# Common types and constants in the API docs are enriched with hyperlinks to
# their corresponding docs.

# The following dictionary specifies type names and the corresponding links.
# The link is only added if the name has inline code format, e.g. ``foo``.
COMMON_TYPES_LINKS = {
    # Python standard types.
    'int': 'https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex',
    'float': 'https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex',
    'complex': 'https://docs.python.org/3/library/stdtypes.html#numeric-types-int-float-complex',
    'str': 'https://docs.python.org/3/library/stdtypes.html#text-sequence-type-str',
    'boolean': 'https://docs.python.org/3/library/stdtypes.html#boolean-values',
    'tuple': 'https://docs.python.org/3/library/stdtypes.html#tuples',
    'list': 'https://docs.python.org/3/library/stdtypes.html#lists',
    'dict': 'https://docs.python.org/3/library/stdtypes.html#mapping-types-dict',
    'namedtuple': 'https://docs.python.org/3/library/collections.html#namedtuple-factory-function-for-tuples-with-named-fields',
    'callable': 'https://docs.python.org/3/library/functions.html#callable',
    'dataclass': 'https://docs.python.org/3/library/dataclasses.html',
    # Python constants.
    'False': 'https://docs.python.org/3/library/constants.html#False',
    'True': 'https://docs.python.org/3/library/constants.html#True',
    'None': 'https://docs.python.org/3/library/constants.html#None',
    # NumPy types.
    'np.ndarray': 'https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html',
    'np.inf': 'https://numpy.org/doc/stable/reference/constants.html#numpy.inf',
    'np.nan': 'https://numpy.org/doc/stable/reference/constants.html#numpy.nan',
}


def process_docstring(app, what, name, obj, options, lines):  # pylint: disable=missing-param-doc,unused-argument
  """Processes autodoc docstrings."""
  # Regular expressions.
  blankline_re = re.compile(r"^\s*$")
  prompt_re = re.compile(r"^\s*>>>")
  tf_symbol_re = re.compile(r"`(?P<symbol>tf\.[a-zA-Z0-9_.]+)`")

  # Loop initialization. `insert_lines` keeps a list of lines to be inserted
  # as well as their positions.
  insert_lines = []
  in_prompt = False

  # Iterate line by line.
  for lineno, line in enumerate(lines):

    # Check if we're in a prompt block.
    if in_prompt:
      # Check if end of prompt block.
      if blankline_re.match(line):
        in_prompt = False
        insert_lines.append((lineno, "```"))
        continue

    # Check for >>> prompt, if found insert code block (unless already in
    # prompt).
    m = prompt_re.match(line)
    if m and not in_prompt:
      in_prompt = True
      # We need to insert a new line. It's not safe to modify the list we're
      # iterating over, so instead we store the line in `insert_lines` and we
      # insert it after the loop.
      insert_lines.append((lineno, "```python"))
      continue

    # Add links to TF symbols.
    m = tf_symbol_re.search(line)
    if m:
      symbol = m.group('symbol')
      link = f"https://www.tensorflow.org/api_docs/python/{symbol.replace('.', '/')}"
      lines[lineno] = line.replace(f"`{symbol}`", f"[`{symbol}`]({link})")

  # Now insert the lines (in reversed order so that line numbers stay valid).
  for lineno, line in reversed(insert_lines):
    lines.insert(lineno, line)


def get_doc_url(name):
  """Get doc URL for the given TFMRI name."""
  url = 'https://mrphys.github.io/tensorflow-mri/api_docs/'
  url += name.replace('.', '/')
  return url


def setup(app):
  app.connect('autodoc-process-docstring', process_docstring)
