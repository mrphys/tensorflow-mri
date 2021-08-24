# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from os import path
import inspect
import operator
import packaging.version
import sys
import types



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


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []


import tfmr


def linkcode_resolve(domain, info):
    """Find the GitHub URL where an object is defined.
    
    Args:
        domain: The language domain. This is always `py`.
        info: A `dict` with keys `module` and `fullname`.
    
    Returns:
        The GitHub URL to the object, or `None` if not relevant.
    """
    # Split `tfmr` part of module from submodules.
    module = info['module'].split('.', maxsplit=1)
    if len(module) == 2:
        # If length two, we have `tfmr` followed by submodule name.
        module, submodule = module
    else:
        # Otherwise, we have just `tfmr` without submodule.
        module = module[0]
        submodule = None
    # Hopefully we're not documenting anything outside the TFMR package!
    if module != 'tfmr':
        raise ValueError(f"Unexpected module: {module}")
    # If there is a submodule, add to the object name.
    objname = info['fullname']
    if submodule is not None:
        objname = submodule + '.' + objname

    # Get the object.
    obj = operator.attrgetter(objname)(tfmr)
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


def process_docstring(app, what, name, obj, options, lines):
    """Process autodoc docstrings."""
    # Replace markdown literal markers (`) by ReST literal markers (``).
    myst = '\n'.join(lines)
    text = myst.replace('`', '``')
    lines[:] = text.splitlines()


def setup(app):
    app.connect('autodoc-process-docstring', process_docstring)
