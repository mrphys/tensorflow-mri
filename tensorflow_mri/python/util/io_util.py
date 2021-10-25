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
"""Utilities for I/O."""

import glob
import math
import os
import shutil

import h5py
import numpy as np


def read_hdf5(filename):
  """Basic reader for HDF5 files.

  This function recursively loads all datasets and returns them as a Python
  dictionary.

  Args:
    filename: A `str`. The path to the HDF5 file to read from.

  Returns:
    A `dict` with one key-value pair for each dataset in the file.
  """
  contents = {}
  def visitor(name, obj):
    if isinstance(obj, h5py.Dataset):
      contents[name] = obj[...]
  with h5py.File(filename, 'r') as f:
    f.visititems(visitor)
  return contents


def write_hdf5(filename, contents):
  """Basic writer for HDF5 files.

  This function writes all datasets in a Python dictionary to an HDF5 file.

  Args:
    filename: A `str`.
    contents: A `dict`.
  """
  with h5py.File(filename, 'w') as f:
    for k, v in contents.items():
      f.create_dataset(k, data=v)


def read_distributed_hdf5(dirpath, prefixes=None, indices=None):
  """Read a distributed HDF5 set.

  Loads a set of HDF5 files from the specified path. If `prefixes` and/or
  `indices` are specified, loads a set of files with the structure
  `prefix_index.h5`.

  This function has several modes:

  * If `prefixes` and `indices` are `None`, reads all HDF5 files in the
    specified path.
  * If `prefixes` is not `None` and `indices` is `None`, reads all files with
    one of the specified prefixes.
  * If `prefixes` is not `None` and `indices` is not `None`, reads the files
    with the specified prefixes and indices. If `prefixes` is a list, `prefixes`
    and `indices` are interpreted in a pairwise fashion and must have the same
    length. If `prefixes` is a `str` containing a single prefix, reads the
    specified `indices` for the given prefix.

  Args:
    dirpath: Path to a directory.
    prefixes: A `str` or list of `str`. The prefixes of the files to read. 
    indices: An `int` or list of `int`. The indices of the files to read.

  Returns:
    A list of dictionaries. Each dictionary has the contents of a single file.
  """
  # Get list of files to load.
  filenames = get_distributed_hdf5_filenames(
      dirpath, prefixes=prefixes, indices=indices)

  # Read the files.
  data = [None] * len(filenames)
  for i, f in enumerate(filenames):
    data[i] = read_hdf5(f)

  return data


def write_distributed_hdf5(dirpath, data, prefixes, indices=None, overwrite=False):
  """Write a distributed HDF5 dataset.
  
  Writes data to a set of HDF5 files at the specified path.
  
  This function has several modes:

  * If `prefixes` is a `str` and `indices` is `None`, all files are saved with
    the specified prefix and new indices are generated automatically. Generated
    indices are in the range `[0, len(data)]` if `dirpath` does not exist or
    `overwrite` is `True`. Otherwise, indices are incremented as needed to
    append new files to the folder rather than overwrite them.
  * If `prefixes` and `indices` are lists, create new files with the specified
    indices and prefixes.

  Args:
    dirpath: Path to a directory.
    data: A list of dictionary. The contents of each dictionary will be written
      to a separate file.
    prefixes: A `str` or a list of `str`. The prefixes of the new files.
    indices: An `int` or a list of `int`. The indices of the new files.
    overwrite: A `bool`. If `True`, overwrites the contents of `dirpath`, if
      any.
  """
  # Accept tuples.
  if isinstance(prefixes, tuple):
    prefixes = list(prefixes)
  if isinstance(indices, tuple):
    indices = list(indices)

  # Number of files.
  num_files = len(data)

  if isinstance(prefixes, str):
    prefix = prefixes
    prefixes = [prefixes] * num_files
  else:
    prefix = None

  if len(prefixes) != len(data):
    raise ValueError(
        f"`data` and `prefixes` must have the same length, but got "
        f"{len(data)} and {len(prefixes)}, respectively.")

  if indices is None:
    if prefix is None:
      # If prefix is None, it means the user passed something other than a
      # string in the `prefixes` argument.
      raise ValueError(
          "A list of `prefixes` must be specified together with a list of "
          "`indices`.")
    # Indices not specified. Generate a list of indices.
    if not os.path.isdir(dirpath) or overwrite is True:
      # Directory does not exist or we are overwriting it, so we start from
      # index 0.
      first_index = 0
    else:
      # Directory exists and we are not overwriting it, so we need to create
      # new indices that are appended to previous ones.
      current_indices = get_distributed_hdf5_indices(dirpath, prefix)
      first_index = max(current_indices) + 1 if current_indices else 0
    indices = list(range(first_index, first_index + num_files))

  # Create filenames.
  filenames = _assemble_filenames(dirpath, prefixes, indices)

  # Create directory, overwriting if needed.
  if os.path.exists(dirpath) and overwrite is True:
    shutil.rmtree(dirpath)
  os.makedirs(dirpath, exist_ok=True)

  # Write files.
  for f, d in zip(filenames, data):
    write_hdf5(f, d)


def get_distributed_hdf5_filenames(dirpath, prefixes=None, indices=None):
  """Get the filenames in a distributed HDF5 set.

  Args:
    dirpath: Path to a directory.
    prefixes: A `str` or list of `str`. The prefixes of the files to read. 
    indices: An `int` or list of `int`. The indices of the files to read.

  Returns:
    A list of filenames.
  """
  # Verify that directory exists.
  if not os.path.isdir(dirpath):
    raise OSError(f"`dirpath` is not a directory or does not exist: {dirpath}")

  # Accept tuples.
  if isinstance(prefixes, tuple):
    prefixes = list(prefixes)
  if isinstance(indices, tuple):
    indices = list(indices)

  if prefixes is None and indices is None:
    filenames = glob.glob(os.path.join(dirpath, '*.h5'))
    filenames.sort()

  elif prefixes is not None and indices is None:
    if isinstance(prefixes, str):
      prefixes = [prefixes]
    filenames = []
    for p in prefixes:
      tmp = glob.glob(os.path.join(dirpath, p + '*.h5'))
      tmp.sort()
      filenames.extend(tmp)

  elif prefixes is not None and indices is not None:
    if isinstance(prefixes, str) and isinstance(indices, int):
      # Single file.
      prefixes = [prefixes]
      indices = [indices]
    if isinstance(prefixes, str) and isinstance(indices, list):
      # Single prefix, multiple indices.
      prefixes = [prefixes] * len(indices)

    filenames = _assemble_filenames(dirpath, prefixes, indices)

    for pfx, idx, fname in zip(prefixes, indices, filenames):
      if not os.path.isfile(fname):
        raise ValueError(f"No existing file with prefix {pfx} and index {idx}.")

  elif prefixes is None and indices is not None:
    raise ValueError("`indices` must be specified together with `prefixes`.")

  return filenames


def get_distributed_hdf5_prefixes_and_indices(dirpath):
  """Get the prefixes and indices of a distributed HDF5 dataset.
  
  Args:
    dirpath: Path to a directory.

  Returns:
    A tuple containing a list of prefixes and a list of indices.
  """
  # Verify that directory exists.
  if not os.path.isdir(dirpath):
    raise OSError(f"`dirpath` is not a directory or does not exist: {dirpath}")

  # Get list of filenames.
  filenames = glob.glob(os.path.join(dirpath, '*.h5'))
  names = [os.path.splitext(os.path.basename(name))[0] for name in filenames]

  # Extract prefixes and indices from filenames.
  prefixes = []
  indices = []
  for name in names:
    pfx, idx = name.rsplit(sep='_', maxsplit=1)
    idx = int(idx)
    prefixes.append(pfx)
    indices.append(idx)

  return prefixes, indices


def get_distributed_hdf5_indices(dirpath, prefix):
  """Get the indices for the given prefix of a distributed HDF5 dataset.

  Args:
    dirpath: Path to a directory.
    prefix: The prefix to get indices for.

  Returns:
    A list of indices.
  """
  prefixes, indices = get_distributed_hdf5_prefixes_and_indices(dirpath)
  return [idx for pfx, idx in zip(prefixes, indices) if pfx == prefix]


def _assemble_filenames(dirpath, prefixes, indices):
  """Assemble filenames from directory, prefixes and indices.
  
  Args:
    dirpath: Path to a directory.
    prefixes: A list of filename prefixes.
    indices: A list of file indices.

  Returns:
    A list of filenames.
  """
  if not isinstance(prefixes, list):
      raise ValueError(f"`prefixes` must be a `list` but got type: "
                       f"{type(prefixes).__name__}")

  if not isinstance(indices, list):
    raise ValueError(f"`indices` must be an `list` but got type: "
                      f"{type(indices).__name__}")

  if len(prefixes) != len(indices):
    raise ValueError("`prefixes` and `indices` must have the same length.")

  return [os.path.join(dirpath, pfx + '_' + _padnum(idx, 5) + '.h5') \
          for pfx, idx in zip(prefixes, indices)]


def _padnum(n, length):
  """Pad a number with zeros.

  Args:
    n: An `int`.
    length: The desired number of digits.

  Returns:
    A string with the zero-padded number.
  """
  return '0' * (length - _ndigits(n)) + str(n)


def _ndigits(n):
  """Get number of digits in input number.

  Args:
    n: An `int`.

  Returns:
    The number of digits in input number.
  """
  if not isinstance(n, int):
    raise ValueError(f"`n` must be an integer, but got: "
                     f"{n} with type {type(n).__name__}")
  if n == 0:
    return 1
  return int(math.log10(n)) + 1
