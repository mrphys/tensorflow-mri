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

import h5py


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
