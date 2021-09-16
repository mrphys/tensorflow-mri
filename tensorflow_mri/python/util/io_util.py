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
import ismrmrd
import numpy as np


def read_hdf5(filename):
  """Basic reader for HDF5 files.

  This function recursively loads all datasets and returns them as a Python
  dictionary.

  Args:
    filename: A `string`. The path to the HDF5 file to read from.

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


def read_ismrmrd_kspace(filename):
  """Read an HDF5 file containing acquisitions into a *k*-space array.

  Can be used to read data from OCMR dataset files.

  Args:
    filename: A `string`. The path to the HDF5 file to read from.

  Returns:
    A *k*-space array.
  """
  f = ismrmrd.File(filename, mode='r')
  dataset = f['dataset']
  header = dataset.header

  encoding = dataset.header.encoding[0]
  encoding_limits = encoding.encodingLimits
  shape = (header.acquisitionSystemInformation.receiverChannels,
           encoding.encodedSpace.matrixSize.x,
           encoding_limits.kspace_encoding_step_1.maximum + 1,
           encoding_limits.kspace_encoding_step_2.maximum + 1,
           encoding_limits.average.maximum + 1,
           encoding_limits.slice.maximum + 1,
           encoding_limits.contrast.maximum + 1,
           encoding_limits.phase.maximum + 1,
           encoding_limits.repetition.maximum + 1,
           encoding_limits.set.maximum + 1
          #  encoding_limits.segment.maximum + 1 # Ignore segments.
  )

  kspace = np.zeros(shape, dtype=np.complex64)

  for acq in dataset.acquisitions:

    encoding_size = kspace.shape[1]
    num_samples = acq.number_of_samples
    first = encoding_size // 2 - acq.center_sample
    last = first + num_samples
    kspace[:, first:last,
           acq.idx.kspace_encode_step_1, acq.idx.kspace_encode_step_2,
           acq.idx.average, acq.idx.slice, acq.idx.contrast, acq.idx.phase,
           acq.idx.repetition, acq.idx.set] = acq.data[:]

  kspace = np.transpose(kspace)
  kspace = np.squeeze(kspace)
  kspace = np.transpose(kspace, [0, 3, 1, 2])
  return kspace
