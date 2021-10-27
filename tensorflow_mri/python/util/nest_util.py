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
"""Utilities for nested structures."""

import tensorflow as tf


def unstack_nested_tensors(structure):
  """Make list of unstacked nested tensors.

  Args:
    structure: Nested structure of tensors whose first dimension is to be
      unstacked.

  Returns:
    A list of the unstacked nested tensors.
  """
  flat_sequence = tf.nest.flatten(structure)
  unstacked_flat_sequence = [tf.unstack(tensor) for tensor in flat_sequence]

  return [
      tf.nest.pack_sequence_as(structure, sequence)
      for sequence in zip(*unstacked_flat_sequence)
  ]
