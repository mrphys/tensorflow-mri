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
"""Utilities for file download and caching."""

import functools
import os

import keras

# The default cache dir where things will be stored.
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.tfmri')

# Same as https://www.tensorflow.org/api_docs/python/tf/keras/utils/get_file,
# but replaces default cache dir of "~/.keras" with "~/.tfmri".
get_file = functools.partial(keras.utils.get_file, cache_dir=CACHE_DIR)
