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
"""Keras initializers."""

from tensorflow_mri.python.initializers import initializers


TFMRI_INITIALIZERS = {
    'VarianceScaling': initializers.VarianceScaling,
    'GlorotNormal': initializers.GlorotNormal,
    'GlorotUniform': initializers.GlorotUniform,
    'HeNormal': initializers.HeNormal,
    'HeUniform': initializers.HeUniform,
    'LecunNormal': initializers.LecunNormal,
    'LecunUniform': initializers.LecunUniform,
    'variance_scaling': initializers.VarianceScaling,
    'glorot_normal': initializers.GlorotNormal,
    'glorot_uniform': initializers.GlorotUniform,
    'he_normal': initializers.HeNormal,
    'he_uniform': initializers.HeUniform,
    'lecun_normal': initializers.LecunNormal,
    'lecun_uniform': initializers.LecunUniform,
}
