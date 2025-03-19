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
"""Deprecation utilities."""

from tensorflow.python.util import deprecation

# The following dictionary contains the removal date for deprecations
# at a given release.
REMOVAL_DATE = {
    '0.19.0': '2022-09-01',
    '0.20.0': '2022-10-01'
}

deprecated_alias = deprecation.deprecated_alias
deprecated = deprecation.deprecated
deprecated_endpoints = deprecation.deprecated_endpoints
deprecated_args = deprecation.deprecated_args
deprecated_arg_values = deprecation.deprecated_arg_values
deprecated_argument_lookup = deprecation.deprecated_argument_lookup
