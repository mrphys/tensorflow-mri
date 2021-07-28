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
"""Utilities for testing."""

import functools
import itertools


def parameterized_test(**params):
  """Decorates a test to run with multiple parameter combinations.

  All possible combinations (Cartesian product) of the parameters will be
  tested.

  Args:
    **params: The keyword parameters to be passed to the subtests. Each keyword
      argument should be a list of values to be tested.

  Returns:
    A decorator which decorates the test to be called with the specified
    parameter combinations.
  """
  param_lists = params

  def decorator(func):

    @functools.wraps(func)
    def run(self):

      # Create combinations of the parameters above.
      values = itertools.product(*param_lists.values())
      params = [dict(zip(param_lists.keys(), v)) for v in values]

      # Now call decorated function with each set of parameters.
      for p in params:
        with self.subTest(**p):
          func(self, **p)

    return run
  return decorator
