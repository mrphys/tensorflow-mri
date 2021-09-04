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

from absl.testing import parameterized
import tensorflow as tf


class TestCase(tf.test.TestCase, parameterized.TestCase):
  """Class to provide TensorFlow MRI specific test features."""


def run_in_graph_and_eager_modes(func=None, config=None, use_gpu=True):
  """Execute the decorated test in both graph mode and eager mode.

  This function returns a decorator intended to be applied to test methods in
  a `test_case.TestCase` class. Doing so will cause the contents of the test
  method to be executed twice - once in graph mode, and once with eager
  execution enabled. This allows unittests to confirm the equivalence between
  eager and graph execution.

  .. note::
    This decorator can only be used when executing eagerly in the outer scope.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.
    config: An optional config_pb2.ConfigProto to use to configure the session
      when executing graphs.
    use_gpu: If `True`, attempt to run as many operations as possible on GPU.

  Returns:
    Returns a decorator that will run the decorated test method twice:
    once by constructing and executing a graph in a session and once with
    eager execution enabled.
  """
  def decorator(f): # pylint: disable=missing-param-doc
    """Decorator for a method."""

    def decorated(self, *args, **kwargs): # pylint: disable=missing-param-doc
      """Run the decorated test method."""
      if not tf.executing_eagerly():
        raise ValueError('Must be executing eagerly when using the '
                         'run_in_graph_and_eager_modes decorator.')

      # Run eager block
      f(self, *args, **kwargs)
      self.tearDown()

      # Run in graph mode block
      with tf.Graph().as_default():
        self.setUp()
        with self.test_session(use_gpu=use_gpu, config=config):
          f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator
