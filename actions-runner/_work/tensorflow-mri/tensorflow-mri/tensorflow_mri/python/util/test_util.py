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

import unittest

from absl.testing import parameterized
from keras.testing_infra import test_combinations
from keras.testing_infra import test_utils
import numpy as np
import tensorflow as tf


layer_test = test_utils.layer_test
run_all_execution_modes = test_combinations.run_all_keras_modes


class TestCase(tf.test.TestCase, parameterized.TestCase):
  """Class to provide TensorFlow MRI specific test features."""
  # pylint: disable=invalid-name

  def assertAllTrue(self, a):
    """Assert that all entries in a boolean `Tensor` are True.

    Args:
      a: A `Tensor`.
    """
    a_ = self.get_nd_array(a)
    all_true = np.ones_like(a_, dtype=np.bool)
    self.assertAllEqual(all_true, a_)

  def assertAllFalse(self, a):
    """Assert that all entries in a boolean `Tensor` are False.

    Args:
      a: A `Tensor`.
    """
    a_ = self.get_nd_array(a)
    all_false = np.zeros_like(a_, dtype=np.bool)
    self.assertAllEqual(all_false, a_)

  def assertAllFinite(self, a):
    """Assert that all entries in a `Tensor` are finite.

    Args:
      a: A `Tensor`.
    """
    is_finite = np.isfinite(self.get_nd_array(a))
    all_true = np.ones_like(is_finite, dtype=np.bool)
    self.assertAllEqual(all_true, is_finite)

  def assertAllPositiveInf(self, a):
    """Assert that all entries in a `Tensor` are equal to positive infinity.

    Args:
      a: A `Tensor`.
    """
    is_positive_inf = np.isposinf(self.get_nd_array(a))
    all_true = np.ones_like(is_positive_inf, dtype=np.bool)
    self.assertAllEqual(all_true, is_positive_inf)

  def assertAllNegativeInf(self, a):
    """Assert that all entries in a `Tensor` are negative infinity.

    Args:
      a: A `Tensor`.
    """
    is_negative_inf = np.isneginf(self.get_nd_array(a))
    all_true = np.ones_like(is_negative_inf, dtype=np.bool)
    self.assertAllEqual(all_true, is_negative_inf)

  def get_nd_array(self, a):
    """Convert a `Tensor` to an `ndarray`.

    Args:
      a: A `Tensor`.

    Returns:
      An `ndarray`.
    """
    if tf.is_tensor(a):
      if tf.executing_eagerly():
        a = a.numpy()
      else:
        a = self.evaluate(a)
    if not isinstance(a, np.ndarray):
      return np.array(a)
    return a

  def skip_if_no_xla(self):
    """Skip this test if XLA is not available."""
    try:
      tf.function(lambda: tf.constant(0), jit_compile=True)()
    except (tf.errors.UnimplementedError, NotImplementedError) as e:
      if 'Could not find compiler' in str(e):
        self.skipTest('XLA not available')


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

      # Run in eager mode.
      f(self, *args, **kwargs)
      self.tearDown()

      # Run in graph mode.
      with tf.Graph().as_default():
        self.setUp()
        with self.test_session(use_gpu=use_gpu, config=config):
          f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def run_all_in_graph_and_eager_modes(cls):
  """Execute all test methods in the given class with and without eager."""
  base_decorator = run_in_graph_and_eager_modes

  for name in dir(cls):
    if (not name.startswith(unittest.TestLoader.testMethodPrefix) or
        name.startswith("testSkipEager") or
        name.startswith("test_skip_eager") or
        name == "test_session"):
      continue
    value = getattr(cls, name, None)
    if callable(value):
      setattr(cls, name, base_decorator(value))

  return cls
