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
"""Math utilities."""


def fibonacci_sequence(n):
  """Compute the Fibonacci sequence.

  Args:
    n: An `int`. The number of terms to compute. Must be >= 2.

  Returns:
    A list of `int`s.
  """
  return generalized_fibonacci_sequence(n, 1)


def generalized_fibonacci_sequence(n, p):
  """Compute the generalized Fibonacci sequence.

  Args:
    n: An `int`. The number of terms to compute. Must be >= 2.
    p: An `int`. The number of the generalized sequence. Must be >= 1. If `p` is
      1, the sequence is the standard Fibonacci sequence.

  Returns:
    A list of `int`s.
  """
  a = [1, p]
  for _ in range(n - 2):
    a.append(a[-2] + a[-1])
  return a


def fibonacci_sequence_up_to(n):
  """Compute the Fibonacci sequence up to a given number.

  Args:
    n: An `int`. Return all members of the Fibonacci sequence not greater than
      this number.

  Returns:
    A list of `int`s.
  """
  return generalized_fibonacci_sequence_up_to(n, 1)


def generalized_fibonacci_sequence_up_to(n, p):
  """Compute the generalized Fibonacci sequence up to a given number.

  Args:
    n: An `int`. Return all members of the generalized Fibonacci sequence not
      greater than this number.
    p: An `int`. The number of the generalized sequence. Must be >= 1. If `p` is
      1, the sequence is the standard Fibonacci sequence.

  Returns:
    A list of `int`s.
  """
  a = [1, p]
  while True:
    latest = a[-2] + a[-1]
    if latest > n:
      break
    a.append(latest)
  return a


def is_fibonacci_number(n):
  """Determine whether a number is a Fibonacci number.

  Args:
    n: An `int`.

  Returns:
    A `boolean`.
  """
  return is_generalized_fibonacci_number(n, 1)


def is_generalized_fibonacci_number(n, p):
  """Determine whether a number is a generalized Fibonacci number.

  Args:
    n: An `int`.
    p: An `int`. The number of the generalized sequence. Must be >= 1. If `p` is
      1, the sequence is the standard Fibonacci sequence.

  Returns:
    A `boolean`.
  """
  return n in generalized_fibonacci_sequence_up_to(n, p)
