# Copyright 2022 The TensorFlow MRI Authors. All Rights Reserved.
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
"""Base linear operator."""

import string
import warnings

import tensorflow as tf
from tensorflow.python.framework import type_spec
from tensorflow.python.ops.linalg.linear_operator import (
    _extract_attrs, _extract_type_spec_recursively)
from tensorflow.python.util import dispatch

from tensorflow_mri.python.linalg import linear_operator_algebra
from tensorflow_mri.python.linalg import linear_operator_util
from tensorflow_mri.python.util import api_util
from tensorflow_mri.python.util import doc_util


def make_linear_operator(cls):
  """Class decorator for subclasses of `LinearOperator`."""
  # Add extensions if decorating base class.
  if cls is tf.linalg.LinearOperator:
    extensions = {
        "lstsq": lstsq,
        "_lstsq": _lstsq,
        "lstsqvec": lstsqvec,
        "_lstsqvec": _lstsqvec,
        "_dense_lstsq": _dense_lstsq,
        "add": add,
        "__add__": __add__
    }

    for key, value in extensions.items():
      if hasattr(cls, key):
        raise ValueError(f"{cls.__name__} already has attribute: {key}")
      setattr(cls, key, value)

  # Make composite tensor. This also adds additional functionality to the class.
  cls = make_composite_tensor(cls)

  # Add notice to docstring.
  cls = update_docstring(cls)

  return cls


def make_composite_tensor(cls, module_name="tfmri.linalg"):
  """Class decorator to convert `LinearOperator`s to `CompositeTensor`s.

  Overrides the default `make_composite_tensor` to use the custom
  `LinearOperatorSpec`.
  """
  spec_name = "{}Spec".format(cls.__name__)
  spec_type = type(spec_name, (_LinearOperatorSpec,), {"value_type": cls})
  type_spec.register("{}.{}".format(module_name, spec_name))(spec_type)
  cls._type_spec = property(spec_type.from_operator)  # pylint: disable=protected-access
  return cls


def update_docstring(cls):
  """Updates docstring to describe new functionality."""
  tfmri_additional_functionality = string.Template("""
  ```{rubric} Additional functionality (TensorFlow MRI)
  ```

  This operator supports additional functionality not present in core TF
  operators.

  - `lstsq` and `lstsqvec` finds the least-squares solution to the linear
    system(s) defined by this operator.
  - `_lstsq` and `_lstsqvec` can be overridden to provide a custom
    implementation of `lstsq` and `lstsqvec`, respectively.
  - `_type_spec` has been patched to improve support in Keras models.

  ```{seealso}
  The TensorFlow MRI
  [linear algebra guide](https://mrphys.github.io/tensorflow-mri/guide/linalg/).
  ```
  """).substitute()

  tfmri_tf_compatibility = string.Template("""
  ```{rubric} Compatibility with core TensorFlow
  ```

  This operator is a drop-in replacement for `tf.linalg.${class_name}`.

  ```{tip}
  You can use `tfmri.linalg.${class_name}` and `tf.linalg.${class_name}`
  interchangeably, as the latter has been monkey-patched to be an alias of
  this operator.
  ```
  """).substitute(class_name=cls.__name__)

  docstring = cls.__doc__
  doclines = docstring.split('\n')
  doclines += tfmri_additional_functionality.split('\n')
  if is_tf_builtin(cls):
    doclines += tfmri_tf_compatibility.split('\n')
  docstring = '\n'.join(doclines)
  cls.__doc__ = docstring

  return cls


def is_tf_builtin(cls):
  """Returns `True` if `cls` is a built-in linear operator."""
  return hasattr(tf.linalg, cls.__name__)


# New attributes to be added to `LinearOperator` class.

def lstsq(self, rhs, adjoint=False, adjoint_arg=False, name="lstsq"):
  """Solve the (batch) linear system $A X = B$ in the least-squares sense.

  Given $A$ represented by this linear operator with shape `[..., M, N]`,
  computes the least-squares solution $X$ to the batch of linear systems
  $A X = B$. For systems without an exact solution, returns a "best fit"
  solution in the least squares sense. For systems with multiple solutions,
  returns the solution with the smallest Euclidean norm.

  This is equivalent to solving for the normal equations $A^H A X = A^H B$.

  Args:
    rhs: A `tf.Tensor` with same `dtype` as this operator and shape
      `[..., M, K]`. `rhs` is treated like a [batch] matrix meaning for
      every set of leading dimensions, the last two dimensions define a
      matrix.
    adjoint: A boolean. If `True`, solve the system involving the adjoint
      of this operator, $A^H X = B$. Default is `False`.
    adjoint_arg: A boolean. If `True`, solve $A X = B^H$ where $B^H$ is the
      Hermitian transpose (transposition and complex conjugation). Default
      is `False`.
    name: A name scope to use for ops added by this method.

  Returns:
    A `tf.Tensor` with shape `[..., N, K]` and same `dtype` as `rhs`.
  """
  if isinstance(rhs, LinearOperator):
    left_operator = self.adjoint() if adjoint else self
    right_operator = rhs.adjoint() if adjoint_arg else rhs

    if (right_operator.range_dimension is not None and
        left_operator.domain_dimension is not None and
        right_operator.range_dimension != left_operator.domain_dimension):
      raise ValueError(
          "Operators are incompatible. Expected `rhs` to have dimension"
          " {} but got {}.".format(
              left_operator.domain_dimension, right_operator.range_dimension))
    with self._name_scope(name):  # pylint: disable=not-callable
      return linear_operator_algebra.lstsq(left_operator, right_operator)

  with self._name_scope(name):  # pylint: disable=not-callable
    rhs = tf.convert_to_tensor(rhs, name="rhs")
    self._check_input_dtype(rhs)

    self_dim = -1 if adjoint else -2
    arg_dim = -1 if adjoint_arg else -2
    tf.compat.dimension_at_index(
        self.shape, self_dim).assert_is_compatible_with(
            rhs.shape[arg_dim])

    return self._lstsq(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

def _lstsq(self, rhs, adjoint=False, adjoint_arg=False):
  """Default implementation of `_lstsq`."""
  warnings.warn(
      "Using (possibly slow) default implementation of lstsq. "
      "Requires conversion to a dense matrix and O(N^3) operations.")
  return self._dense_lstsq(rhs, adjoint=adjoint, adjoint_arg=adjoint_arg)

def lstsqvec(self, rhs, adjoint=False, name="lstsqvec"):
  """Solve the linear system $A x = b$ in the least-squares sense.

  Given $A$ represented by this linear operator with shape `[..., M, N]`,
  computes the least-squares solution $x$ to the linear system $A x = b$.
  For systems without an exact solution, returns a "best fit" solution in
  the least squares sense. For systems with multiple solutions, returns the
  solution with the smallest Euclidean norm.

  This is equivalent to solving for the normal equations $A^H A x = A^H b$.

  Args:
    rhs: A `tf.Tensor` with same `dtype` as this operator and shape
      `[..., M]`. `rhs` is treated like a [batch] matrix meaning for
      every set of leading dimensions, the last two dimensions define a
      matrix.
    adjoint: A boolean. If `True`, solve the system involving the adjoint
      of this operator, $A^H x = b$. Default is `False`.
    name: A name scope to use for ops added by this method.

  Returns:
    A `tf.Tensor` with shape `[..., N]` and same `dtype` as `rhs`.
  """
  with self._name_scope(name):  # pylint: disable=not-callable
    rhs = tf.convert_to_tensor(rhs, name="rhs")
    self._check_input_dtype(rhs)
    self_dim = -1 if adjoint else -2
    tf.compat.dimension_at_index(
        self.shape, self_dim).assert_is_compatible_with(rhs.shape[-1])

    return self._lstsqvec(rhs, adjoint=adjoint)

def _lstsqvec(self, rhs, adjoint=False):
  """Default implementation of `_lstsqvec`."""
  rhs_mat = tf.expand_dims(rhs, axis=-1)
  solution_mat = self.lstsq(rhs_mat, adjoint=adjoint)
  return tf.squeeze(solution_mat, axis=-1)

def _dense_lstsq(self, rhs, adjoint=False, adjoint_arg=False):
  """Solve least squares by conversion to a dense matrix."""
  rhs = tf.linalg.adjoint(rhs) if adjoint_arg else rhs
  return linear_operator_util.matrix_solve_ls_with_broadcast(
      self.to_dense(), rhs, adjoint=adjoint)

def add(self, x, name="add"):
  """Add this operator to matrix `x`.

  Example:
    >>> operator = LinearOperatorIdentity(2)
    >>> x = tf.linalg.eye(2)
    >>> x = operator.add(x)
    >>> x.numpy()
    array([[2., 0.],
           [0., 2.]], dtype=float32)

  Args:
    x: A `LinearOperator` or `Tensor` with compatible shape and same `dtype` as
      `self`. See class docstring for definition of compatibility.
    name: A name for this `Op`.

  Returns:
    A `LinearOperator` or `Tensor` with same shape and same dtype as `self`.
  """
  if isinstance(x, LinearOperator):
    left_operator = self
    right_operator = x

    if (not left_operator.shape[-2:].is_compatible_with(
        right_operator.shape[-2:])):
      raise ValueError(
          f"Operators are incompatible. Expected `x` to have shape "
          f"{left_operator.shape[-2:]} but got {right_operator.shape[-2:]}.")
    with self._name_scope(name):
      return linear_operator_algebra.add(left_operator, right_operator)

  with self._name_scope(name):  # pylint: disable=not-callable
    return self.add_to_tensor(x)

def __add__(self, other):
  return self.add(other)



class _LinearOperatorSpec(type_spec.BatchableTypeSpec):  # pylint: disable=abstract-method
  """A tf.TypeSpec for `LinearOperator` objects.

  This is very similar to `tf.linalg.LinearOperatorSpec`, but it adds
  `shape` and `dtype` attributes which are required by Keras.

  These attributes are redundant, as they can always be computed from
  other parameters. However, the details of this computation vary between
  operators, so it's easier to just store it.
  """
  __slots__ = ("_param_specs",
               "_non_tensor_params",
               "_prefer_static_fields",
               "_shape",
               "_dtype")

  def __init__(self,
               param_specs,
               non_tensor_params,
               prefer_static_fields,
               shape=None,
               dtype=None):
    """Initializes a new `_LinearOperatorSpec`.

    Args:
      param_specs: Python `dict` of `tf.TypeSpec` instances that describe
        kwargs to the `LinearOperator`'s constructor that are `Tensor`-like or
        `CompositeTensor` subclasses.
      non_tensor_params: Python `dict` containing non-`Tensor` and non-
        `CompositeTensor` kwargs to the `LinearOperator`'s constructor.
      prefer_static_fields: Python `tuple` of strings corresponding to the names
        of `Tensor`-like args to the `LinearOperator`s constructor that may be
        stored as static values, if known. These are typically shapes, indices,
        or axis values.
      shape: A `tf.TensorShape`. The shape of the `LinearOperator`.
      dtype: A `tf.dtypes.DType`. The dtype of the `LinearOperator`.
    """
    self._param_specs = param_specs
    self._non_tensor_params = non_tensor_params
    self._prefer_static_fields = prefer_static_fields
    self._shape = shape
    self._dtype = dtype

  @classmethod
  def from_operator(cls, operator):
    """Builds a `_LinearOperatorSpec` from a `LinearOperator` instance.

    Args:
      operator: An instance of `LinearOperator`.

    Returns:
      linear_operator_spec: An instance of `_LinearOperatorSpec` to be used as
        the `TypeSpec` of `operator`.
    """
    validation_fields = ("is_non_singular", "is_self_adjoint",
                         "is_positive_definite", "is_square")
    kwargs = _extract_attrs(  # pylint: disable=protected-access
        operator,
        keys=set(operator._composite_tensor_fields + validation_fields))  # pylint: disable=protected-access

    non_tensor_params = {}
    param_specs = {}
    for k, v in list(kwargs.items()):
      type_spec_or_v = _extract_type_spec_recursively(v)  # pylint: disable=protected-access
      is_tensor = [isinstance(x, type_spec.TypeSpec)
                   for x in tf.nest.flatten(type_spec_or_v)]
      if all(is_tensor):
        param_specs[k] = type_spec_or_v
      elif not any(is_tensor):
        non_tensor_params[k] = v
      else:
        raise NotImplementedError(f"Field {k} contains a mix of `Tensor` and "
                                  f" non-`Tensor` values.")

    return cls(
        param_specs=param_specs,
        non_tensor_params=non_tensor_params,
        prefer_static_fields=operator._composite_tensor_prefer_static_fields,  # pylint: disable=protected-access
        shape=operator.shape,
        dtype=operator.dtype)

  def _to_components(self, obj):
    return _extract_attrs(obj, keys=list(self._param_specs))

  def _from_components(self, components):
    kwargs = dict(self._non_tensor_params, **components)
    return self.value_type(**kwargs)

  @property
  def _component_specs(self):
    return self._param_specs

  def _serialize(self):
    return (self._param_specs,
            self._non_tensor_params,
            self._prefer_static_fields,
            self._shape,
            self._dtype)

  def _copy(self, **overrides):
    kwargs = {
        "param_specs": self._param_specs,
        "non_tensor_params": self._non_tensor_params,
        "prefer_static_fields": self._prefer_static_fields,
        "shape": self._shape,
        "dtype": self._dtype
    }
    kwargs.update(overrides)
    return type(self)(**kwargs)

  def _batch(self, batch_size):
    """Returns a TypeSpec representing a batch of objects with this TypeSpec."""
    return self._copy(
        param_specs=tf.nest.map_structure(
            lambda spec: spec._batch(batch_size),  # pylint: disable=protected-access
            self._param_specs))

  def _unbatch(self):
    """Returns a TypeSpec representing a single element of this TypeSpec."""
    return self._copy(
        param_specs=tf.nest.map_structure(
            lambda spec: spec._unbatch(),  # pylint: disable=protected-access
            self._param_specs))

  @property
  def shape(self):
    """Returns a `tf.TensorShape` representing the static shape."""
    # This property is required to use linear operators with Keras.
    return self._shape

  @property
  def dtype(self):
    """Returns a `tf.dtypes.DType` representing the dtype."""
    return self._dtype

  def with_shape(self, shape):
    """Returns a new `tf.TypeSpec` with the given shape."""
    # This method is required to use linear operators with Keras.
    return self._copy(shape=shape)

  def _to_legacy_output_shapes(self):
    return self._shape

  def _to_legacy_output_types(self):
    return self._dtype


# Define new `LinearOperator` class.
LinearOperator = api_util.export("linalg.LinearOperator")(
    doc_util.no_linkcode(make_linear_operator(tf.linalg.LinearOperator)))


# Monkey-patch original operator so that core TF operator and TFMRI
# operator become aliases.
tf.linalg.LinearOperator = LinearOperator


@dispatch.dispatch_for_types(tf.math.add, LinearOperator)
def _add(x, y, name=None):
  if not isinstance(x, LinearOperator):
    return y.add(x, name=name)
  return x.add(y, name=name)
