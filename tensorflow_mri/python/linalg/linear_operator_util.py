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
"""Utilities for linear operators."""

import string

import tensorflow as tf

from tensorflow_mri.python.linalg import linear_operator
from tensorflow_mri.python.linalg import linear_operator_algebra


# @api_util.export("linalg.LinearOperator")
# class LinearOperator(LinearOperatorMixin, LinearOperatorBase):  # pylint: disable=abstract-method
#   r"""Base class defining a [batch of] linear operator[s].

#   Provides access to common matrix operations without the need to materialize
#   the matrix.

#   This operator is similar to `tf.linalg.LinearOperator`_, but has additional
#   methods to simplify operations on images, while maintaining compatibility
#   with the TensorFlow linear algebra framework.

#   Inputs and outputs to this linear operator or its subclasses may have
#   meaningful non-vectorized N-D shapes. Thus this class defines the additional
#   properties `domain_shape` and `range_shape` and the methods
#   `domain_shape_tensor` and `range_shape_tensor`. These enrich the information
#   provided by the built-in properties `shape`, `domain_dimension`,
#   `range_dimension` and methods `domain_dimension_tensor` and
#   `range_dimension_tensor`, which only have information about the vectorized 1D
#   shapes.

#   Subclasses of this operator must define the methods `_domain_shape` and
#   `_range_shape`, which return the static domain and range shapes of the
#   operator. Optionally, subclasses may also define the methods
#   `_domain_shape_tensor` and `_range_shape_tensor`, which return the dynamic
#   domain and range shapes of the operator. These two methods will only be called
#   if `_domain_shape` and `_range_shape` do not return fully defined static
#   shapes.

#   Subclasses must define the abstract method `_transform`, which
#   applies the operator (or its adjoint) to a [batch of] images. This internal
#   method is called by `transform`. In general, subclasses of this operator
#   should not define the methods `_matvec` or `_matmul`. These have default
#   implementations which call `_transform`.

#   Operators derived from this class may be used in any of the following ways:

#   1. Using method `transform`, which expects a full-shaped input and returns
#      a full-shaped output, i.e. a tensor with shape `[...] + shape`, where
#      `shape` is either the `domain_shape` or the `range_shape`. This method is
#      unique to operators derived from this class.
#   2. Using method `matvec`, which expects a vectorized input and returns a
#      vectorized output, i.e. a tensor with shape `[..., n]` where `n` is
#      either the `domain_dimension` or the `range_dimension`. This method is
#      part of the TensorFlow linear algebra framework.
#   3. Using method `matmul`, which expects matrix inputs and returns matrix
#      outputs. Note that a matrix is just a column vector in this context, i.e.
#      a tensor with shape `[..., n, 1]`, where `n` is either the
#      `domain_dimension` or the `range_dimension`. Matrices which are not column
#      vectors (i.e. whose last dimension is not 1) are not supported. This
#      method is part of the TensorFlow linear algebra framework.

#   Operators derived from this class may also be used with the functions
#   `tf.linalg.matvec`_ and `tf.linalg.matmul`_, which will call the
#   corresponding methods.

#   This class also provides the convenience functions `flatten_domain_shape` and
#   `flatten_range_shape` to flatten full-shaped inputs/outputs to their
#   vectorized form. Conversely, `expand_domain_dimension` and
#   `expand_range_dimension` may be used to expand vectorized inputs/outputs to
#   their full-shaped form.

#   **Preprocessing and post-processing**

#   It can sometimes be useful to modify a linear operator in order to maintain
#   certain mathematical properties, such as Hermitian symmetry or positive
#   definiteness (e.g., [1]). As a result of these modifications the linear
#   operator may no longer accurately represent the physical system under
#   consideration. This can be compensated through the use of a pre-processing
#   step and/or post-processing step. To this end linear operators expose a
#   `preprocess` method and a `postprocess` method. The user may define their
#   behavior by overriding the `_preprocess` and/or `_postprocess` methods. If
#   not overriden, the default behavior is to apply the identity. In the context
#   of optimization methods, these steps typically only need to be applied at the
#   beginning or at the end of the optimization.

#   **Subclassing**

#   Subclasses must always define `_transform`, which implements this operator's
#   functionality (and its adjoint). In general, subclasses should not define the
#   methods `_matvec` or `_matmul`. These have default implementations which call
#   `_transform`.

#   Subclasses must always define `_domain_shape`
#   and `_range_shape`, which return the static domain/range shapes of the
#   operator. If the subclassed operator needs to provide dynamic domain/range
#   shapes and the static shapes are not always fully-defined, it must also define
#   `_domain_shape_tensor` and `_range_shape_tensor`, which return the dynamic
#   domain/range shapes of the operator. In general, subclasses should not define
#   the methods `_shape` or `_shape_tensor`. These have default implementations.

#   If the subclassed operator has a non-scalar batch shape, it must also define
#   `_batch_shape` which returns the static batch shape. If the static batch shape
#   is not always fully-defined, the subclass must also define
#   `_batch_shape_tensor`, which returns the dynamic batch shape.

#   Args:
#     dtype: The `tf.dtypes.DType` of the matrix that this operator represents.
#     is_non_singular: Expect that this operator is non-singular.
#     is_self_adjoint: Expect that this operator is equal to its Hermitian
#       transpose. If `dtype` is real, this is equivalent to being symmetric.
#     is_positive_definite: Expect that this operator is positive definite,
#       meaning the quadratic form $x^H A x$ has positive real part for all
#       nonzero $x$. Note that we do not require the operator to be
#       self-adjoint to be positive-definite.
#     is_square: Expect that this operator acts like square [batch] matrices.
#     name: A name for this `LinearOperator`.

#   References:
#     1. https://onlinelibrary.wiley.com/doi/full/10.1002/mrm.1241

#   .. _tf.linalg.LinearOperator: https://www.tensorflow.org/api_docs/python/tf/linalg/LinearOperator
#   .. _tf.linalg.matvec: https://www.tensorflow.org/api_docs/python/tf/linalg/matvec
#   .. _tf.linalg.matmul: https://www.tensorflow.org/api_docs/python/tf/linalg/matmul
#   """
