Release 0.16.0
==============

Breaking Changes
----------------

* ``tfmri.convex``:

  * Several of the inputs and outputs of ``admm_minimize`` have been renamed
    to improve clarity and to make the interface more consistent with the
    ``tfmri.optimize`` module.

Major Features and Improvements
-------------------------------

* ``tfmri.convex``:

  * ``admm_minimize`` now supports batches of inputs.

  * ``admm_minimize`` has a new argument ``cg_kwargs`` that allows passing
    additional arguments to the internal conjugate gradient solver.

  * ``admm_minimize`` has a new argument ``name`` that allows specifying
    the name of the operation.

  * ``ConvexFunctionQuadratic`` and ``ConvexFunctionLeastSquares`` have a new
    argument ``cg_kwargs`` that allows passing additional arguments to the
    internal conjugate gradient solver.

  * New properties ``shape`` and ``batch_shape`` for ``ConvexFunction`` and
    its subclasses. These allow retrieval of static shape information.

  * New methods ``ndim_tensor``, ``shape_tensor`` and ``batch_shape_tensor``
    for ``ConvexFunction`` and its subclasses. These allow retrieval of the
    dynamic shape information.

* ``tfmri.recon``:

  * ``lstsq`` has a new argument ``return_optimizer_state`` which allows the
    user to retrieve the internal optimizer state.

* ``tfmri.optimize``:

  * New function ``gradient_descent`` implementing the gradient descent method
    for minimization of differentiable functions.


Bug Fixes and Other Changes
---------------------------

* ``tfmri.convex``:

  * Fixed a bug in method ``prox`` of ``ConvexFunctionNorm``,
    ``ConvexFunctionL2Squared`` and ``ConvexFunctionQuadratic`` that caused
    errors when running in graph mode.

* ``tfmri.linalg``:

  * Fixed a bug in internal linear algebra framework that would cause errors
    when running in graph mode.
