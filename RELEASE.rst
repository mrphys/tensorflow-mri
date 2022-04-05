Release 0.16.0
==============

Major Features and Improvements
-------------------------------

* ``tfmri.convex``:

  * ``admm_minimize`` has a new argument ``cg_kwargs`` that allows passing
    additional arguments to the internal conjugate gradient solver.

  * ``ConvexFunctionQuadratic`` and ``ConvexFunctionLeastSquares`` have a new
    argument ``cg_kwargs`` that allows passing additional arguments to the
    internal conjugate gradient solver.

* ``tfmri.recon``:

  * ``lstsq`` has a new argument ``return_optimizer_state`` which allows the
    user to retrieve the internal optimizer state.


Bug Fixes and Other Changes
---------------------------

* ``tfmri.convex``:

  * Fixed a bug in method ``prox`` of ``ConvexFunctionNorm``,
    ``ConvexFunctionL2Squared`` and ``ConvexFunctionQuadratic`` that caused
    errors when running in graph mode.

* ``tfmri.linalg``:

  * Fixed a bug in internal linear algebra framework that would cause errors
    when running in graph mode.
