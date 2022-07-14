Release 0.21.0
==============

Breaking Changes
----------------

* ``tfmri.convex``:

  * Argument ``ndim`` has been removed from all functions.
  * All functions will now require the domain dimension (or shape) to be
    specified. Therefore, in most functions `domain_dimension` is now the first
    positional argument.

Major Features and Improvements
-------------------------------

* ``tfmri.linalg``:

  * Added new class ``LinearOperatorWavelet``.


Bug Fixes and Other Changes
---------------------------

* ``tfmri.recon``:

  * Improved error reporting for ``least_squares``.
