Release 0.11.0
==============

This release includes a redesign of the API docs.

Breaking Changes
----------------

* ``tfmri``:

  * ``LinearOperatorMRI``: Argument ``sens_norm`` now defaults to ``True``.
  * ``conjugate_gradient``: Argument ``max_iter`` is now called
    ``max_iterations``.
  * ``SVDCoilCompressor`` renamed to ``CoilCompressorSVD`` for consistency
    with the rest of the API.
  * ``filter_kspace``: Argument ``filter_type`` has been renamed to
    ``filter_fn`` and can accept a callable in addition to a string.

Major Features and Improvements
-------------------------------

* ``tfmri``:

  * Added new ops ``expand_trajectory`` and ``expand_density``, which
    complement the existing ``flatten_trajectory`` and ``flatten_density``.

Bug Fixes and Other Changes
---------------------------

* ``tfmri``:

  * ``hann`` and ``hamming`` now return 0 outside their domain, as expected.
  * ``atanfilt`` now returns the correct values for the negative part of the
    domain.
  * Improved error reporting for ``filter_kspace``.
