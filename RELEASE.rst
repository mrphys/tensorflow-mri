Release 0.19.0
==============

This release bumps the supported TensorFlow version to 2.9.

Major Features and Improvements
-------------------------------

* ``tfmri.image``:

  * Added new arguments ``batch_dims`` and ``image_dims`` to ``image_gradients``
    and ``gradient_magnitude``.

* ``tfmri.recon``:

  * Added new argument ``preserve_phase`` to ``tfmri.recon.pf``. This allows
    the user to recover the phase as well as the magnitude during partial
    Fourier reconstruction. Argument ``return_complex`` has the same behaviour
    and is now deprecated.
  * Added new aliases ``adjoint`` (for ``adj``), ``least_squares``
    (for ``lstsq``) and ``partial_fourier`` (for ``pf``). These are now the
    canonical aliases, but the old ones will still be supported.


Bug Fixes and Other Changes
---------------------------

* Fixed a bug with *k*-space weighting in homodyne detection method of
  ``tfmri.recon.pf``. 
