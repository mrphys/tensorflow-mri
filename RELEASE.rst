Release 0.19.0
==============

This release bumps the supported TensorFlow version to 2.9.

Major Features and Improvements
-------------------------------

* ``tfmri.image``:

  * Added new arguments ``batch_dims`` and ``image_dims`` to
    ``image_gradients``, ``gradient_magnitude``, ``psnr``, ``ssim`` and
    ``ssim_multiscale``.
  * Argument ``rank`` of ``psnr``, ``ssim`` and ``ssim_multiscale`` is now
    deprecated. To update, use ``image_dims`` instead.
  * ``image_gradients`` and ``gradient_magnitude`` now support complex inputs.

* ``tfmri.metrics``:

  * Image quality metrics can now accept complex inputs without also specifying
    ``complex_part``, in which case the unmodified complex values will be passed
    to the downstream function. This may not be supported for all metrics.

* ``tfmri.recon``:

  * Added new argument ``preserve_phase`` to ``tfmri.recon.pf``. This allows
    the user to recover the phase as well as the magnitude during partial
    Fourier reconstruction. Argument ``return_complex`` has the same behaviour
    and is now deprecated.
  * Added new aliases ``adjoint`` (for ``adj``), ``least_squares``
    (for ``lstsq``) and ``partial_fourier`` (for ``pf``). These are now the
    canonical aliases, but the old ones will still be supported.

* ``tfmri.plot``:

  * Added new argument ``norm`` to ``image_sequence``, ``tiled_image`` and
    ``tiled_image_sequence``. This allows the user to specify the scaling
    to be applied before the colormap.

Bug Fixes and Other Changes
---------------------------

* Fixed a bug with *k*-space weighting in homodyne detection method of
  ``tfmri.recon.partial_fourier``. 
