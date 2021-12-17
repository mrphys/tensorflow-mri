Release 0.10.0
==============

<INSERT SMALL BLURB ABOUT RELEASE FOCUS AREA AND POTENTIAL TOOLCHAIN CHANGES>

Breaking Changes
----------------

* The argument ``norm`` of ``LinearOperatorFFT`` and ``LinearOperatorNUFFT``
  now defaults to ``'ortho'``. This is the optimal behavior in most
  circumstances, because subsequent application of the operator and its adjoint
  does not scale the input.
* The argument ``norm`` of ``LinearOperatorParallelMRI`` is now called
  ``fft_normalization``. Its behaviour is unchanged.
* The ``TensorBoardImages`` callback no longer accepts the argument
  ``display_func`` to customize display images. Instead, users should subclass
  ``TensorBoardImages`` and override its ``display_image`` method. This enables
  serialization of ``TensorBoardImages`` objects.

Known Caveats
-------------

* <CAVEATS REGARDING THE RELEASE (BUT NOT BREAKING CHANGES).>
* <ADDING/BUMPING DEPENDENCIES SHOULD GO HERE>
* <KNOWN LACK OF SUPPORT ON SOME PLATFORM, SHOULD GO HERE>

Major Features and Improvements
-------------------------------

* ``tfmr``:

  * Added new op ``extract_and_scale_complex_part`` to extract and scale to
    image range parts from complex tensors.
  * Added new argument ``norm`` to ``LinearOperatorSensitivityModulation``,
    used to control whether coil sensitivity maps should be normalized.
  * Added new argument ``normalize_sensitivities`` to
    ``LinearOperatorParallelMRI``, used to control whether coil sensitivity maps
    should be normalized.

* ``tfmr.metrics``:

  * Added new argument ``multichannel`` to ``PeakSignalToNoiseRatio``,
    ``StructuralSimilarity`` and ``MultiscaleStructuralSimilarity``. This
    enables these metrics to accept inputs that do not have a channel axis,
    by setting ``multichannel=False``.
  * Added new argument ``complex_part`` to ``PeakSignalToNoiseRatio``,
    ``StructuralSimilarity`` and ``MultiscaleStructuralSimilarity``. This
    enables these metrics to accept complex inputs and calculate the metric
    from the specified part. 

* ``tfmr.callbacks``:

  * Added new callback ``TensorBoardImagesRecon``, which may be used to store
    images from MRI reconstruction models to TensorBoard.

Bug Fixes and Other Changes
---------------------------

* Fixed a bug in ``TensorBoardImages`` callback that would result in the inputs
  multi-input models receiving only the first input, thus resulting in a failure
  during prediction.
