Release 0.20.0
==============

Major Features and Improvements
-------------------------------

* ``tfmri.layers``:

  * Added new layers ``MaxPooling1D``, ``MaxPooling2D``, ``MaxPooling3D``,
    ``AveragePooling1D``, ``AveragePooling2D`` and ``AveragePooling3D``.
    These are drop-in replacements for the core Keras layers of the same name,
    but they also support complex values.
  * Added new layers ``DWT1D``, ``DWT2D``, ``DWT3D``, ``IDWT1D``, ``IDWT2D``,
    and ``IDWT3D`` to compute 1D, 2D and 3D forward and inverse discrete wavelet
    transforms.
  * Layer ``ConvBlock`` is now deprecated in favor of the new endpoints in
    the ``tfmri.models`` submodule.
  * Layer ``UNet`` is now deprecated in favor of the new endpoints in
    the ``tfmri.models`` submodule.

* ``tfmri.models``:

  * Added new models ``ConvBlock1D``, ``ConvBlock2D`` and ``ConvBlock3D``. These
    replace the previous ``ConvBlock`` layer, which is now deprecated.
  * Added new models ``UNet1D``, ``UNet2D`` and ``UNet3D``. These replace
    the previous ``UNet`` layer, which is now deprecated.


Bug Fixes and Other Changes
---------------------------

* ``tfmri.signal``:

  * Improved static shape inference for ``dwt`` op.
