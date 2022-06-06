Release 0.20.0
==============

Major Features and Improvements
-------------------------------

* ``tfmri.models``:

  * Added new models ``UNet1D``, ``UNet2D`` and ``UNet3D``. These replace
    the previous ``UNet`` layer, which is now deprecated.
  * Added tight frame support to U-Net models. This can be enabled with
    ``use_tight_frame=True``.
  
* ``tfmri.layers``:

  * Added new layers ``DWT1D``, ``DWT2D``, ``DWT3D``, ``IDWT1D``, ``IDWT2D``,
    and ``IDWT3D`` to compute 1D, 2D and 3D forward and inverse discrete wavelet
    transforms.
  * Layer ``UNet`` is now deprecated in favour of the new endpoints in
    the ``tfmri.models`` submodule.


Bug Fixes and Other Changes
---------------------------

* ``tfmri.signal``:

  * Improved static shape inference for ``dwt`` op.
