Release 0.12.0
==============

Breaking Changes
----------------



Major Features and Improvements
-------------------------------

* ``tfmri.plot``:

  * New module containing plotting utilities.
  * New functions ``image_sequence``, ``tiled_image_sequence`` and ``show``.

* ``tfmri.recon``:

  * New module containing functionality for image reconstruction.
  * New functions ``adj``, ``grappa``, ``lstsq``, ``pf`` and ``sense``. These
    are now the canonical API for image reconstruction.

Bug Fixes and Other Changes
---------------------------

* New API export system, currently enabled for namespaces ``tfmri.linalg``,
  ``tfmri.plot``, ``tfmri.recon`` and ``tfmri.summary``. The remaining
  namespaces will be moved to this API system in future releases.
* Improvements to documentation: reduced verbosity of TOC tree and added links
  for common types.
