Release 0.19.0
==============

Major Features and Improvements
-------------------------------

* ``tfmri.recon``:

  * Added new argument ``preserve_phase`` to ``tfmri.recon.pf``. This allows
    the user to recover the phase as well as the magnitude during partial
    Fourier reconstruction. Argument ``return_complex`` has the same behaviour
    and is now deprecated.


Bug Fixes and Other Changes
---------------------------

* Fixed a bug with *k*-space weighting in homodyne detection method of
  ``tfmri.recon.pf``. 
