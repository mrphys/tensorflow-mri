Release 0.18.0
==============

Major Features and Improvements
-------------------------------

* ``tfmri.convex``:

  * All references to ``ndim`` in ``ConvexFunction`` and its subclasses have
    been deprecated in favor of ``domain_dimension`` or ``domain_shape``. This
    applies to constructor arguments, properties and methods.
  * Refactored static and dynamic shape properties and methods to single-source
    shape information on the ``_shape`` and ``_shape_tensor`` methods.
    ``domain_dimension``, ``domain_dimension_tensor``, ``batch_shape`` and
    ``batch_shape_tensor`` now just call ``shape`` and ``shape_tensor`` and
    should not be overriden.

* ``tfmri.initializers``:

  * New module with initializers ``VarianceScaling``, ``GlorotNormal``,
    ``GlorotUniform``, ``HeNormal``, ``HeUniform``, ``LecunNormal`` and
    ``LecunUniform``. All initializers are drop-in replacements for their
    Keras counterparts and support complex values.

* ``tfmri.io``:

  * New function ``parse_twix`` to parse TWIX RAID files (Siemens raw data).

* ``tfmri.layers``:

  * Added new layers ``Conv1D``, ``Conv2D`` and ``Conv3D``. All layers
    are drop-in replacements for their Keras counterparts and support
    complex values.


Bug Fixes and Other Changes
---------------------------

* Using a new API import system, which should address some issues with
  the autocomplete features of some editors.
