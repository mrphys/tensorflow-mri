Release 0.14.0
==============

Major Features and Improvements
-------------------------------

* ``tfmri.convex``:

  * Added new classes ``ConvexFunctionNorm``, ``ConvexFunctionIndicatorBall``,
    ``ConvexFunctionIndicatorL1Ball`` and ``ConvexFunctionIndicatorL2Ball``.
  * Added new method ``conj`` to ``ConvexFunction`` and its subclasses to
    support computation of convex conjugates.

* ``tfmri.math``:

  * Added new indicator functions: ``indicator_box``, ``indicator_simplex`` and
    ``indicator_ball``.
  * Added new projection functions: ``project_onto_box``,
    ``project_onto_simplex`` and ``project_onto_ball``.

* ``tfmri.plot``:

  * ``image_sequence`` and ``tiled_image_sequence`` now support RGB(A) data.
  * ``image_sequence`` and ``tiled_image_sequence`` now support use of
    titles via the arguments ``fig_title`` and ``subplot_titles``. 


Bug Fixes and Other Changes
---------------------------

* ``tfmri.convex``:

  * ``ConvexFunction`` will no longer raise an error when passed a
    ``tf.Tensor`` in the ``scale`` parameter.

* ``tfmri.sampling``:

  * ``spiral_trajectory`` will now return a tensor with known static shape.
