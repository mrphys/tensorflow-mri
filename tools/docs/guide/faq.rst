Frequently Asked Questions
==========================

**When trying to install TensorFlow MRI, I get an error about OpenEXR which
includes:
``OpenEXR.cpp:36:10: fatal error: ImathBox.h: No such file or directory``. What
do I do?**

OpenEXR is needed by TensorFlow Graphics, which is a dependency of TensorFlow
MRI. This issue can be fixed by installing the OpenEXR library. On
Debian/Ubuntu:

.. code-block:: console

    $ apt install libopenexr-dev
