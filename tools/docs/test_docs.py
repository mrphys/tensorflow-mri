import doctest
import pathlib
import sys
wdir = pathlib.Path().absolute()
sys.path.insert(0, str(wdir))

from tensorflow_mri.python.activations import complex_activations
from tensorflow_mri.python.ops import array_ops
from tensorflow_mri.python.ops import wavelet_ops

kwargs = dict(raise_on_error=True)

doctest.testmod(complex_activations, **kwargs)
doctest.testmod(array_ops, **kwargs)
doctest.testmod(wavelet_ops, **kwargs)
