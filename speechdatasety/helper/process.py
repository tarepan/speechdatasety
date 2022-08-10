"""Process data."""

import numpy as np
from numpy.typing import NDArray


s16scale: int =  32768
s16min: int   = -32768
s16max: int   = +32767


def unit_to_s16pcm(unit: NDArray[np.float32]) -> NDArray[np.int16]:
    """Convert [-1, 1) unit-scale fp32 values into [-32768, +32767] s16-scale int16 discrete values.

    Args:
        unit :: (...) - fp32 Values in [-1, 1]
    Returns:
        :: (...) - sint16 Values in [-32768, +32767]
    """
    return np.clip(np.round(s16scale * unit), a_min=s16min, a_max=s16max).astype(np.int16)
