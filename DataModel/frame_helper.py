

import numpy as np

def Normalize(frames):
    normalized = np.float32(frames) / 255.
    return normalized


