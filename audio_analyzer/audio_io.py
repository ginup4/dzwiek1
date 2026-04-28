import numpy as np
from scipy.io import wavfile


def load_wav_mono(path):
    sample_rate, samples = wavfile.read(path)
    dtype = samples.dtype
    samples = samples.astype(np.float32)
    if samples.ndim == 2:
        samples = samples[:, 0] + samples[:, 1]
    if dtype == np.uint8:
        samples = (samples - 128.0) / 128.0
    elif dtype == np.int16:
        samples /= 2 ** 15
    elif dtype == np.int32:
        samples /= 2 ** 31
    return sample_rate, samples
