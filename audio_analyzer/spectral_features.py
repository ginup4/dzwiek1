import numpy as np

WINDOW_ALIASES = {
    "rect": "prostokatne",
    "rectangle": "prostokatne",
    "triangular": "trojkatne",
    "hanning": "hann",
    "van hann": "hann",
    "blackman": "blackman",
}

WINDOW_FUNCTIONS = {
    "prostokatne": lambda n: np.ones(n, dtype=np.float32),
    "trojkatne": lambda n: np.bartlett(n).astype(np.float32),
    "hamming": lambda n: np.hamming(n).astype(np.float32),
    "hann": lambda n: np.hanning(n).astype(np.float32),
    "blackman": lambda n: np.blackman(n).astype(np.float32),
}


def normalize_window_name(window_name):
    if not window_name:
        return "prostokatne"
    key = window_name.strip().lower()
    return WINDOW_ALIASES.get(key, key)


def get_window(window_name, length):
    if length <= 0:
        return np.array([], dtype=np.float32)
    key = normalize_window_name(window_name)
    if key not in WINDOW_FUNCTIONS:
        raise ValueError(f"Nieznane okno: {window_name}")
    return WINDOW_FUNCTIONS[key](length)


def apply_window(signal, window_name):
    signal = np.asarray(signal, dtype=np.float32)
    window = get_window(window_name, len(signal))
    if len(window) == 0:
        return signal, window
    return signal * window, window


def fft_spectrum(signal, sample_rate):
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) == 0 or sample_rate <= 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)
    spec = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    magnitude = np.abs(spec) / len(signal)
    return freqs, magnitude


def spectral_centroid(freqs, magnitude):
    if len(magnitude) == 0:
        return 0.0
    weight = np.asarray(magnitude, dtype=np.float64)
    total = np.sum(weight)
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * weight) / total)


def spectral_bandwidth(freqs, magnitude):
    if len(magnitude) == 0:
        return 0.0
    centroid = spectral_centroid(freqs, magnitude)
    weight = np.asarray(magnitude, dtype=np.float64)
    total = np.sum(weight)
    if total <= 0:
        return 0.0
    spread = np.sum(((freqs - centroid) ** 2) * weight) / total
    return float(np.sqrt(spread))


def spectral_rolloff(freqs, magnitude, rolloff_pct=0.85):
    if len(magnitude) == 0:
        return 0.0
    power = np.asarray(magnitude, dtype=np.float64) ** 2
    total = np.sum(power)
    if total <= 0:
        return 0.0
    threshold = rolloff_pct * total
    cumulative = np.cumsum(power)
    idx = int(np.searchsorted(cumulative, threshold))
    idx = min(idx, len(freqs) - 1)
    return float(freqs[idx])


def spectral_flatness(magnitude):
    if len(magnitude) == 0:
        return 0.0
    mag = np.maximum(magnitude, 1e-12)
    geo = float(np.exp(np.mean(np.log(mag))))
    arith = float(np.mean(mag))
    if arith <= 0:
        return 0.0
    return geo / arith


def fundamental_cepstrum(magnitude, sample_rate, fmin=50, fmax=400):
    magnitude = np.maximum(magnitude, 1e-10)
    log_spectrum = np.log(magnitude)
    cepstrum = np.fft.irfft(log_spectrum)
    qmin = int(sample_rate / fmax)
    qmax = int(sample_rate / fmin)
    cepstrum_region = cepstrum[qmin:qmax]
    if len(cepstrum_region) == 0:
        return 0.0
    peak_index = np.argmax(cepstrum_region)
    peak_quefrency = peak_index + qmin
    return sample_rate / peak_quefrency


def compute_spectral_features(frames, sample_rate, spectrogram_max_freq, window_name="prostokatne", rolloff_pct=0.85):
    centroid_values = []
    bandwidth_values = []
    rolloff_values = []
    flatness_values = []
    f0_cepstrum_values = []

    magnitude_arrays = []

    if sample_rate <= 0:
        zeros = [0.0] * len(frames)
        return {
            "spectral_centroid_hz": zeros,
            "spectral_bandwidth_hz": zeros,
            "spectral_rolloff_hz": zeros,
            "spectral_flatness": zeros,
        }

    spectrogram_max_i = None

    for frame in frames:
        if len(frame) == 0:
            centroid_values.append(0.0)
            bandwidth_values.append(0.0)
            rolloff_values.append(0.0)
            flatness_values.append(0.0)
            continue

        windowed, _ = apply_window(frame, window_name)
        freqs, magnitude = fft_spectrum(windowed, sample_rate)
        if spectrogram_max_i is None:
            indices = np.where(freqs > spectrogram_max_freq)
            spectrogram_max_i = indices[0][0] if indices[0].size > 0 else -1
        magnitude_arrays.append(magnitude)
        centroid_values.append(spectral_centroid(freqs, magnitude))
        bandwidth_values.append(spectral_bandwidth(freqs, magnitude))
        rolloff_values.append(spectral_rolloff(freqs, magnitude, rolloff_pct))
        flatness_values.append(spectral_flatness(magnitude))
        f0_cepstrum_values.append(fundamental_cepstrum(magnitude, sample_rate))

    spectrogram_matrix = np.column_stack([mag[:spectrogram_max_i] for mag in magnitude_arrays if mag.shape == magnitude_arrays[0].shape])
    spectrogram_matrix /= np.max(spectrogram_matrix)

    return {
        "spectral_centroid_hz": centroid_values,
        "spectral_bandwidth_hz": bandwidth_values,
        "spectral_rolloff_hz": rolloff_values,
        "spectral_flatness": flatness_values,
        "spectrogram": spectrogram_matrix,
        "fundamental_cepstrum": f0_cepstrum_values,
    }
