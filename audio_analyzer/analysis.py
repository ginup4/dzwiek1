import numpy as np

from .spectral_features import compute_spectral_features
from .time_features import (
    clip_metrics_from_features,
    format_clip_metrics,
    fundamental_amdf,
    fundamental_autocorr,
    short_time_energy,
    silence_from_features,
    speech_music_from_features,
    volume,
    voiced_unvoiced_from_features,
    zero_crossing_rate,
)


def frame_signal(samples, sample_rate, frame_length_ms):
    frame_slen = int(sample_rate * frame_length_ms / 1000)
    if frame_slen <= 0:
        raise ValueError("Dlugosc ramki musi byc > 0")
    slen = len(samples)
    frames = [samples[s:s + frame_slen] for s in range(0, slen, frame_slen)]
    if sample_rate > 0:
        time_axis = np.arange(slen, dtype=np.float64) / sample_rate
    else:
        time_axis = np.zeros(slen, dtype=np.float64)
    frame_time = np.arange(len(frames), dtype=np.float64) * (frame_length_ms / 1000.0)
    return frames, time_axis, frame_time


def analyze_signal(samples, sample_rate, frame_length_ms, vol_threshold, zcr_threshold, include_amdf=False):
    frames, time_axis, frame_time = frame_signal(samples, sample_rate, frame_length_ms)

    volume_values = volume(frames)
    ste_values = short_time_energy(frames)
    zcr_values = zero_crossing_rate(frames)
    f0_autocorr_values = fundamental_autocorr(frames, sample_rate)
    f0_amdf_values = fundamental_amdf(frames, sample_rate) if include_amdf else []
    silence_flags = silence_from_features(volume_values, zcr_values, vol_threshold, zcr_threshold)
    voiced_flags = voiced_unvoiced_from_features(volume_values, zcr_values, f0_autocorr_values, vol_threshold, zcr_threshold)
    speech_music_flags = speech_music_from_features(volume_values, zcr_values, f0_autocorr_values, silence_flags)

    spectral_features = compute_spectral_features(frames, sample_rate)
    clip_metrics = clip_metrics_from_features(
        samples,
        sample_rate,
        volume_values,
        ste_values,
        zcr_values,
        f0_autocorr_values,
        silence_flags,
        voiced_flags,
        speech_music_flags,
    )
    summary_text = format_clip_metrics(clip_metrics)

    frame_features = {
        "volume": volume_values,
        "short_time_energy": ste_values,
        "zero_crossing_rate": zcr_values,
        "fundamental_autocorr": f0_autocorr_values,
        "fundamental_amdf": f0_amdf_values,
        "silence": silence_flags,
        "voiced_unvoiced": voiced_flags,
        "speech_music": speech_music_flags,
        "spectral_centroid_hz": spectral_features["spectral_centroid_hz"],
        "spectral_bandwidth_hz": spectral_features["spectral_bandwidth_hz"],
        "spectral_rolloff_hz": spectral_features["spectral_rolloff_hz"],
        "spectral_flatness": spectral_features["spectral_flatness"],
    }

    duration_s = len(samples) / sample_rate if sample_rate else 0.0

    return {
        "time": time_axis,
        "frame_time": frame_time,
        "frame_length_ms": frame_length_ms,
        "frame_features": frame_features,
        "clip_metrics": clip_metrics,
        "summary_text": summary_text,
        "duration_s": duration_s,
    }
