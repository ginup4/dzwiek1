from math import log2, sqrt


def volume(frames):
    ret = []
    for frame in frames:
        frame_len = len(frame)
        if frame_len == 0:
            ret.append(0.0)
            continue
        energy_sum = 0.0
        for sample in frame:
            value = float(sample)
            energy_sum += value * value
        ret.append(sqrt(energy_sum / frame_len))
    return ret


def short_time_energy(frames):
    ret = []
    for frame in frames:
        frame_len = len(frame)
        if frame_len == 0:
            ret.append(0.0)
            continue
        energy_sum = 0.0
        for sample in frame:
            value = float(sample)
            energy_sum += value * value
        ret.append(energy_sum / frame_len)
    return ret


def zero_crossing_rate(frames):
    ret = []
    for frame in frames:
        frame_len = len(frame)
        if frame_len < 2:
            ret.append(0.0)
            continue
        prev_sign = 1 if frame[0] > 0 else -1 if frame[0] < 0 else 0
        crossings = 0.0
        for idx in range(1, frame_len):
            current = frame[idx]
            curr_sign = 1 if current > 0 else -1 if current < 0 else 0
            crossings += abs(curr_sign - prev_sign)
            prev_sign = curr_sign
        ret.append(crossings / frame_len / 2.0)
    return ret


def autocorr(frame, lag):
    upper = len(frame) - lag - 1
    if upper <= 0:
        return 0.0
    total = 0.0
    for i in range(upper):
        total += float(frame[i + lag]) * float(frame[i])
    return total


def amdf(frame, lag):
    upper = len(frame) - lag - 1
    if upper <= 0:
        return 0.0
    total = 0.0
    for i in range(upper):
        total += abs(float(frame[i]) - float(frame[i + lag]))
    return total / upper


def _prepare_pitch_frame(frame, sample_rate):
    target_pitch_rate = 8000
    step = max(1, int(sample_rate / target_pitch_rate))
    if step == 1:
        return frame, float(sample_rate)
    return frame[::step], float(sample_rate) / step


def _lag_bounds(frame_len, sample_rate):
    min_f0 = 50
    max_f0 = 400
    min_lag = max(1, int(sample_rate / max_f0))
    max_lag = min(frame_len - 2, int(sample_rate / min_f0))
    if min_lag > max_lag:
        return None
    return range(min_lag, max_lag + 1)


def fundamental_autocorr(frames, sample_rate):
    ret = []
    for frame in frames:
        pitch_frame, pitch_rate = _prepare_pitch_frame(frame, sample_rate)
        lags = _lag_bounds(len(pitch_frame), pitch_rate)
        if lags is None:
            ret.append(0.0)
            continue
        best_lag = max(lags, key=lambda lag: autocorr(pitch_frame, lag))
        score = autocorr(pitch_frame, best_lag)
        if score <= 0:
            ret.append(0.0)
            continue
        ret.append(pitch_rate / best_lag)
    return ret


def fundamental_amdf(frames, sample_rate):
    ret = []
    for frame in frames:
        pitch_frame, pitch_rate = _prepare_pitch_frame(frame, sample_rate)
        lags = _lag_bounds(len(pitch_frame), pitch_rate)
        if lags is None:
            ret.append(0.0)
            continue
        best_lag = min(lags, key=lambda lag: amdf(pitch_frame, lag))
        ret.append(pitch_rate / best_lag)
    return ret


def silence_from_features(vol, zcr, vol_threshold, zcr_threshold):
    return [v < vol_threshold and z < zcr_threshold for v, z in zip(vol, zcr)]


def voiced_unvoiced_from_features(vol, zcr, f0, vol_threshold, zcr_threshold):
    ret = []
    for v, z, pitch in zip(vol, zcr, f0):
        is_voiced = v >= vol_threshold and z <= zcr_threshold and 70 <= pitch <= 400
        ret.append(1 if is_voiced else 0)
    return ret


def speech_music_from_features(vol, zcr, f0, silence_flags):
    vol_median = median(vol)
    labels = []
    prev_pitch = 0.0

    for idx in range(len(vol)):
        if silence_flags[idx]:
            labels.append(-1)
            continue

        v = vol[idx]
        z = zcr[idx]
        pitch = f0[idx]

        speech_score = 0
        music_score = 0

        if 85 <= pitch <= 280:
            speech_score += 2
        elif pitch > 0:
            music_score += 1

        if 0.03 <= z <= 0.20:
            speech_score += 1
        else:
            music_score += 1

        if v <= 1.8 * vol_median:
            speech_score += 1
        else:
            music_score += 1

        if prev_pitch > 0 and pitch > 0:
            if abs(pitch - prev_pitch) < 8:
                music_score += 1
            else:
                speech_score += 1
        if pitch > 0:
            prev_pitch = pitch

        labels.append(0 if speech_score >= music_score else 1)

    return labels


def safe_mean(values):
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def clip_zcr(samples):
    if len(samples) < 2:
        return 0.0
    prev_sign = 1 if samples[0] > 0 else -1 if samples[0] < 0 else 0
    crossings = 0.0
    for idx in range(1, len(samples)):
        current = samples[idx]
        curr_sign = 1 if current > 0 else -1 if current < 0 else 0
        crossings += abs(curr_sign - prev_sign)
        prev_sign = curr_sign
    return crossings / len(samples) / 2.0


def energy_entropy(ste_values):
    if not ste_values:
        return 0.0
    total = sum(v for v in ste_values if v > 0)
    if total <= 0:
        return 0.0
    probs = [v / total for v in ste_values if v > 0]
    if len(probs) < 2:
        return 0.0
    entropy = -sum(p * log2(p) for p in probs)
    return entropy / log2(len(probs))


def speech_music_label(code):
    if code == -1:
        return "silence"
    if code == 0:
        return "speech"
    return "music"


def clip_metrics_from_features(samples, sample_rate, vol, ste, zcr, f0, silence_flags, voiced_flags, speech_music_flags):
    duration_s = len(samples) / sample_rate if sample_rate else 0.0
    if len(samples) > 0:
        energy_sum = 0.0
        peak = 0.0
        for sample in samples:
            value = float(sample)
            energy_sum += value * value
            abs_value = abs(value)
            if abs_value > peak:
                peak = abs_value
        clip_energy = energy_sum / len(samples)
    else:
        clip_energy = 0.0
        peak = 0.0
    clip_rms = sqrt(clip_energy) if clip_energy > 0 else 0.0
    crest_factor = peak / clip_rms if clip_rms > 0 else 0.0

    voiced_f0 = [pitch for pitch, flag in zip(f0, voiced_flags) if flag == 1 and pitch > 0]
    nonsilent = [label for label in speech_music_flags if label != -1]
    speech_ratio = sum(1 for label in nonsilent if label == 0) / len(nonsilent) if nonsilent else 0.0
    music_ratio = sum(1 for label in nonsilent if label == 1) / len(nonsilent) if nonsilent else 0.0
    if speech_ratio == 0 and music_ratio == 0:
        dominant = "unknown"
    elif speech_ratio >= music_ratio:
        dominant = "speech"
    else:
        dominant = "music"

    return {
        "duration_s": round(duration_s, 4),
        "sample_rate_hz": sample_rate,
        "num_samples": len(samples),
        "num_frames": len(vol),
        "clip_rms": round(clip_rms, 6),
        "clip_energy": round(clip_energy, 6),
        "clip_zcr": round(clip_zcr(samples), 6),
        "mean_frame_volume": round(safe_mean(vol), 6),
        "mean_frame_ste": round(safe_mean(ste), 6),
        "mean_frame_zcr": round(safe_mean(zcr), 6),
        "mean_voiced_f0_hz": round(safe_mean(voiced_f0), 3),
        "silence_ratio": round(safe_mean([1 if x else 0 for x in silence_flags]), 4),
        "voiced_ratio": round(safe_mean(voiced_flags), 4),
        "speech_ratio": round(speech_ratio, 4),
        "music_ratio": round(music_ratio, 4),
        "dominant_content": dominant,
        "crest_factor": round(crest_factor, 6),
        "energy_entropy_norm": round(energy_entropy(ste), 6),
    }


def format_clip_metrics(metrics):
    lines = [
        "Wyniki analizy klipu:",
        "",
    ]
    for key, value in metrics.items():
        lines.append(f"{key}: {value}")
    lines.append("")
    lines.append("Uwagi:")
    lines.append("- speech/music to klasyfikacja heurystyczna oparta o cechy czasowe")
    lines.append("- energy_entropy_norm i crest_factor to dodatkowe cechy")
    return "\n".join(lines)


def voiced_unvoiced(frames, sample_rate, vol_threshold, zcr_threshold):
    vol = volume(frames)
    zcr = zero_crossing_rate(frames)
    f0 = fundamental_autocorr(frames, sample_rate)
    return voiced_unvoiced_from_features(vol, zcr, f0, vol_threshold, zcr_threshold)


def silence(frames, vol_threshold, zcr_threshold):
    vol = volume(frames)
    zcr = zero_crossing_rate(frames)
    return silence_from_features(vol, zcr, vol_threshold, zcr_threshold)
