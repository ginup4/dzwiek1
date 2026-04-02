#!/usr/bin/env python3

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from math import sqrt, log2
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

srate = None
MAX_SIGNAL_PLOT_POINTS = 200000


class AudioAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")
        self.root.geometry("760x620")

        self.filename = None
        self.last_analysis = None
        self.frame_length_ms = tk.IntVar(value=40)
        self.show_volume = tk.BooleanVar(value=False)
        self.show_ste = tk.BooleanVar(value=False)
        self.show_zcr = tk.BooleanVar(value=False)
        self.show_fund = tk.BooleanVar(value=False)
        self.show_fund_amdf = tk.BooleanVar(value=False)
        self.show_voiced = tk.BooleanVar(value=False)
        self.show_speech_music = tk.BooleanVar(value=False)
        self.show_silence = tk.BooleanVar(value=False)
        self.silence_vol_threshold = tk.DoubleVar(value=0.01)
        self.silence_zcr_threshold = tk.DoubleVar(value=0.1)

        self._build_ui()

    def _build_ui(self):
        file_frame = ttk.Frame(self.root, padding=10)
        file_frame.pack(fill="x")

        ttk.Button(file_frame, text="Wybierz plik WAV", command=self.pick_file).pack(side="left")
        self.file_label = ttk.Label(file_frame, text="Brak wybranego pliku")
        self.file_label.pack(side="left", padx=10)

        settings_frame = ttk.LabelFrame(self.root, text="Ustawienia", padding=10)
        settings_frame.pack(fill="x", padx=10, pady=5)

        ttk.Label(settings_frame, text="Dlugosc ramki [ms]:").grid(row=0, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.frame_length_ms, width=10).grid(row=0, column=1, sticky="w")

        ttk.Label(settings_frame, text="Silence VOL threshold:").grid(row=1, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.silence_vol_threshold, width=10).grid(row=1, column=1, sticky="w")

        ttk.Label(settings_frame, text="Silence ZCR threshold:").grid(row=2, column=0, sticky="w")
        ttk.Entry(settings_frame, textvariable=self.silence_zcr_threshold, width=10).grid(row=2, column=1, sticky="w")

        options_frame = ttk.LabelFrame(self.root, text="Wykresy parametrow", padding=10)
        options_frame.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(options_frame, text="Volume", variable=self.show_volume).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Short Time Energy", variable=self.show_ste).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="Zero Crossing Rate", variable=self.show_zcr).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Fundamental (autocorr)", variable=self.show_fund).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="Fundamental (AMDF)", variable=self.show_fund_amdf).grid(row=2, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Voiced/Unvoiced", variable=self.show_voiced).grid(row=2, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="Speech/Music", variable=self.show_speech_music).grid(row=3, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Silence", variable=self.show_silence).grid(row=3, column=1, sticky="w")

        actions_frame = ttk.Frame(self.root, padding=10)
        actions_frame.pack(fill="x")
        ttk.Button(actions_frame, text="Rysuj wykresy", command=self.plot).pack(side="left", padx=4)
        ttk.Button(actions_frame, text="Zapisz CSV", command=self.save_csv).pack(side="left", padx=4)
        ttk.Button(actions_frame, text="Zapisz TXT", command=self.save_txt).pack(side="left", padx=4)

        summary_frame = ttk.LabelFrame(self.root, text="Parametry klipu (5.0)", padding=10)
        summary_frame.pack(fill="both", expand=True, padx=10, pady=5)

        self.summary_text = tk.Text(summary_frame, height=14, wrap="word")
        self.summary_text.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        scroll.pack(side="right", fill="y")
        self.summary_text.configure(yscrollcommand=scroll.set)
        self._set_summary("Brak wynikow. Wczytaj WAV i kliknij Rysuj wykresy.")

    def _set_summary(self, text):
        self.summary_text.configure(state="normal")
        self.summary_text.delete("1.0", "end")
        self.summary_text.insert("1.0", text)
        self.summary_text.configure(state="disabled")

    def pick_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.filename = filename
            self.file_label.config(text=filename)

    def save_csv(self):
        if not self.last_analysis:
            messagebox.showwarning("Brak danych", "Najpierw uruchom analize (Rysuj wykresy).")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile="audio_analysis.csv",
        )
        if not out_path:
            return

        metrics = self.last_analysis["clip_metrics"]
        frame_time = self.last_analysis["frame_time"]
        frame_features = self.last_analysis["frame_features"]

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["clip_metric", "value"])
            for key, value in metrics.items():
                writer.writerow([key, value])

            writer.writerow([])
            writer.writerow([
                "frame_index",
                "time_s",
                "volume",
                "short_time_energy",
                "zero_crossing_rate",
                "fundamental_autocorr",
                "fundamental_amdf",
                "silence",
                "voiced_unvoiced",
                "speech_music",
            ])

            f0_amdf = frame_features.get("fundamental_amdf", [])
            for idx in range(len(frame_time)):
                writer.writerow([
                    idx,
                    frame_time[idx],
                    frame_features["volume"][idx],
                    frame_features["short_time_energy"][idx],
                    frame_features["zero_crossing_rate"][idx],
                    frame_features["fundamental_autocorr"][idx],
                    f0_amdf[idx] if idx < len(f0_amdf) else "",
                    frame_features["silence"][idx],
                    frame_features["voiced_unvoiced"][idx],
                    speech_music_label(frame_features["speech_music"][idx]),
                ])

        messagebox.showinfo("Zapisano", f"Zapisano wyniki do CSV:\n{out_path}")

    def save_txt(self):
        if not self.last_analysis:
            messagebox.showwarning("Brak danych", "Najpierw uruchom analize (Rysuj wykresy).")
            return

        out_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text", "*.txt")],
            initialfile="audio_analysis.txt",
        )
        if not out_path:
            return

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(self.last_analysis["summary_text"])
            f.write("\n\nFrame labels (index,time_s,speech_music):\n")
            frame_time = self.last_analysis["frame_time"]
            speech_music = self.last_analysis["frame_features"]["speech_music"]
            for idx, label in enumerate(speech_music):
                f.write(f"{idx},{frame_time[idx]:.3f},{speech_music_label(label)}\n")

        messagebox.showinfo("Zapisano", f"Zapisano wyniki do TXT:\n{out_path}")

    def plot(self):
        global srate

        if not self.filename:
            messagebox.showerror("Blad", "Najpierw wybierz plik WAV")
            return

        try:
            frame_length = int(self.frame_length_ms.get())
            vol_thr = float(self.silence_vol_threshold.get())
            zcr_thr = float(self.silence_zcr_threshold.get())
        except ValueError:
            messagebox.showerror("Blad", "Niepoprawne wartosci numeryczne")
            return

        # wczytanie pliku
        try:
            srate, samples = wavfile.read(self.filename)
        except Exception as exc:
            messagebox.showerror("Blad", f"Nie mozna otworzyc pliku WAV: {exc}")
            return

        dtype = samples.dtype
        samples = samples.astype(np.float32)
        if len(samples.shape) == 2:
            samples = samples[:, 0] + samples[:, 1]
        if dtype == np.uint8:
            samples = (samples - 128.0) / 128.0
        if dtype == np.int16:
            samples /= 2 ** 15
        if dtype == np.int32:
            samples /= 2 ** 31

        slen = samples.shape[0]
        dt = 1 / srate

        # podzial na liste ramek
        frame_slen = int(srate * frame_length / 1000)
        if frame_slen <= 0:
            messagebox.showerror("Blad", "Dlugosc ramki musi byc > 0")
            return
        frames = [samples[s:s + frame_slen] for s in range(0, slen, frame_slen)]

        # pomocnicze listy zeby wykresy mialy sekundy na osi x
        time = np.arange(slen, dtype=np.float64) * dt
        frame_time = np.arange(len(frames), dtype=np.float64) * (frame_length / 1000.0)

        # Obliczamy baze cech raz; sa uzywane do wykresow, klasyfikacji i zapisu.
        volume_values = volume(frames)
        ste_values = short_time_energy(frames)
        zcr_values = zero_crossing_rate(frames)
        f0_autocorr_values = fundamental_autocorr(frames)
        f0_amdf_values = fundamental_amdf(frames) if self.show_fund_amdf.get() else []
        silence_flags = silence_from_features(volume_values, zcr_values, vol_thr, zcr_thr)
        voiced_flags = voiced_unvoiced_from_features(volume_values, zcr_values, f0_autocorr_values, vol_thr, zcr_thr)
        speech_music_flags = speech_music_from_features(volume_values, zcr_values, f0_autocorr_values, silence_flags)

        clip_metrics = clip_metrics_from_features(
            samples,
            srate,
            volume_values,
            ste_values,
            zcr_values,
            f0_autocorr_values,
            silence_flags,
            voiced_flags,
            speech_music_flags,
        )
        summary_text = format_clip_metrics(clip_metrics)
        self._set_summary(summary_text)

        frame_features = {
            "volume": volume_values,
            "short_time_energy": ste_values,
            "zero_crossing_rate": zcr_values,
            "fundamental_autocorr": f0_autocorr_values,
            "fundamental_amdf": f0_amdf_values,
            "silence": silence_flags,
            "voiced_unvoiced": voiced_flags,
            "speech_music": speech_music_flags,
        }
        self.last_analysis = {
            "filename": self.filename,
            "sample_rate": srate,
            "frame_length_ms": frame_length,
            "frame_time": frame_time,
            "frame_features": frame_features,
            "clip_metrics": clip_metrics,
            "summary_text": summary_text,
        }

        result_map = {}
        if self.show_volume.get():
            result_map["volume"] = volume_values
        if self.show_ste.get():
            result_map["short_time_energy"] = ste_values
        if self.show_zcr.get():
            result_map["zero_crossing_rate"] = zcr_values
        if self.show_fund.get():
            result_map["fundamental_autocorr"] = f0_autocorr_values
        if self.show_fund_amdf.get():
            result_map["fundamental_amdf"] = f0_amdf_values
        if self.show_voiced.get():
            result_map["voiced_unvoiced"] = voiced_flags
        if self.show_speech_music.get():
            result_map["speech_music"] = speech_music_flags
        if self.show_silence.get():
            result_map["silence"] = silence_flags

        plot_order = []
        if self.show_volume.get():
            plot_order.append("volume")
        if self.show_ste.get():
            plot_order.append("short_time_energy")
        if self.show_zcr.get():
            plot_order.append("zero_crossing_rate")
        if self.show_fund.get():
            plot_order.append("fundamental_autocorr")
        if self.show_fund_amdf.get():
            plot_order.append("fundamental_amdf")
        if self.show_voiced.get():
            plot_order.append("voiced_unvoiced")
        if self.show_speech_music.get():
            plot_order.append("speech_music")
        if self.show_silence.get():
            plot_order.append("silence")

        # wykresy
        nplots = 1 + len(plot_order)
        plt.figure(figsize=(10, 2.7 * nplots))
        ax_signal = plt.subplot(nplots, 1, 1)
        # rysowanie oryginalnego sygnalu
        ax_signal.set_title("original recording")
        plot_time, plot_samples = downsample_signal_for_plot(time, samples)
        ax_signal.plot(plot_time, plot_samples, label="signal")

        show_overlay_legend = False

        if self.show_silence.get():
            show_overlay_legend = True
            silence_flags = result_map["silence"]
            silence_labeled = False
            for start_sec, end_sec in true_segments(silence_flags, frame_length / 1000, slen * dt):
                label = "silence" if not silence_labeled else None
                ax_signal.axvspan(start_sec, end_sec, color="orange", alpha=0.25, label=label)
                silence_labeled = True

        if self.show_voiced.get():
            show_overlay_legend = True
            voiced_labeled = False
            for start_sec, end_sec in true_segments(voiced_flags, frame_length / 1000, slen * dt):
                label = "voiced" if not voiced_labeled else None
                ax_signal.axvspan(start_sec, end_sec, color="green", alpha=0.18, label=label)
                voiced_labeled = True

        if self.show_speech_music.get():
            show_overlay_legend = True
            speech_labeled = False
            music_labeled = False
            for label, start_sec, end_sec in label_segments(speech_music_flags, frame_length / 1000, slen * dt):
                if label == 0:
                    span_label = "speech" if not speech_labeled else None
                    ax_signal.axvspan(start_sec, end_sec, color="deepskyblue", alpha=0.14, label=span_label)
                    speech_labeled = True
                elif label == 1:
                    span_label = "music" if not music_labeled else None
                    ax_signal.axvspan(start_sec, end_sec, color="crimson", alpha=0.14, label=span_label)
                    music_labeled = True

        if show_overlay_legend:
            ax_signal.legend(loc="upper right", fontsize=9)

        # rysowanie parametrow wybranych w GUI
        for i, name in enumerate(plot_order):
            ax = plt.subplot(nplots, 1, 2 + i)
            ax.set_title(name)
            if name == "speech_music":
                ax.step(frame_time, result_map[name], where="post")
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels(["silence", "speech", "music"])
            else:
                ax.plot(frame_time, result_map[name])

        plt.tight_layout()
        plt.show()

# funkcje liczace parametry sygnalu

def downsample_signal_for_plot(time, samples, max_points=MAX_SIGNAL_PLOT_POINTS):
    sample_count = len(samples)
    if sample_count <= max_points:
        return time, samples
    step = max(1, int(np.ceil(sample_count / max_points)))
    return time[::step], samples[::step]

def true_segments(flags, frame_duration_s, total_duration_s):
    start_idx = None
    for idx, flag in enumerate(flags):
        if flag and start_idx is None:
            start_idx = idx
        if not flag and start_idx is not None:
            yield start_idx * frame_duration_s, min(idx * frame_duration_s, total_duration_s)
            start_idx = None

    if start_idx is not None:
        yield start_idx * frame_duration_s, total_duration_s

def label_segments(labels, frame_duration_s, total_duration_s):
    if not labels:
        return

    current_label = labels[0]
    start_idx = 0
    for idx in range(1, len(labels)):
        if labels[idx] != current_label:
            yield current_label, start_idx * frame_duration_s, min(idx * frame_duration_s, total_duration_s)
            current_label = labels[idx]
            start_idx = idx

    yield current_label, start_idx * frame_duration_s, total_duration_s

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

def autocorr(frame, l):
    upper = len(frame) - l - 1
    if upper <= 0:
        return 0.0
    total = 0.0
    for i in range(upper):
        total += float(frame[i + l]) * float(frame[i])
    return total

def amdf(frame, l):
    upper = len(frame) - l - 1
    if upper <= 0:
        return 0.0
    total = 0.0
    for i in range(upper):
        total += abs(float(frame[i]) - float(frame[i + l]))
    return total / upper

def _prepare_pitch_frame(frame):
    target_pitch_rate = 8000
    step = max(1, int(srate / target_pitch_rate))
    if step == 1:
        return frame, float(srate)
    return frame[::step], float(srate) / step

def _lag_bounds(frame_len, sample_rate):
    min_f0 = 50
    max_f0 = 400
    min_lag = max(1, int(sample_rate / max_f0))
    max_lag = min(frame_len - 2, int(sample_rate / min_f0))
    if min_lag > max_lag:
        return None
    return range(min_lag, max_lag + 1)

def fundamental_autocorr(frames):
    ret = []
    for frame in frames:
        pitch_frame, pitch_rate = _prepare_pitch_frame(frame)
        lags = _lag_bounds(len(pitch_frame), pitch_rate)
        if lags is None:
            ret.append(0.0)
            continue
        best_lag = max(lags, key=lambda l: autocorr(pitch_frame, l))
        score = autocorr(pitch_frame, best_lag)
        if score <= 0:
            ret.append(0.0)
            continue
        ret.append(pitch_rate / best_lag)
    return ret

def fundamental_amdf(frames):
    ret = []
    for frame in frames:
        pitch_frame, pitch_rate = _prepare_pitch_frame(frame)
        lags = _lag_bounds(len(pitch_frame), pitch_rate)
        if lags is None:
            ret.append(0.0)
            continue
        best_lag = min(lags, key=lambda l: amdf(pitch_frame, l))
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

def voiced_unvoiced(frames, vol_threshold, zcr_threshold):
    vol = volume(frames)
    zcr = zero_crossing_rate(frames)
    f0 = fundamental_autocorr(frames)
    return voiced_unvoiced_from_features(vol, zcr, f0, vol_threshold, zcr_threshold)

def silence(frames, vol_threshold, zcr_threshold):
    vol = volume(frames)
    zcr = zero_crossing_rate(frames)
    return silence_from_features(vol, zcr, vol_threshold, zcr_threshold)

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerGUI(root)
    root.mainloop()
