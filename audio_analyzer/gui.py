import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np

from .analysis import analyze_signal
from .audio_io import load_wav_mono
from .spectral_features import apply_window, fft_spectrum
from .time_features import speech_music_label
from .utils import downsample_signal_for_plot, label_segments, true_segments


class AudioAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")
        self.root.geometry("780x720")

        self.filename = None
        self.samples = None
        self.sample_rate = None
        self.last_loaded_filename = None
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
        self.show_spectral_centroid = tk.BooleanVar(value=False)
        self.show_spectral_bandwidth = tk.BooleanVar(value=False)
        self.show_spectral_rolloff = tk.BooleanVar(value=False)
        self.show_spectral_flatness = tk.BooleanVar(value=False)
        self.silence_vol_threshold = tk.DoubleVar(value=0.01)
        self.silence_zcr_threshold = tk.DoubleVar(value=0.1)

        self.fft_use_full_signal = tk.BooleanVar(value=True)
        self.fft_start_s = tk.DoubleVar(value=0.0)
        self.fft_length_ms = tk.IntVar(value=50)
        self.window_name = tk.StringVar(value="hann")
        self.show_windowed_time = tk.BooleanVar(value=True)
        self.show_windowed_fft = tk.BooleanVar(value=True)

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

        spectral_frame = ttk.LabelFrame(self.root, text="Parametry widmowe", padding=10)
        spectral_frame.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(spectral_frame, text="Spectral centroid", variable=self.show_spectral_centroid).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(spectral_frame, text="Spectral bandwidth", variable=self.show_spectral_bandwidth).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(spectral_frame, text="Spectral rolloff", variable=self.show_spectral_rolloff).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(spectral_frame, text="Spectral flatness", variable=self.show_spectral_flatness).grid(row=1, column=1, sticky="w")

        fft_frame = ttk.LabelFrame(self.root, text="FFT i okna", padding=10)
        fft_frame.pack(fill="x", padx=10, pady=5)

        ttk.Checkbutton(fft_frame, text="Caly sygnal", variable=self.fft_use_full_signal).grid(row=0, column=0, sticky="w")

        ttk.Label(fft_frame, text="Start [s]:").grid(row=1, column=0, sticky="w")
        ttk.Entry(fft_frame, textvariable=self.fft_start_s, width=10).grid(row=1, column=1, sticky="w")
        ttk.Label(fft_frame, text="Dlugosc [ms]:").grid(row=1, column=2, sticky="w")
        ttk.Entry(fft_frame, textvariable=self.fft_length_ms, width=10).grid(row=1, column=3, sticky="w")

        ttk.Label(fft_frame, text="Okno:").grid(row=2, column=0, sticky="w")
        window_combo = ttk.Combobox(
            fft_frame,
            textvariable=self.window_name,
            state="readonly",
            width=12,
            values=["prostokatne", "trojkatne", "hamming", "hann", "blackman"],
        )
        window_combo.grid(row=2, column=1, sticky="w")

        ttk.Checkbutton(
            fft_frame,
            text="Pokaz sygnal po oknie (czas)",
            variable=self.show_windowed_time,
        ).grid(row=3, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(
            fft_frame,
            text="Pokaz widmo po oknie",
            variable=self.show_windowed_fft,
        ).grid(row=3, column=2, columnspan=2, sticky="w")

        actions_frame = ttk.Frame(self.root, padding=10)
        actions_frame.pack(fill="x")
        ttk.Button(actions_frame, text="Rysuj wykresy", command=self.plot).pack(side="left", padx=4)
        ttk.Button(actions_frame, text="Rysuj FFT", command=self.plot_fft).pack(side="left", padx=4)
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

    def _ensure_samples_loaded(self):
        if not self.filename:
            raise ValueError("Najpierw wybierz plik WAV")
        if self.samples is not None and self.sample_rate is not None and self.last_loaded_filename == self.filename:
            return self.sample_rate, self.samples
        sample_rate, samples = load_wav_mono(self.filename)
        self.samples = samples
        self.sample_rate = sample_rate
        self.last_loaded_filename = self.filename
        return sample_rate, samples

    def pick_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.filename = filename
            self.file_label.config(text=filename)
            self.samples = None
            self.sample_rate = None
            self.last_loaded_filename = None
            self.last_analysis = None

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

        def value_at(values, idx):
            return values[idx] if idx < len(values) else ""

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
                "spectral_centroid_hz",
                "spectral_bandwidth_hz",
                "spectral_rolloff_hz",
                "spectral_flatness",
            ])

            spectral_centroid = frame_features.get("spectral_centroid_hz", [])
            spectral_bandwidth = frame_features.get("spectral_bandwidth_hz", [])
            spectral_rolloff = frame_features.get("spectral_rolloff_hz", [])
            spectral_flatness = frame_features.get("spectral_flatness", [])

            for idx in range(len(frame_time)):
                speech_value = value_at(frame_features["speech_music"], idx)
                speech_label = speech_music_label(speech_value) if speech_value != "" else ""
                writer.writerow([
                    idx,
                    frame_time[idx],
                    value_at(frame_features["volume"], idx),
                    value_at(frame_features["short_time_energy"], idx),
                    value_at(frame_features["zero_crossing_rate"], idx),
                    value_at(frame_features["fundamental_autocorr"], idx),
                    value_at(frame_features["fundamental_amdf"], idx),
                    value_at(frame_features["silence"], idx),
                    value_at(frame_features["voiced_unvoiced"], idx),
                    speech_label,
                    value_at(spectral_centroid, idx),
                    value_at(spectral_bandwidth, idx),
                    value_at(spectral_rolloff, idx),
                    value_at(spectral_flatness, idx),
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

        if frame_length <= 0:
            messagebox.showerror("Blad", "Dlugosc ramki musi byc > 0")
            return

        try:
            sample_rate, samples = self._ensure_samples_loaded()
        except Exception as exc:
            messagebox.showerror("Blad", f"Nie mozna otworzyc pliku WAV: {exc}")
            return

        if len(samples) == 0:
            messagebox.showerror("Blad", "Pusty sygnal")
            return

        try:
            analysis = analyze_signal(
                samples,
                sample_rate,
                frame_length,
                vol_thr,
                zcr_thr,
                include_amdf=self.show_fund_amdf.get(),
            )
        except ValueError as exc:
            messagebox.showerror("Blad", str(exc))
            return

        frame_time = analysis["frame_time"]
        frame_features = analysis["frame_features"]
        time_axis = analysis["time"]
        summary_text = analysis["summary_text"]

        self._set_summary(summary_text)

        self.last_analysis = {
            "filename": self.filename,
            "sample_rate": sample_rate,
            "frame_length_ms": frame_length,
            "frame_time": frame_time,
            "frame_features": frame_features,
            "clip_metrics": analysis["clip_metrics"],
            "summary_text": summary_text,
        }

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
        if self.show_spectral_centroid.get():
            plot_order.append("spectral_centroid_hz")
        if self.show_spectral_bandwidth.get():
            plot_order.append("spectral_bandwidth_hz")
        if self.show_spectral_rolloff.get():
            plot_order.append("spectral_rolloff_hz")
        if self.show_spectral_flatness.get():
            plot_order.append("spectral_flatness")

        plot_titles = {
            "volume": "volume",
            "short_time_energy": "short_time_energy",
            "zero_crossing_rate": "zero_crossing_rate",
            "fundamental_autocorr": "fundamental_autocorr",
            "fundamental_amdf": "fundamental_amdf",
            "voiced_unvoiced": "voiced_unvoiced",
            "speech_music": "speech_music",
            "silence": "silence",
            "spectral_centroid_hz": "spectral_centroid_hz",
            "spectral_bandwidth_hz": "spectral_bandwidth_hz",
            "spectral_rolloff_hz": "spectral_rolloff_hz",
            "spectral_flatness": "spectral_flatness",
        }

        nplots = 1 + len(plot_order)
        plt.figure(figsize=(10, 2.7 * nplots))
        ax_signal = plt.subplot(nplots, 1, 1)
        ax_signal.set_title("original recording")
        plot_time, plot_samples = downsample_signal_for_plot(time_axis, samples)
        ax_signal.plot(plot_time, plot_samples, label="signal")

        frame_duration_s = frame_length / 1000.0
        total_duration_s = len(samples) / sample_rate if sample_rate else 0.0
        silence_flags = frame_features["silence"]
        voiced_flags = frame_features["voiced_unvoiced"]
        speech_music_flags = frame_features["speech_music"]

        show_overlay_legend = False

        if self.show_silence.get():
            show_overlay_legend = True
            silence_labeled = False
            for start_sec, end_sec in true_segments(silence_flags, frame_duration_s, total_duration_s):
                label = "silence" if not silence_labeled else None
                ax_signal.axvspan(start_sec, end_sec, color="orange", alpha=0.25, label=label)
                silence_labeled = True

        if self.show_voiced.get():
            show_overlay_legend = True
            voiced_labeled = False
            for start_sec, end_sec in true_segments(voiced_flags, frame_duration_s, total_duration_s):
                label = "voiced" if not voiced_labeled else None
                ax_signal.axvspan(start_sec, end_sec, color="green", alpha=0.18, label=label)
                voiced_labeled = True

        if self.show_speech_music.get():
            show_overlay_legend = True
            speech_labeled = False
            music_labeled = False
            for label, start_sec, end_sec in label_segments(speech_music_flags, frame_duration_s, total_duration_s):
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

        for i, name in enumerate(plot_order):
            ax = plt.subplot(nplots, 1, 2 + i)
            ax.set_title(plot_titles.get(name, name))
            if name in ("speech_music", "silence", "voiced_unvoiced"):
                ax.step(frame_time, frame_features[name], where="post")
                if name == "speech_music":
                    ax.set_yticks([-1, 0, 1])
                    ax.set_yticklabels(["silence", "speech", "music"])
            else:
                ax.plot(frame_time, frame_features[name])

        plt.tight_layout()
        plt.show()

    def plot_fft(self):
        if not self.filename:
            messagebox.showerror("Blad", "Najpierw wybierz plik WAV")
            return

        try:
            sample_rate, samples = self._ensure_samples_loaded()
        except Exception as exc:
            messagebox.showerror("Blad", f"Nie mozna otworzyc pliku WAV: {exc}")
            return

        if len(samples) == 0:
            messagebox.showerror("Blad", "Pusty sygnal")
            return

        try:
            start_s = float(self.fft_start_s.get())
            length_ms = int(self.fft_length_ms.get())
        except ValueError:
            messagebox.showerror("Blad", "Niepoprawne wartosci FFT")
            return

        use_full = self.fft_use_full_signal.get()
        if use_full:
            segment = samples
            segment_label = "full signal"
        else:
            if length_ms <= 0:
                messagebox.showerror("Blad", "Dlugosc FFT musi byc > 0")
                return
            if start_s < 0:
                messagebox.showerror("Blad", "Start FFT musi byc >= 0")
                return
            start_idx = int(start_s * sample_rate)
            if start_idx >= len(samples):
                messagebox.showerror("Blad", "Start FFT poza zakresem sygnalu")
                return
            length_samples = int(sample_rate * length_ms / 1000.0)
            if length_samples <= 0:
                messagebox.showerror("Blad", "Dlugosc FFT musi byc > 0")
                return
            end_idx = min(start_idx + length_samples, len(samples))
            if end_idx <= start_idx:
                messagebox.showerror("Blad", "Za krotki fragment do FFT")
                return
            segment = samples[start_idx:end_idx]
            segment_label = f"segment start={start_s:.3f}s length={len(segment) / sample_rate:.3f}s"

        window_name = self.window_name.get()
        try:
            windowed_segment, _ = apply_window(segment, window_name)
        except ValueError as exc:
            messagebox.showerror("Blad", str(exc))
            return

        segment_time = np.arange(len(segment), dtype=np.float64) / sample_rate
        plot_time, plot_samples = downsample_signal_for_plot(segment_time, segment)

        plt.figure(figsize=(10, 6))
        ax_time = plt.subplot(2, 1, 1)
        ax_time.set_title(f"time domain ({segment_label})")
        ax_time.plot(plot_time, plot_samples, label="signal")

        if self.show_windowed_time.get():
            plot_time_w, plot_samples_w = downsample_signal_for_plot(segment_time, windowed_segment)
            ax_time.plot(plot_time_w, plot_samples_w, label=f"windowed ({window_name})", alpha=0.7)
            ax_time.legend(loc="upper right", fontsize=9)

        freqs_raw, mag_raw = fft_spectrum(segment, sample_rate)
        ax_fft = plt.subplot(2, 1, 2)
        ax_fft.set_title("magnitude spectrum (FFT)")
        ax_fft.plot(freqs_raw, mag_raw, label="raw")

        if self.show_windowed_fft.get():
            freqs_win, mag_win = fft_spectrum(windowed_segment, sample_rate)
            ax_fft.plot(freqs_win, mag_win, label=f"windowed ({window_name})", alpha=0.8)
            ax_fft.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        plt.show()


def run_app():
    root = tk.Tk()
    app = AudioAnalyzerGUI(root)
    root.mainloop()
