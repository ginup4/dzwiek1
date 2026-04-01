#!/usr/bin/env python3

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

srate = None


class AudioAnalyzerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Analyzer")
        self.root.geometry("520x360")

        self.filename = None
        self.frame_length_ms = tk.IntVar(value=40)
        self.show_volume = tk.BooleanVar(value=False)
        self.show_ste = tk.BooleanVar(value=False)
        self.show_zcr = tk.BooleanVar(value=False)
        self.show_fund = tk.BooleanVar(value=False)
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
        ttk.Checkbutton(options_frame, text="Fundamental", variable=self.show_fund).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(options_frame, text="Silence", variable=self.show_silence).grid(row=2, column=0, sticky="w")

        actions_frame = ttk.Frame(self.root, padding=10)
        actions_frame.pack(fill="x")
        ttk.Button(actions_frame, text="Rysuj wykresy", command=self.plot).pack(side="left")

    def pick_file(self):
        filename = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if filename:
            self.filename = filename
            self.file_label.config(text=filename)

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
        samples = samples.astype(float)
        if len(samples.shape) == 2:
            samples = samples[:, 0] + samples[:, 1]
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
        time = [i * dt for i in range(slen)]
        frame_time = [i * frame_length / 1000 for i in range(len(frames))]

        funcs = []
        if self.show_volume.get():
            funcs.append(("volume", volume))
        if self.show_ste.get():
            funcs.append(("short_time_energy", short_time_energy))
        if self.show_zcr.get():
            funcs.append(("zero_crossing_rate", zero_crossing_rate))
        if self.show_fund.get():
            funcs.append(("fundamental", fundamental))
        if self.show_silence.get():
            funcs.append(("silence", lambda frm: silence(frm, vol_thr, zcr_thr)))

        # wykresy
        nplots = 1 + len(funcs)
        plt.figure(figsize=(10, 2.7 * nplots))
        plt.subplot(nplots, 1, 1)
        # rysowanie oryginalnego sygnalu
        plt.title("original recording")
        plt.plot(time, samples)

        # rysowanie parametrow wybranych w GUI
        for i, (name, func) in enumerate(funcs):
            plt.subplot(nplots, 1, 2 + i)
            plt.title(name)
            plt.plot(frame_time, func(frames))

        plt.tight_layout()
        plt.show()

# funkcje liczace parametry sygnalu

def volume(frames):
    ret = []
    for frame in frames:
        ret.append(sqrt(sum(s ** 2 for s in frame) / len(frame)))
    return ret

def short_time_energy(frames):
    ret = []
    for frame in frames:
        ret.append(sum(s ** 2 for s in frame) / len(frame))
    return ret

def zero_crossing_rate(frames):
    ret = []
    for frame in frames:
        signs = np.sign(frame)
        ret.append(sum(abs(signs[i] - signs[i-1]) for i in range(1, len(signs))) / len(frame) / 2)
    return ret

def autocorr(frame, l):
    return sum(frame[i+l] * frame[i] for i in range(len(frame) - l - 1))

def amfd(frame, l):
    return sum(abs(frame[i] - frame[i+l]) for i in range(len(frame) - l - 1)) / (len(frame) - l - 1)

def fundamental(frames):
    ret = []
    for frame in frames:
        ret.append(srate / max(range(20, len(frame)-1), key=lambda l: autocorr(frame, l)))
    return ret

def silence(frames, vol_threshold, zcr_threshold):
    vol = volume(frames)
    zcr = zero_crossing_rate(frames)
    return [v < vol_threshold and z < zcr_threshold for v, z in zip(vol, zcr)]

if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzerGUI(root)
    root.mainloop()
