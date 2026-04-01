#!/bin/env python

import sys
import argparse
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def main():
    global args, srate, dt
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-l", "--length", default=40, type=int, help="length of frames in milliseconds")

    # parametry liczone na ramkach
    parser.add_argument("-v", "--volume", action="append_const", dest="funcs", const=volume, help="display volume")
    parser.add_argument("-e", "--ste", action="append_const", dest="funcs", const=short_time_energy, help="display short time energy")
    parser.add_argument("-z", "--zcr", action="append_const", dest="funcs", const=zero_crossing_rate, help="display zero crossing rate")
    parser.add_argument("-f", "--fund", action="append_const", dest="funcs", const=fundamental, help="display fundamental frequency")
    parser.add_argument("-s", "--silence", nargs=2, type=float, metavar=("VOL", "ZCR"), help="silence detection, vol threshold, zcr threshold")

    args = parser.parse_args()
    if args.silence:
        args.funcs.append(silence)

    # wczytanie pliku
    try:
        (srate, samples) = wavfile.read(args.filename)
        print(samples.dtype)
    except:
        print(f"coudn't open {args.filename} as .wav file", file=sys.stderr)
        sys.exit(1)
        
    dtype = samples.dtype
    samples = samples.astype(float)
    if len(samples.shape) == 2:
        samples = samples[:,0] + samples[:,1]
    if dtype == np.int16:
        samples /= 2 ** 15
    if dtype == np.int32:
        samples /= 2 ** 31

    slen = samples.shape[0]
    dt = 1 / srate

    # podzial na liste ramek
    frame_slen = int(srate * args.length / 1000)
    frames = [samples[s:s+frame_slen] for s in range(0, slen, frame_slen)]

    # pomocnicze listy zeby wykresy mialy sekundy na osi x
    time = [i * dt for i in range(slen)]
    frame_time = [i * args.length / 1000 for i in range(len(frames))]

    # wykresy
    if args.funcs is None:
        args.funcs = []
    nplots = 1 + len(args.funcs)
    plt.subplot(nplots, 1, 1)
    # rysowanie oryginalnego sygnalu
    plt.title("original recording")
    plt.plot(time, samples)

    # rysowanie parametrow wybranych w argumentach programu
    for i, func in enumerate(args.funcs):
        plt.subplot(nplots, 1, 2 + i)
        plt.title(func.__name__)
        plt.plot(frame_time, func(frames))

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

def silence(frames):
    vol = volume(frames)
    zcr = zero_crossing_rate(frames)
    return [v < args.silence[0] and z < args.silence[1] for v, z in zip(vol, zcr)]

if __name__ == "__main__":
    main()
