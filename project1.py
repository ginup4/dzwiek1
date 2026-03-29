#!/bin/env python

import sys
import argparse
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename")
    parser.add_argument("-l", "--length", default=20, type=int, help="length of frames in milliseconds")

    # parametry liczone na ramkach
    parser.add_argument("-v", "--volume", action="append_const", dest="funcs", const=volume, help="display volume")
    parser.add_argument("-e", "--ste", action="append_const", dest="funcs", const=ste, help="display short time energy")
    parser.add_argument("-z", "--zcr", action="append_const", dest="funcs", const=zcr, help="display zero crossing rate")

    args = parser.parse_args()

    # wczytanie pliku
    try:
        (srate, samples) = wavfile.read(args.filename)
    except:
        print(f"coudn't open {args.filename} as .wav file", file=sys.stderr)
        sys.exit(1)
        
    # zamiana na float i mapowanie na przedzial -1 do 1
    # nie wiem czy tak powinno sie robic
    samples = samples.astype(float)
    if len(samples.shape) == 2:
        samples = samples[:,0] + samples[:,1]
    samples /= np.abs(samples).max()

    slen = samples.shape[0]
    dt = 1 / srate

    # podzial na liste ramek
    frame_slen = int(srate * args.length / 1000)
    frames = [samples[s:s+frame_slen] for s in range(0, slen, frame_slen)]

    # pomocnicze listy zeby wykresy mialy sekundy na osi x
    time = [i * dt for i in range(slen)]
    frame_time = [i * args.length / 1000 for i in range(len(frames))]

    if args.funcs is None:
        args.funcs = []
    nplots = 1 + len(args.funcs)
    plt.subplot(nplots, 1, 1)
    # rysowanie oryginalnego sygnalu
    plt.plot(time, samples)

    # rysowanie parametrow wybranych w argumentach programu
    for i, func in enumerate(args.funcs):
        plt.subplot(nplots, 1, 2 + i)
        plt.plot(frame_time, func(frames))

    plt.show()

# funkcje liczace parametry sygnalu

def volume(frames):
    ret = []
    for frame in frames:
        ret.append(sqrt(sum(s ** 2 for s in frame) / len(frame)))
    return ret

def ste(frames):
    ret = []
    for frame in frames:
        ret.append(sum(s ** 2 for s in frame) / len(frame))
    return ret

def zcr(frames):
    ret = []
    for frame in frames:
        signs = np.sign(frame)
        ret.append(sum(abs(signs[i] - signs[i-1]) for i in range(1, len(signs))) / len(frame) / 2)
    return ret

if __name__ == "__main__":
    main()
