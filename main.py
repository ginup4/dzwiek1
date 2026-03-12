#!/bin/env python

import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def usage():
    print("usage:")
    print(f"{sys.argv[0]} FILE")
    sys.exit()

def main():
    if len(sys.argv) == 1:
        usage()
    (srate, samples) = wavfile.read(sys.argv[1])
    samples = samples[:,0] + samples[:,1]
    slen = samples.shape[0]
    dt = 1 / srate

    time = [i * dt for i in range(slen)]

    plt.subplot(2, 1, 1)
    plt.plot(time, samples)

    plt.subplot(2, 1, 2)

    plt.show()

if __name__ == "__main__":
    main()
