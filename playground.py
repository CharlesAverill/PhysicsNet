from scipy.io.wavfile import read

import numpy as np
import matplotlib.pyplot as plt

wavfile = read("data/piano/notes/out012.wav")

wav_arr = np.array(wavfile[1], dtype=float)
# wav_slice = np.sin(np.arange(0, 100, 0.1))
wav_slice = wav_arr[:250].T[0]
# wav_slice = wav_arr.copy()

fft = np.fft.fft(wav_slice)

print(np.std(fft))

plt.subplot(1, 2, 1)

plt.plot(wav_slice)
plt.ylabel("Amplitude")
plt.xlabel("Time")
plt.title("Wave Function")
plt.legend(["Function"])

plt.subplot(1, 2, 2)

N = wav_slice.size

plt.bar(np.arange(N / 2), (np.abs(fft)[:N // 2] * 1 / N))
plt.ylabel("Amplitude")
plt.xlabel("Frequency")
plt.title("Fourier Transform")
plt.legend(["FFT"])
plt.show()
