import os
import pandas as pd
import numpy as np
import wave
from scipy.io.wavfile import read

file_paths = []
sample_rates = []

for root, dirs, files in os.walk("./data/piano_clipped"):
    for x in files:
        if x.endswith(".wav"):
            file_paths.append(os.path.join(root, x))
            with wave.open(os.path.join(root, x), "rb") as wave_file:
                sample_rates.append(wave_file.getframerate())

rows = []

for fp, sr in zip(file_paths, sample_rates):
    wav_file = read(fp)
    wav_arr = np.array(wav_file[1], dtype=float)
    fft = np.fft.fft(wav_arr)
    score = np.std(fft)
    rows.append([fp, sr, score])

scores_df = pd.DataFrame(rows, columns=["path", "sample_rate", "score"])

scores_df.to_csv("./data/scores.csv")
