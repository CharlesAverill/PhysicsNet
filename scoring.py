import os
import pandas as pd
import numpy as np
from scipy.io.wavfile import read

file_paths = []

for root, dirs, files in os.walk("."):
    for x in files:
        if x.endswith(".wav"):
            file_paths.append(os.path.join(root, x))

rows = []

for fp in file_paths:
    wav_file = read(fp)
    wav_arr = np.array(wav_file[1], dtype=float)
    fft = np.fft.fft(wav_arr)
    score = np.std(fft)
    rows.append([fp, score])

scores_df = pd.DataFrame(rows, columns=["path", "score"])

scores_df.to_csv("./data/scores.csv")
