from utils import *
from sklearn.preprocessing import scale

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("./data/scores.csv", index_col=0)

X = np.array(convert_wav_to_image(df))
X = np.array(normalize_dataset(X))
Y = df["score"].values
Y = scale(Y)

for i in range(len(X)):
    plt.figure(figsize=(15,10))
    plt.title(df["path"][i], weight='bold')
    plt.imshow(X[i])
    plt.show()
