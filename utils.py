import numpy as np
import librosa

# Preprocessing parameters
sr = 44100  # Sampling rate
duration = 5
hop_length = 347  # to make time steps 128
fmin = 20
fmax = sr // 2
n_mels = 128
n_fft = n_mels * 20
samples = sr * duration


def read_audio(path):
    """
    Reads in the audio file and returns
    an array that we can turn into a melspectogram
    """
    y, _ = librosa.core.load(path, sr=44100)
    # trim silence
    if 0 < len(y):  # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y)
    if len(y) > samples:  # long enough
        y = y[0:0 + samples]
    else:  # pad blank
        padding = samples - len(y)
        offset = padding // 2
        y = np.pad(y, (offset, samples - len(y) - offset), 'constant')
    return y


def audio_to_melspectrogram(audio):
    """
    Convert to melspectrogram after audio is read in
    """
    spectrogram = librosa.feature.melspectrogram(audio,
                                                 sr=sr,
                                                 n_mels=n_mels,
                                                 hop_length=hop_length,
                                                 n_fft=n_fft,
                                                 fmin=fmin,
                                                 fmax=fmax)
    return librosa.power_to_db(spectrogram).astype(np.float32)


def read_as_melspectrogram(path):
    """
    Convert audio into a melspectrogram
    so we can use machine learning
    """
    mels = audio_to_melspectrogram(read_audio(path))
    return mels


def convert_wav_to_image(df):
    output = []
    for _, row in df.iterrows():
        x = read_as_melspectrogram(str(row['path']))
        output.append(x.transpose())
    return output


def normalize(img):
    """
    Normalizes an array
    (subtract mean and divide by standard deviation)
    """
    eps = 0.001
    if np.std(img) != 0:
        img = (img - np.mean(img)) / np.std(img)
    else:
        img = (img - np.mean(img)) / eps
    return img


def normalize_dataset(x):
    """
    Normalizes list of arrays
    (subtract mean and divide by standard deviation)
    """
    normalized_dataset = []
    for img in x:
        normalized = normalize(img)
        normalized_dataset.append(normalized)
    return normalized_dataset
