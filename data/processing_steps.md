# Data Processing
1. Obtain .WAV files
2. Using Audacity's "Noise Reduction" tool, remove background noise (machinery, breathing, etc.)
3. Run `clipping.py` to remove blank space at the beginning and end of the audio files. Some adjustment may be required.
4. Read the .WAV file into a numpy array using `scipy.io.wavfile.read`
5. Take the Fast Fourier Transform (FFT) of the .WAV array using  `numpy.fft.fft`
6. The score for the given .WAV file is calculated as the standard deviation (STD) of the FFT. This is calculated using `numpy.std`
7. Save the .WAV filepath and its score in a CSV file using pandas