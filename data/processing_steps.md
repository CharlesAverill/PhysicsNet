# Data Processing
1. Obtain .WAV files
2. Using Audacity's "Noise Reduction" tool, remove background noise (machinery, breathing, etc.)
3. Read the .WAV file into a numpy array using `scipy.io.wavfile.read`
4. Take the Fast Fourier Transform (FFT) of the .WAV array using  `numpy.fft.fft`
5. The score for the given .WAV file is calculated as the standard deviation (STD) of the FFT. This is calculated using `numpy.std`
6. Save the .WAV filepath and its score in a CSV file using pandas