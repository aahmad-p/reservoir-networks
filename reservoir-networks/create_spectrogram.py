import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import numpy as np
import csv


# Extract time samples from wavfile (take single channel)
sample_rate, samples = wavfile.read("data\\piano2_dual.wav")
samples = samples[:,0]


# Plot time series
plt.figure(0)
plt.plot(np.arange(1,len(samples)+1)/sample_rate,samples)
plt.title("Wav file samples")
plt.xlabel("Time [s]")


# Create spectrogram
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate, scaling='density')
print('max', np.max(spectrogram))


plt.figure(1)
plt.pcolormesh(times, frequencies, np.log10(spectrogram))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Log10(spectrogram)')
plt.colorbar()
plt.show()

#plt.figure(2)
#plt.specgram(samples, Fs=sample_rate)
#plt.show()


# Write spectrogram with time samples for processing
with open("data\\spectrogram.csv","w+") as my_csv:
    out_data = np.append(np.reshape(times,(1,-1)),spectrogram,axis=0)
    csvWriter = csv.writer(my_csv,delimiter=',')
    csvWriter.writerows(out_data)