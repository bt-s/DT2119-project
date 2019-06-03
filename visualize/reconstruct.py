'''
Reconstuct wav file from Mel Spectrogram
'''

import numpy as np
from scipy.io import wavfile
import librosa
import os
import scipy
import scipy.optimize as spopt
import sys
import matplotlib.pyplot as plt


global target
target = None

label_map = {"cel" : 0, "cla" : 1, "flu" : 2, "gac" : 3, "gel" : 4, "org" : 5,
        "pia" : 6, "sax" : 7, "tru" : 8, "vio" : 9, "voi" : 10}


if __name__ == "__main__":
    
    # load intital values to reconstruct
    targets = np.load("opt_one_per_instr.npy")

    # each instrument
    for key, value in label_map.items():
        print(key,")")

        # save initial value
        target = targets[value,:,:]
        target = np.reshape(target, (128,43))
        target = np.exp(target)
        print("inverting")
        
        # find inverse of mel spectrogram
        recon = librosa.feature.mel_to_audio(target, sr=22050, n_fft=2048, hop_length=512, win_length=1024, center=True)

        print("inverted")
        fs = 22050
        out_f = str(key) + '.wav'

        # copy for 10 seconds
        res = 100 * np.tile(recon, 10)

        # set True to plot the pulse wave
        # for the first second
        if False:
            plt.plot(res[:43*1024])
            plt.show()

        res.astype(np.dtype('int16'))

        
        # filter to apply
        from scipy.signal import butter, lfilter, freqz


        def butter_lowpass(cutoff, fs, order=5):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return b, a

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter_lowpass(cutoff, fs, order=order)
            y = lfilter(b, a, data)
            return y

        
        def butter_bandpass(lowcut, highcut, fs, order=5):
            nyq = 0.5 * fs
            low = lowcut / nyq
            high = highcut / nyq
            b, a = butter(order, [low, high], btype='band')
            return b, a

        def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
            b, a = butter_bandpass(lowcut, highcut, fs, order=order)
            y = lfilter(b, a, data)
            return y

        # apply filter and then save into wav file
        res = butter_bandpass_filter(res,300,3000 ,fs)
        wavfile.write(out_f, fs,res)

