import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile   
import librosa
import os
import sys
from lab1 import *

Folders = ['IRMAS-TestingData-Part3/Part3/']

FS = 44100 # sampling rate
WINLEN =  882# 25 msec
WINSRIDE = 441 # 10 msec


label_map = {
    "cel" : 0,
    "cla" : 1,
    "flu" : 2,    
    "gac" : 3,
    "gel" : 4,
    "org" : 5,
    "pia" : 6,
    "sax" : 7,
    "tru" : 8,
    "vio" : 9,
    "voi" : 10
}

def progress_bar(count, total, suffix=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()  # As suggested by Rom Ruben


def extract_from_file(filename, preemp=False, window=False, fft_len=512, seconds=1):
    # Read the data
    fs, data = wavfile.read(filename)

    # Take mean
    mono = np.mean(data, axis=1) 
    # Downsample
    mono = mono[np.arange(0, mono.size, 2)]
    # Normalize
    mono /= np.max(np.abs(mono))
    # Remove single dimensions
    mono = np.squeeze(mono)

    # Compute Short Time Fourier Transform
    stft = np.abs(librosa.stft(mono, win_length=1024, hop_length=512, center=True))
    # Convert to Mel Spectogram
    mel_spec = librosa.feature.melspectrogram(S=stft, sr=22050, n_mels=128)
    # Take the natural logarithm
    ln_mel_spec = np.log(mel_spec + np.finfo(float).eps)

    # Segementation of spectogram
    seg_dur = 43 * seconds
    spec_list = []
    for idx in range(0, ln_mel_spec.shape[1] - seg_dur + 1, seg_dur):
        spec_list.append(ln_mel_spec[:, idx:(idx+seg_dur)])
    #print('Number of spectrograms:', len(spec_list))
    X = np.expand_dims(np.array(spec_list), axis=1)
    #print(X.shape)

    # Extract feature on both the channels
    filename_cut = filename[:-4]

    res = {}

    res["filename"] = filename_cut
    res["mono"]     = mono 
    res["stft"]     = stft
    res["mspec"]    = X
    res["labels"]   = np.zeros([11])

    with open(filename_cut + '.txt', 'r') as fp:  
        lines = fp.readlines()        
        for l in lines:
            res["labels"][label_map[l[:3]]] = 1

    return res


def main():
    for folder in Folders:
        print("Entering folder ", folder)

        features = []
        for root, dirs, files in os.walk(folder):
            total_files = len(files)/2

            count = 0

            for file in files[:100]:
                if file.endswith('.wav'):
                    count += 1
                    progress_bar(count, total_files, suffix='')
                    feat = extract_from_file(folder+file)
                    features.append(feat)

        np.save("%s_features" % folder[:23], features)


if __name__ == "__main__":
    main()

