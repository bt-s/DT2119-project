#!/usr/bin/python3

"""extract_feature.py: Implementation of the feature extraction pipeline
as presented in: 'Deep convolutional neural networks for predominant
instrument recognition in polyphonic music' by Han et al.

Part of a project for the 2019 DT2119 Speech and Speaker Recognition course at
KTH Royal Institute of Technology"""

__author__ = "Pietro Alovisi, Romain Deffayet & Bas Straathof"


import numpy as np
from scipy.io import wavfile
import librosa
import os
import sys

testing_folders = ['IRMAS-TestingData-Part1/Part1/', 'IRMAS-TestingData-Part2/Part2/',
            'IRMAS-TestingData-Part3/Part3/']
training_folder = 'IRMAS-TrainingData/'

label_map = {"cel" : 0, "cla" : 1, "flu" : 2, "gac" : 3, "gel" : 4, "org" : 5,
        "pia" : 6, "sax" : 7, "tru" : 8, "vio" : 9, "voi" : 10}


def progress_bar(count, total, suffix=''):
    """Prints a progress bar

    Args:
        count  (int): counter of the current step
        total  (int): total number of steps
        suffix (str): to suffix the progress bar
    """
    bar_len = 60
    
    filled_len = int(round(60 * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', suffix))
    sys.stdout.flush()


def extract_from_file(filename, seconds=1):
    """Extracts the mel-spectogram from the input file

    Args:
        filename (str): the name of the input file
        seconds  (int): lenght of audio excerpt in seconds

    Returns:
        features (dict): contains the filename, the mono audio signal, the STFT,
                         the mspec and the labels
    """
    # Read the data
    fs, data = wavfile.read(filename)

    # Convert to a mono signal by taking the mean of the left and right channels
    audio = np.mean(data, axis=1)
    # Downsample from 44,100 Hz to 22,050 Hz
    audio = audio[np.arange(0, audio.size, 2)]
    # Normalize by dividing the time-domain signal with its maximum value
    audio /= np.max(np.abs(audio))
    # Remove single dimensions
    audio = np.squeeze(audio)

    # Compute Short Time Fourier Transform
    stft = np.abs(librosa.stft(audio, win_length=1024, hop_length=512,
        center=True))
    # Convert to Mel Spectogram
    mel_spec = librosa.feature.melspectrogram(S=stft, sr=22050, n_mels=128)
    # Take the natural logarithm of the Mel Spectogram
    ln_mel_spec = np.log(mel_spec + np.finfo(float).eps)

    # Segementation of the spectogram
    seg_dur = 43 * seconds
    spec_list = []
    for idx in range(0, ln_mel_spec.shape[1] - seg_dur + 1, seg_dur):
        spec_list.append(ln_mel_spec[:, idx:(idx+seg_dur)])
    mspecs = np.expand_dims(np.array(spec_list), axis=1)

    features = {}

    features["filename"] = filename[:-4]
    features["mspec"]    = mspecs
    features["labels"]   = np.zeros([11])

    with open(filename[:-4] + '.txt', 'r') as fp:
        lines = fp.readlines()
        for l in lines:
            features["labels"][label_map[l[:3]]] = 1

    return features


def main_testing():
    """Extracts the mel-spectogram from all testing files"""
    for folder in testing_folders:
        print("Entering folder ", folder)

        features = []
        for root, dirs, files in os.walk(folder):
            total_files = len(files)/2

            count = 0
            for file in files:
                if file.endswith('.wav'):
                    count += 1
                    progress_bar(count, total_files, suffix='')
                    feat = extract_from_file(folder+file)
                    features.append(feat)

        np.save("%s_features" % folder[:23], features)


def main_training():
    """Extracts the mel-spectogram from all training files"""
    features = []
    for instrument in label_map.keys() :
        print("Entering folder ", training_folder + instrument)

        for root, dirs, files in os.walk(training_folder + instrument):
            
            total_files = len(files)

            count = 0
            for file in files:
                if file.endswith('.wav'):
                    count += 1
                    progress_bar(count, total_files, suffix='')
                    feat = extract_from_file(training_folder+instrument+ "/" + file, label_map[instrument] )
                    features.append(feat)
                    #print(file)
                    
    np.save(training_folder + "../" + "IRMAS-TrainingData_features", features)


if __name__ == "__main__":
    main_training()
    main_testing()

