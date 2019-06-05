#!/usr/bin/python3

"""visualize.py: Script to visualize the learned patterns in the audio data.

Part of a project for the 2019 DT2119 Speech and Speaker Recognition course at
KTH Royal Institute of Technology"""

__author__ = "Pietro Alovisi, Romain Deffayet & Bas Straathof"


import numpy as np
from scipy.io import wavfile
import librosa
import scipy
import scipy.optimize as spopt
from keras.models import load_model

model = load_model("results/with_batch_norm/model.h5")

label_map = {"cel" : 0, "cla" : 1, "flu" : 2, "gac" : 3, "gel" : 4, "org" : 5,
        "pia" : 6, "sax" : 7, "tru" : 8, "vio" : 9, "voi" : 10}


def extract_mfcc(audio_single, seconds=1):
    """Extracts the mel-spectogram from the input file
    Args:
        filename (str): the name of the input file
        seconds  (int): lenght of audio excerpt in seconds

    Returns:
        features (dict): contains the filename, the mono audio signal, the STFT,
                         the mspec and the labels
    """
    
    # Normalize by dividing the time-domain signal with its maximum value
    audio_single = audio_single /np.max(np.abs(audio_single))

    # Remove single dimensions
    audio_single = np.squeeze(audio_single)

    audio = np.zeros(21*1024)

    for i in range(21):
        audio[i*1024 : (1+i)*1024] = audio_single 

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
        spec_list.append(ln_mel_spec[:, idx : (idx + seg_dur)])

    mspecs = np.expand_dims(np.array(spec_list), axis=1)
    return mspecs


def test_on_network(x, instrument=3):
    """Test x on the imput and return the loss function."""
    inp = extract_mfcc(x)
    global model 
    inp = np.reshape(inp, (1,128,43,1))
    pred = model.predict(inp)

    pred = scipy.special.softmax(pred)

    return 1 - pred[0, instrument]


def main():
    # Start from a random input
    x = np.random.normal(size=1024)
    
    # Minimize the loss 
    result = spopt.minimize(test_on_network, x)

    # Save the result
    np.save("guitar",result)


if __name__=="__main__":
    main()

