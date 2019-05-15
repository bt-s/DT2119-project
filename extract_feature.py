import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile   
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


def main():

    for folder in Folders:
        print("Entering folder ", folder)

        for root, dirs, files in os.walk(folder):
            total_files = len(files)/2

            count = 0

            for file in files:
                if file.endswith('.wav'):
                    count += 1
                    progress_bar(count, total_files, suffix='')
                    feat = extract_from_file(folder+file)




def extract_from_file(filename, preemp=False, window=False, fft_len=512):

    #print(filename)
    # read the data
    fs, data = wavfile.read(filename)
    print(fs)

    # extract feature on both the channels

    filename_cut = filename[:-4]

    res = {}

    res["filename"] = filename_cut

    res["fft"] = np.zeros([512,2])
    res["mspec"] = np.zeros([40,2])
    res["ceps"] = np.zeros([13,2])
    res["lmfcc"] = np.zeros([13,2])

    for channel in [0,1]:
        channeldata = data[:,channel].T
        channeldata.shape = (1, channeldata.size)

        if preemp:
            preemph = preemp(channeldata)
        else:
            preemph = channeldata
        if window:
            windowed = windowing(preemph)
        else:
            windowed = preemph
        spec = powerSpectrum(windowed, fft_len)
        mspec = logMelSpectrum(spec, FS)
        ceps = cepstrum(mspec, 13)
        lift = lifter(ceps, 22)

        res["fft"][:,channel] = spec
        res["mspec"][:,channel] = mspec
        res["ceps"][:,channel] =  ceps
        res["lmfcc"][:,channel] = lift


    # read labels
    res["labels"] = np.zeros([11])

    with open(filename_cut + '.txt', 'r')   as fp:  
        lines = fp.readlines()        
        for l in lines:
            res["labels"][label_map[l[:3]]] = 1
    print(res["labels"])            

    return res

if __name__ == "__main__":

    main()
