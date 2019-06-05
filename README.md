# Project on Instrument Recognition in Polyphonic Music by Using Convolutional Neural Network
An endeavour for the *Speech and Speaker Recognition: DT2119* course, 2019 at KTH Royal Institute of Technology.

## Authors: Pietro Alovisi, Romain Deffayet and Bas Straathof

## Project file structure

- `cnn.py`\
Contains the architecture and training and testing methods for the Convolutional Neural Network used for instrument recognition.

- `extract_feature.py`\
Script to extract features from the raw audio files.

- `metrics.py`\
Script to obtain performance metrics.

- `plots.py`\
Script to plot performance metrics.

- `predictions.py`\
Script to make predictions using the CNN model.

- `split_dataset.py`\
Script to split the IRMAS dataset into a training, validation and test set.

- `/visualization`\
Folder for visualization of the learned audio patterns by the CNN.

- `*.wav` Audio representation of the learned patterns per instrument.
- `reconstruct.py` Script to reconstructd .wav fifles from Mel spectograms.
- `visualize.py` Script for learned pattern visualization.
- `visualize_no_backprop.py` Script for learned pattern visualization without back propagation.

