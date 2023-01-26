# lasse-py
Python code used at LASSE / UFPA projects concerning signal processing, machine learning, digital communications, etc.

## Installation

Make sure your Python environment has all dependencies using:

conda install --file requirements.txt

If you do not have an environment yet, it is a good idea to follow the hints at https://github.com/lasseufpa/python_template and install one.

PS: To update the requirements.txt file we use

pip install pipreqs

And from the project folder:

pipreqs .

(which will parse the files and save a new requirements.txt. You may move it if the file already exists)

## Packages we adopt

https://github.com/librosa/librosa - audio and music analysis

https://github.com/keunwoochoi/kapre - for processing audio in GPU when using Keras

https://docs.scipy.org/doc/scipy/reference/signal.html - signal processing

https://github.com/crflynn/stochastic - stochastic (random) processes

### Regarding audio I/O

We need to evaluate code to record and play sound:

https://github.com/spatialaudio/python-sounddevice/ - bindings for the PortAudio library and a few convenience functions to play and record NumPy arrays containing audio signals

https://github.com/PortAudio/portaudio - audio I/O library in C that runs on Windows, Mac and Linux

### Regarding telecom

We need to evaluate taking in account a good class definition for signals to be used in PHY simulations:

https://github.com/veeresht/CommPy/network/dependencies

https://github.com/rwnobrega/komm/network/dependencies

https://github.com/darcamo/pyphysim/blob/master/notebooks/TDL_and_OFDM.ipynb

https://github.com/MeowLucian/Multipath_Simulation

https://github.com/kirlf/ModulationPy

## Guidelines for developers

https://guicommits.com/organize-python-code-like-a-pro/

### Packages we can extract code from

akmimo

https://github.com/aldebaro/dsp-class-ufpa/blob/main/stochastic_proc.ipynb

