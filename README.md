# lasse-py

`lasse-py` is a Python library used at LASSE / UFPA for digital signal processing and quantization research, developed by the LASSE group to support simulation and analysis of modern communication systems.
 audio, dsp, io (input / output), statistics, stochastic (processes), transforms (DCT, Fourier, etc) and util.

`lasse-py` provides a modular set of tools for scalar quantization, bit-level processing, and signal modeling — making it ideal for academic research, prototyping, and 6G experimental platforms.

## Installation for users (not developers)

If you do not have an environment yet, it is a good idea to follow the hints at https://github.com/lasseufpa/python_template and install one.

Go to the folder in which the `lasse-py` Python library was installed.

Make sure your Python environment has all dependencies using conda or similar software. For instance, for conda:

conda install --file requirements.txt

Then install the `lasse-py` library using

``python setup.py install``

## Guidelines for developers

Then `lasse-py` library uses isort, black, etc., and was created using the template at
https://github.com/lasseufpa/python_template.

#### Use PYTHONPATH if you are editing the code

If you installed using
``python setup.py install``
after each code modification, you will have to reinstall using the same command to update the code.

Hence, when changing the code, it is more convenient to set the environment variable PYTHONPATH to find the code.
For instance, when using Windows' cmd:

``set PATH=c:\mylib\lasse-py``

in case the folder lasse is located at c:\mylib\lasse-py\lasse.

### Install tools for commiting code

Before doing git commit, you need to locally check whether your code is compliant, otherwise github may block the commit.

For instance, using conda, you may install
``conda install -c conda-forge isort flake8 click``
and complement with pip:
``pip install pre-commit black``

More information at 
https://guicommits.com/organize-python-code-like-a-pro/

### To update the requirements.txt file 

We use:

``pip install pipreqs``

And from the project folder:

``pipreqs .``

(which will parse the files and save a new requirements.txt. You may move it if the file already exists)

## Extra information and packages we adopt

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

### Packages we can extract code from

akmimo

https://github.com/aldebaro/dsp-class-ufpa/blob/main/stochastic_proc.ipynb

