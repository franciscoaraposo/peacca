# PEACCA

This repository contains the implementation code used in the experiments of the following paper:

Francisco Raposo, David Martins de Matos, Ricardo Ribeiro, Suhua Tang, Yi Yu, "Towards Deep Modeling of Music Semantics using EEG Regularizers", CoRR, vol. arXiv:1712.05197, 2017
https://arxiv.org/pdf/1712.05197.pdf

Please cite the previously mentioned article if you use PEACCA in your research and/or software.

# download and install Anaconda
wget https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh; bash Anaconda3-5.0.1-Linux-x86_64.sh

# create anaconda environment
conda create -n peacca python=3.6.4

# activate environment
source activate peacca

# install tensorflow
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.5.0-cp36-cp36m-linux_x86_64.whl

# install keras, pywt, h5py
pip install keras pywavelets h5py

# filter and cut openbci raw eeg files
for i in 000 001 002 003 004 005 006 007 008 009 011 012 014 015 996 997 998 999; do python filter_and_cut_eeg.py datasets/audio_eeg/eeg/openbci/$i.openbci $i 16 datasets/audio_eeg/audio/info/$i.info datasets/audio_eeg/eeg/sources; done

# preprocess eeg source files
for i in datasets/audio_eeg/eeg/sources/*; do o=$( basename $i ); python preprocess_eeg.py --war 5 --wsd 0.5 $i datasets/audio_eeg/eeg/preprocessed/$o; done

# create models directory
mkdir models

# train audio-eeg model and generate embeddings
python compute_audio_embeddings.py

# evaluate audio-lyrics model (choi embeddings)
python evaluate_audio_lyrics_retrieval.py datasets/audio_lyrics/audio/features/choi

# evaluate audio-lyrics model (spotify features)
python evaluate_audio_lyrics_retrieval.py datasets/audio_lyrics/audio/features/spotify

# evaluate audio-lyrics model (audio-eeg model embeddings)
for run in {0..4}; do for fold in {0..4}; do python evaluate_audio_lyrics_retrieval.py datasets/audio_lyrics/audio/features/embeddings/run-${run}_fold-${fold}; done; done   
