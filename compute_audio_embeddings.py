#!/usr/bin/env python

"""
(C) Copyright 2017-2018 (see AUTHORS file)
Spoken Language Systems Lab, INESC ID, IST/Universidade de Lisboa

This file is part of PEACCA.

PEACCA is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 2 of the License, or (at your
option) any later version.

PEACCA is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307
USA
"""

import os
import subprocess
import sys

import argparse

import numpy

import tensorflow

import keras.backend.tensorflow_backend

import audio_eeg_model
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tensorflow.Session(config=config))

argparser = argparse.ArgumentParser(prog='compute_audio_embeddings', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('--audio', default='datasets/audio_eeg/audio/sources')
argparser.add_argument('--eeg', default='datasets/audio_eeg/eeg/preprocessed')
argparser.add_argument('--folds', default='datasets/audio_eeg/folds')
argparser.add_argument('--train_batch', type=int, default=102)
argparser.add_argument('--test_batch', type=int, default=102)
argparser.add_argument('--input', default='datasets/audio_lyrics/audio/sources')
argparser.add_argument('--output', default='datasets/audio_lyrics/audio/features/embeddings')
argparser.add_argument('--input_batch', type=int, default=103)
argparser.add_argument('--epochs', type=int, default=20)
argparser.add_argument('--ids', action='store_true')

args = argparser.parse_args()

audio_dir = args.audio
eeg_dir = args.eeg
folds_dir = args.folds
train_batch_size = args.train_batch
test_batch_size = args.test_batch
ids = args.ids
input_dir = args.input
output_dir = args.output
input_batch_size = args.input_batch
assert os.path.exists(audio_dir), "audio directory not found"
assert os.path.exists(eeg_dir), "eeg directory not found"
assert os.path.exists(folds_dir), "folds directory not found"
assert input_dir == None or os.path.exists(input_dir), "input directory not found"
assert output_dir == None or os.path.exists(output_dir), "output directory not found"

# audio parameters
audio_chann = 1

# eeg parameters
sample = list(os.listdir(eeg_dir))[0]
sample = numpy.load(eeg_dir + '/' + sample)
eeg_chann = sample.shape[1]

# network parameters
emb_dims = 128
epochs = args.epochs

# evaluation parameters
p1 = subprocess.Popen('ls {}'.format(folds_dir), shell=True, stdout=subprocess.PIPE)
p2 = subprocess.Popen('wc -l', shell=True, stdin=p1.stdout, stdout=subprocess.PIPE)
folds = int(p2.communicate()[0])
runs = 5

# data are dictionaries mapping string ids to numpy arrays
audio_data = utils.read_audio_dataset(audio_dir, int(1.5 * 22050), pad=True)
eeg_data = utils.read_eeg_dataset(eeg_dir, int(1.5 * 250), pad=True, scale=False)
input_data = None
if input_dir is not None:
  input_data = utils.read_audio_dataset(input_dir, int(1.5 * 22050), pad=False)

# delete some chunks so that dataset size is dividable by batch size
del audio_data['150.001.000'], audio_data['186.001.997'], audio_data['133.001.002'], audio_data['124.001.012'], audio_data['330.001.005'], audio_data['194.001.014'] # negative
del eeg_data['150.001.000'], eeg_data['186.001.997'], eeg_data['133.001.002'], eeg_data['124.001.012'], eeg_data['330.001.005'], eeg_data['194.001.014'] # negative
del audio_data['139.000.001'], audio_data['162.000.002'], audio_data['207.000.014'], eeg_data['139.000.001'], eeg_data['162.000.002'], eeg_data['207.000.014'] # positive

for run in range(runs):
  for fold in range(folds):
    model = audio_eeg_model.audio_eeg_model(emb_dims)

    model_path = 'models/audio_eeg_run-{}'.format(run)
    fold_path = '{0}/{1:0>2}.txt'.format(folds_dir, fold)

    model._build_model()
    model._validate_fold([audio_data, eeg_data], epochs, (train_batch_size, test_batch_size), ids, fold, fold_path, model_path)

    embeddings, deep_embeddings = model.embed_audio(input_data, input_batch_size)

    del model

    # compute audio_lyrics data mean embeddings
    data = {}
    deep_data = {}
    for c in range(20):
      for i in range(500):
        data['{:0>3}.{:0>2}'.format(i, c)] = []
        deep_data['{:0>3}.{:0>2}'.format(i, c)] = []
    for idx, key in enumerate(sorted(list(input_data.keys()))):
      data[key[4:]] += [numpy.expand_dims(embeddings[idx,:], -1)]
      deep_data[key[4:]] += [numpy.expand_dims(deep_embeddings[idx,:], -1)]
    for key in data.keys():
      numpy.save(output_dir + '/dcca_embeddings/run-{}_fold-{}/'.format(run, fold) + key, numpy.squeeze(numpy.mean(data[key], axis=0)))
      numpy.save(output_dir + '/deep_embeddings/run-{}_fold-{}/'.format(run, fold) + key, numpy.squeeze(numpy.mean(deep_data[key], axis=0)))
