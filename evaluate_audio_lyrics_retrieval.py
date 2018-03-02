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
import os.path
import subprocess
import sys

import argparse
import random

import numpy as np
import numpy.random

import tensorflow as tf

import keras.backend.tensorflow_backend

import audio_lyrics_model
import utils

argparser = argparse.ArgumentParser(prog='evaluate_audio_lyrics_retrieval', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('audio')
argparser.add_argument('--lyrics', default='datasets/audio_lyrics/lyrics/features/doc2vec')
argparser.add_argument('--folds', default='datasets/audio_lyrics/folds')
argparser.add_argument('--train_batch', type=int, default=1000)
argparser.add_argument('--test_batch', type=int, default=1000)
argparser.add_argument('--epochs', type=int, default=500)

args = argparser.parse_args()

audio_dir = args.audio
lyrics_dir = args.lyrics
folds_dir = args.folds
train_batch_size = args.train_batch
test_batch_size = args.test_batch
assert os.path.exists(audio_dir), "audio_directory not found"
assert os.path.exists(lyrics_dir), "lyrics_directory not found"
assert os.path.exists(folds_dir), "folds_directory not found"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
random.seed(2017)
numpy.random.seed(2017)
tf.set_random_seed(2017)
os.environ['PYTHONHASHSEED'] = '2017'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
keras.backend.tensorflow_backend.set_session(tf.Session(config=config))

# audio parameters
sample = list(os.listdir(audio_dir))[0]
sample = np.load(audio_dir + '/' + sample)
audio_features = sample.shape[-1]

# lyrics parameters
sample = list(os.listdir(lyrics_dir))[0]
sample = np.load(lyrics_dir + '/' + sample)
lyrics_features = sample.shape[-1]

# learning parameters
audio_units = [512, 256, 128, 64]
lyrics_units = [512, 256, 128, 64]
emb_dims = 32
epochs = args.epochs

# evaluation parameters
folds = len(os.listdir(folds_dir))
runs = 1

# audio_data and lyrics_data are dictionaries mapping strings to numpy arrays
audio_data = utils.read_dataset(audio_dir)
lyrics_data = utils.read_dataset(lyrics_dir)
audio_dir = os.path.basename(audio_dir)

instance_mrr, class_mrr = numpy.zeros((2, 2)), numpy.zeros((2, 2))
for run in range(runs):
  model, i_mrr, c_mrr, c_map = None, None, None, None
  name_format = ['{}' for _ in audio_units]
  name_format = '-'.join(name_format)
  audio_format = name_format.format(*audio_units)
  name_format = ['{}' for _ in lyrics_units]
  name_format = '-'.join(name_format)
  lyrics_format = name_format.format(*lyrics_units)
  model_path = 'models/audio_lyrics_{}_embs-{}_auni-{}_luni-{}'.format(audio_dir, emb_dims, audio_format, lyrics_format)
  model = audio_lyrics_model.audio_lyrics_model(emb_dims, [audio_features, lyrics_features], [audio_units, lyrics_units])

  embeddings = model.cross_validate([audio_data, lyrics_data], epochs, folds, folds_dir, (train_batch_size, test_batch_size), model_path)

  labels = utils.prepare_class_labels(audio_data, folds, folds_dir)
  classes = len(numpy.unique(labels[0][0]))

  i_mrr, c_mrr, _ = utils.evaluate_retrieval(embeddings[0], embeddings[1], classes)

  instance_mrr += i_mrr
  class_mrr += c_mrr

instance_mrr /= runs
class_mrr /= runs
print('audio_units: {}     lyrics_units: {}     emb_dims: {: >3}     epochs: {}     test i-MRR: {}     train i-MRR: {}     test c-MRR: {}     train c-MRR: {}'.format(audio_units, lyrics_units, emb_dims, epochs, instance_mrr[1,:], instance_mrr[0,:], class_mrr[1,:], class_mrr[0,:]))
