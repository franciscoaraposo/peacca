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
import random
import itertools
import numpy
import scipy.io.wavfile
import pywt


def preprocess_audio(data, samples):
  # convert to mono
  if len(data.shape) > 1:
    data = numpy.mean(data, 1)
  # truncate last samples
  if data.shape[0] > samples:
    data = numpy.resize(data, (samples))
  # zero pad
  if data.shape[0] < samples:
    data = numpy.append(data, numpy.zeros((samples - data.shape[0]), dtype=data.dtype))
  # constant stream
  if numpy.max(data) == numpy.min(data):
    return numpy.zeros(data.shape)
  # scale to [-1, 1]
  tmp = data - numpy.min(data)
  data = (tmp / numpy.max(tmp) - 0.5) * 2
  return data


# denoises EEG data, assuming time samples are rows and channels are columns
def wavelet_semblance_denoising(data, threshold=0.999, wavelet='db8', mode='symmetric'):
  channels = data.shape[1]
  dwts = [None for _ in range(channels)]
  for channel in range(channels):
    dwts[channel] = pywt.wavedec(data[:,channel], wavelet, mode=mode, axis=-1)

  scales = len(dwts[0])
  for scale in range(scales):
    steps = dwts[0][scale].shape[0]
    for shift in range(steps):
      mrl = wavelet_mean_resultant_length(dwts, scale, shift)
      if mrl < threshold:
        for channel in range(channels):
          dwts[channel][scale][shift] = 0

  filtered = numpy.full(data.shape, numpy.nan)
  for channel in range(channels):
    filtered[:,channel] = pywt.waverec(dwts[channel], wavelet, mode=mode, axis=-1)[:data.shape[0]]

  return filtered


# helper function for wavelet_semblance_denoising
def wavelet_mean_resultant_length(dwts, scale, shift):
  channels, real, imaginary, absolute = len(dwts), 0, 0, 0
  for channel in range(channels):
    real += numpy.real(dwts[channel][scale][shift])
    imaginary += numpy.imag(dwts[channel][scale][shift])
    absolute += numpy.absolute(dwts[channel][scale][shift])
  return numpy.sqrt(real * real + imaginary * imaginary) / absolute


# removes artifacts from EEG data, assuming time samples are rows and channels are columns
def wavelet_artifact_removal(data, threshold=5, wavelet='db8', mode='symmetric'):
  channels = data.shape[1]
  dwts = [None for _ in range(channels)]
  for channel in range(channels):
    dwts[channel] = pywt.wavedec(data[:,channel], wavelet, mode=mode, axis=-1)

  for channel in range(channels):
    scales = len(dwts[channel])
    for scale in range(scales):
      stdev = numpy.std(dwts[channel][scale])
      mean = numpy.mean(dwts[channel][scale])
      dwts[channel][scale][numpy.where(dwts[channel][scale] > mean + threshold * stdev)] = 0
      dwts[channel][scale][numpy.where(dwts[channel][scale] < mean - threshold * stdev)] = 0

  filtered = numpy.full(data.shape, numpy.nan)
  for channel in range(channels):
    filtered[:,channel] = pywt.waverec(dwts[channel], wavelet, mode=mode, axis=-1)[:data.shape[0]]

  return filtered


# computes bi-modal retrieval evaluation metrics, dist is a squared distance matrix
def compute_rr_ap(dist, classes):
  if dist.shape[0] != dist.shape[1]:
    raise ValueError('distance matrix is not squared')
  dataset_size = dist.shape[0]
  instance_rr, class_rr, class_ap = numpy.zeros((dataset_size, 2)), numpy.zeros((dataset_size, 2)), numpy.zeros((dataset_size, 2))
  for instance_idx in range(dataset_size):
    y_ranks = numpy.argsort(dist[instance_idx,:])
    x_ranks = numpy.argsort(dist[:,instance_idx])
    instance_rr[instance_idx,0] = 1.0 / (numpy.argwhere(y_ranks == instance_idx)[0,0] + 1)
    instance_rr[instance_idx,1] = 1.0 / (numpy.argwhere(x_ranks == instance_idx)[0,0] + 1)
    best_y_rank, best_x_rank = dataset_size, dataset_size
    relevant_y_ranks, relevant_x_ranks = numpy.zeros(0), numpy.zeros(0)
    for class_idx in numpy.arange(instance_idx % classes, dataset_size, classes):
      y_rank = numpy.argwhere(y_ranks == class_idx)[0,0]
      x_rank = numpy.argwhere(x_ranks == class_idx)[0,0]
      if y_rank < best_y_rank:
        best_y_rank = y_rank
      if x_rank < best_x_rank:
        best_x_rank = x_rank
      relevant_y_ranks = numpy.append(relevant_y_ranks, y_rank)
      relevant_x_ranks = numpy.append(relevant_x_ranks, x_rank)
    class_rr[instance_idx,0] = 1.0 / (best_y_rank + 1)
    class_rr[instance_idx,1] = 1.0 / (best_x_rank + 1)
    relevant_y_ranks = numpy.sort(relevant_y_ranks + 1)
    relevant_x_ranks = numpy.sort(relevant_x_ranks + 1)
    class_ap[instance_idx,0] = numpy.sum(numpy.arange(1, relevant_y_ranks.shape[0] + 1) / relevant_y_ranks) / relevant_y_ranks.shape[0]
    class_ap[instance_idx,1] = numpy.sum(numpy.arange(1, relevant_x_ranks.shape[0] + 1) / relevant_x_ranks) / relevant_x_ranks.shape[0]
  return instance_rr, class_rr, class_ap


# reads an audio segments dataset (wav files in path) as (samples, 1) numpy arrays
def read_audio_dataset(path, samples, pad=True):
  data = {}
  for filename in os.listdir(path):
    if filename.endswith('.wav'):
      _, audio = scipy.io.wavfile.read(path + '/' + filename)
      audio = audio.astype(float)
      audio = numpy.expand_dims(preprocess_audio(audio, len(audio)), -1)
      for idx, sample in enumerate(numpy.arange(0, len(audio), samples)):
        segment_id = '{:0>3}'.format(idx)
        padded = numpy.zeros((samples, audio.shape[1]))
        size = len(audio[sample:sample+samples,:])
        padded[:size,:] = audio[sample:sample+samples,:]
        data[segment_id+'.'+filename[:-4]] = padded
        if not pad and size != samples:
          del data[segment_id+'.'+filename[:-4]]
  return data


# reads an eeg segments dataset (npy files in path) as (samples, channels) numpy arrays
def read_eeg_dataset(path, samples, pad=True, scale=False):
  data = {}
  for filename in os.listdir(path):
    if filename.endswith('.npy'):
      eeg = numpy.load(path + '/' + filename)
      for idx, sample in enumerate(numpy.arange(0, len(eeg), samples)):
        segment_id = '{:0>3}'.format(idx)
        padded = numpy.zeros((samples, eeg.shape[1]))
        size = len(eeg[sample:sample+samples,:])
        padded[:size,:] = eeg[sample:sample+samples,:]
        if scale:
          tmp = padded - numpy.min(padded, axis=0)
          padded = (tmp / numpy.max(tmp, axis=0) - 0.5) * 2
        data[segment_id+'.'+filename[:-4]] = padded
        if not pad and size != samples:
          del data[segment_id+'.'+filename[:-4]]
  return data


# reads a dataset (npy files in path) to a dictionary mapping the filename (excluding the extension) to a numpy array
def read_dataset(path):
  data = {}
  for filename in os.listdir(path):
    if filename.endswith('.npy'):
      data[filename[:-4]] = numpy.load(path + '/' + filename)
  return data


# performs cosine distance-based retrieval evaluation
# inputs are the paired fold lists of feature matrices (samples x features) sorted according to classes (instance_id.class_id sorting)
# classes is the number of classes
def cosine_evaluation(inputs0, inputs1, classes):
  # initialize metrics arrays
  folds = len(inputs0)
  instance_mrr, class_mrr, class_map = numpy.zeros((folds, 2)), numpy.zeros((folds, 2)), numpy.zeros((folds, 2))

  # evaluate
  for fold in range(folds):
    x0 = inputs0[fold]
    x1 = inputs1[fold]

    x0 /= numpy.expand_dims(numpy.sqrt(numpy.sum(x0 ** 2, axis=1)), axis=-1)
    x1 /= numpy.expand_dims(numpy.sqrt(numpy.sum(x1 ** 2, axis=1)), axis=-1)
    distance_matrix = 1 - numpy.dot(x0, x1.T)

    instance_rr, class_rr, class_ap = compute_rr_ap(distance_matrix, classes)
    instance_mrr[fold,0], class_mrr[fold,0], class_map[fold,0] = numpy.mean(instance_rr[:,0]), numpy.mean(class_rr[:,0]), numpy.mean(class_ap[:,0])
    instance_mrr[fold,1], class_mrr[fold,1], class_map[fold,1] = numpy.mean(instance_rr[:,1]), numpy.mean(class_rr[:,1]), numpy.mean(class_ap[:,1])

  return numpy.mean(instance_mrr, axis=0), numpy.mean(class_mrr, axis=0), numpy.mean(class_map, axis=0)


# loads a batch from dictionary according to ids
def load_batch(data, batch_ids):
  sample = data[list(data.keys())[0]]
  batch = numpy.full((len(batch_ids),) + sample.shape, fill_value=numpy.nan)
  for idx, id in enumerate(batch_ids):
    batch[idx,:] = data[id]
  return batch


# loads a fold ids list
def load_fold_ids(fold_path):
  fold_file = open(fold_path)
  return fold_file.read().splitlines()


# saves embeddings to disk
# embeddings is a set list of fold lists of embeddings sorted according to prepare_simple_multiview_fold
# view_data is the original data dictionary
def save_embeddings(embeddings, view_data, folds_path, embs_path):
  folds = len(embeddings[0])
  for fold in range(folds):
    fold_path = embs_path + '/{:0>2}'.format(fold)
    if not os.path.exists(fold_path):
      os.mkdir(fold_path)
    fold_ids = load_fold_ids('{0}/{1:0>2}.txt'.format(folds_path, fold))
    train_idx, test_idx = 0, 0
    for key in sorted(view_data.keys()):
      if key in fold_ids:
        numpy.save(fold_path + '/' + key + '.npy', embeddings[1][fold][test_idx,:])
        test_idx += 1
      else:
        numpy.save(fold_path + '/' + key + '.npy', embeddings[0][fold][train_idx,:])
        train_idx += 1


# prepares fold labels from class data
# instances ids are formated as instance_id.class_id (where class id is 2 digits)
def prepare_class_labels(view_data, folds, folds_path):
  dataset_size = len(view_data)
  test_size = dataset_size // folds
  train_size = dataset_size - test_size
  labels = [None] * 2
  labels[0] = [None] * folds
  labels[1] = [None] * folds
  for fold in range(folds):
    fold_ids = load_fold_ids('{0}/{1:0>2}.txt'.format(folds_path, fold))
    labels[0][fold] = numpy.full(train_size, fill_value=-1, dtype=int)
    labels[1][fold] = numpy.full(test_size, fill_value=-1, dtype=int)
    test_idx, train_idx = 0, 0
    for key in sorted(view_data.keys()):
      if key in fold_ids:
        labels[1][fold][test_idx] = int(key[-2:])
        test_idx += 1
      else:
        labels[0][fold][train_idx] = int(key[-2:])
        train_idx += 1
  return labels


# prepares fold data for a list of views
# fold_ids is a list of fold test string ids
# view_data is a list of dictionaries mapping string ids to numpy array data instances
# if disk is True, lists of ids are returned instead
def prepare_simple_multiview_fold(fold_ids, view_data, ids):
  dataset_size, test_size = len(view_data[0]), len(fold_ids)
  train_size = dataset_size - test_size

  views = len(view_data)
  view_test, view_train = [None] * views, [None] * views
  for view_idx in range(views):
    if ids:
      view_test[view_idx] = [None] * test_size
      view_train[view_idx] = [None] * train_size
    else:
      data = view_data[view_idx]
      sample = data[list(data.keys())[0]]
      view_test[view_idx] = numpy.full((test_size,) + sample.shape, fill_value=numpy.nan)
      view_train[view_idx] = numpy.full((train_size,) + sample.shape, fill_value=numpy.nan)

  test_idx, train_idx = 0, 0
  for key in sorted(view_data[0].keys()):
    if key in fold_ids:
      for view_idx in range(views):
        if ids:
          view_test[view_idx][test_idx] = key
        else:
          view_test[view_idx][test_idx,:] = view_data[view_idx][key]
      test_idx += 1
    else:
      for view_idx in range(views):
        if ids:
          view_train[view_idx][train_idx] = key
        else:
          view_train[view_idx][train_idx,:] = view_data[view_idx][key]
      train_idx += 1

  return view_train, view_test


# evaluates pair of class-based feature sets for cosine distance-based retrieval
# views_data is a set list of fold lists of samples
def evaluate_retrieval(view0_data, view1_data, classes):
  instance_mrr, class_mrr, class_map = numpy.zeros((len(view0_data), 2)), numpy.zeros((len(view0_data), 2)), numpy.zeros((len(view0_data), 2))
  folds = len(view0_data[0])

  for set_idx in range(len(view0_data)):
    instance_mrr[set_idx,:], class_mrr[set_idx,:], class_map[set_idx,:] = cosine_evaluation(view0_data[set_idx], view1_data[set_idx], classes)

  return instance_mrr, class_mrr, class_map
