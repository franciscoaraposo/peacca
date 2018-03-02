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

import numpy as np
import tensorflow as tf
import keras.callbacks
import keras.layers
import keras.models
import keras.optimizers
import keras.utils
import cca
import utils

def cca_loss(emb_dims):

  def cca_objective(y_true, y_pred):
    reg = 1e-4
    eps = 1e-12
    samples = tf.shape(y_pred)[0]

    audio_emb_dims = emb_dims[0]
    eeg_emb_dims = emb_dims[1]
    audio_embs = tf.transpose(y_pred[:,:audio_emb_dims])
    eeg_embs = tf.transpose(y_pred[:,-eeg_emb_dims:])

    audio_embs_mean = audio_embs - (1.0 / tf.to_float(samples)) * tf.matmul(audio_embs, tf.ones([samples, samples]))
    eeg_embs_mean = eeg_embs - (1.0 / tf.to_float(samples)) * tf.matmul(eeg_embs, tf.ones([samples, samples]))
    audio_embs_cov = (1.0 / (tf.to_float(samples) - 1)) * tf.matmul(audio_embs_mean, tf.transpose(audio_embs_mean)) + reg * tf.eye(audio_emb_dims)
    eeg_embs_cov = (1.0 / (tf.to_float(samples) - 1)) * tf.matmul(eeg_embs_mean, tf.transpose(eeg_embs_mean)) + reg * tf.eye(eeg_emb_dims)
    embs_cross_cov = (1.0 / (tf.to_float(samples) - 1)) * tf.matmul(audio_embs_mean, tf.transpose(eeg_embs_mean))

    # eigen decomposition to find root inverse of covariance
    audio_evals, audio_evecs = tf.self_adjoint_eig(audio_embs_cov)
    eeg_evals, eeg_evecs = tf.self_adjoint_eig(eeg_embs_cov)

    # filter out evals and evecs smaller than eps for stability
    idxs = tf.greater(audio_evals, eps)
    audio_evals = tf.boolean_mask(audio_evals, idxs)
    audio_evecs = tf.transpose(tf.boolean_mask(tf.transpose(audio_evecs), idxs))
    idxs = tf.greater(eeg_evals, eps)
    eeg_evals = tf.boolean_mask(eeg_evals, idxs)
    eeg_evecs = tf.transpose(tf.boolean_mask(tf.transpose(eeg_evecs), idxs))

    audio_embs_cov_root_inv = tf.matmul(tf.matmul(audio_evecs, tf.diag(audio_evals ** -0.5)), tf.transpose(audio_evecs))
    eeg_embs_cov_root_inv = tf.matmul(tf.matmul(eeg_evecs, tf.diag(eeg_evals ** -0.5)), tf.transpose(eeg_evecs))

    t = tf.matmul(tf.matmul(audio_embs_cov_root_inv, embs_cross_cov), eeg_embs_cov_root_inv)

    loss = -tf.sqrt(tf.trace(tf.matmul(tf.transpose(t), t)))

    return loss

  return cca_objective



class batch_generator(keras.utils.Sequence):

  def __init__(self, view_set, batch_size, ids, view_data, shuffle):
    self._view_set, self._batch_size, self._ids, self._view_data, self._shuffle, self._indices = view_set, batch_size, ids, view_data, shuffle, np.arange(0, len(view_set[0]), dtype=int)


  def __len__(self):
    return len(self._view_set[0]) // self._batch_size


  def __getitem__(self, idx):
    if self._shuffle and idx == 0:
      np.random.shuffle(self._indices)
    batch_begin, batch_end = idx * self._batch_size, min((idx + 1) * self._batch_size, len(self._view_set[0]))
    view_batch = [None] * len(self._view_set)
    for view_idx in range(len(self._view_set)):
      view_batch[view_idx] = utils.load_batch(self._view_data[view_idx], [self._view_set[view_idx][instance_idx] for instance_idx in self._indices[batch_begin:batch_end]]) if self._ids else self._view_set[view_idx][self._indices[batch_begin:batch_end],:]
    return view_batch, np.zeros(batch_end - batch_begin)



class audio_eeg_model(object):

  def __init__(self, cca_dims):
    self._model, self._cca_model = None, None
    self._audio_channels, self._eeg_channels = 1, 16
    self._audio_embs, self._eeg_embs = 128, 128
    self._emb_dims, self._view_dims, self._view_units = cca_dims, [self._audio_channels, self._eeg_channels], [[self._audio_embs], [self._eeg_embs]]


  def _build_model(self):
    keras.backend.clear_session()

    x, inputs = [None] * 2, [None] * 2
    activation = 'relu'
    bias = True

    # audio (1.5s at 22050Hz)
    audio_steps = int(1.5 * 22050) # 33075
    x[0] = inputs[0] = keras.layers.Input(shape=(audio_steps, self._audio_channels))
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(128, 3, strides=3, padding='valid', activation=activation, use_bias=bias)(x[0]) # 11025
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(128, 3, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.MaxPooling1D(pool_size=3, strides=None, padding='valid')(x[0]) # 3675
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(256, 3, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.MaxPooling1D(pool_size=3, strides=None, padding='valid')(x[0]) # 1225
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(256, 5, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.MaxPooling1D(pool_size=5, strides=None, padding='valid')(x[0]) # 245
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(512, 5, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.MaxPooling1D(pool_size=5, strides=None, padding='valid')(x[0]) # 49
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(512, 7, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.MaxPooling1D(pool_size=7, strides=None, padding='valid')(x[0]) # 7
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(1024, 7, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.MaxPooling1D(pool_size=7, strides=None, padding='valid')(x[0]) # 1
    x[0] = keras.layers.BatchNormalization()(x[0])
    x[0] = keras.layers.Conv1D(self._audio_embs, 1, strides=1, padding='same', activation=activation, use_bias=bias)(x[0])
    x[0] = keras.layers.Flatten()(x[0])

    # eeg (1.5s at 250Hz)
    eeg_steps = int(1.5 * 250) # 375
    x[1] = inputs[1] = keras.layers.Input(shape=(eeg_steps, self._eeg_channels))
    x[1] = keras.layers.BatchNormalization()(x[1])
    x[1] = keras.layers.Conv1D(128, 3, strides=3, padding='valid', activation=activation, use_bias=bias)(x[1]) # 125
    x[1] = keras.layers.BatchNormalization()(x[1])
    x[1] = keras.layers.Conv1D(256, 5, strides=1, padding='same', activation=activation, use_bias=bias)(x[1])
    x[1] = keras.layers.MaxPooling1D(pool_size=5, strides=None, padding='valid')(x[1]) # 25
    x[1] = keras.layers.BatchNormalization()(x[1])
    x[1] = keras.layers.Conv1D(512, 5, strides=1, padding='same', activation=activation, use_bias=bias)(x[1])
    x[1] = keras.layers.MaxPooling1D(pool_size=5, strides=None, padding='valid')(x[1]) # 5
    x[1] = keras.layers.BatchNormalization()(x[1])
    x[1] = keras.layers.Conv1D(1024, 5, strides=1, padding='same', activation=activation, use_bias=bias)(x[1])
    x[1] = keras.layers.MaxPooling1D(pool_size=5, strides=None, padding='valid')(x[1]) # 1
    x[1] = keras.layers.BatchNormalization()(x[1])
    x[1] = keras.layers.Conv1D(self._eeg_embs, 1, strides=1, padding='same', activation=activation, use_bias=bias)(x[1])
    x[1] = keras.layers.Flatten()(x[1])

    outputs = keras.layers.Concatenate()(x)
    self._model = keras.models.Model(inputs=inputs, outputs=outputs)
#    self._model.compile(loss=cca_loss([self._audio_embs, self._eeg_embs]), optimizer=keras.optimizers.SGD(lr=1e-4))
#    self._model.compile(loss=cca_loss([self._audio_embs, self._eeg_embs]), optimizer=keras.optimizers.RMSprop(lr=1e-3))
    self._model.compile(loss=cca_loss([self._audio_embs, self._eeg_embs]), optimizer=keras.optimizers.Adam(lr=1e-3))


  def _validate_fold(self, view_data, epochs, batch_size, ids, fold, fold_path, model_path):
    view_train, view_test = utils.prepare_simple_multiview_fold(utils.load_fold_ids(fold_path), view_data, ids)

    train_generator = batch_generator(view_train, batch_size[0], ids, view_data, True)
    test_generator = batch_generator(view_test, batch_size[1], ids, view_data, False)
    checkpointer = keras.callbacks.ModelCheckpoint(model_path + '_fold-{}.hdf5'.format(fold), monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')
#    train_generator = batch_generator([sorted(list(view_data[0].keys())), sorted(list(view_data[1].keys()))], batch_size[0], True, view_data, True)
#    checkpointer = keras.callbacks.ModelCheckpoint(model_path + '_fold-{}.hdf5'.format(fold), monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min')

    history = self._model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=epochs, verbose=1, callbacks=[checkpointer], validation_data=test_generator, validation_steps=None, max_queue_size=1, workers=1, use_multiprocessing=False, initial_epoch=0)
#    history = self._model.fit_generator(generator=train_generator, steps_per_epoch=None, epochs=epochs, verbose=1, callbacks=[checkpointer], max_queue_size=1, workers=1, use_multiprocessing=False, initial_epoch=0)

    self._model.load_weights(model_path + '_fold-{}.hdf5'.format(fold))

    train_generator = batch_generator(view_train, batch_size[0], ids, view_data, False)
#    train_generator = batch_generator([sorted(list(view_data[0].keys())), sorted(list(view_data[1].keys()))], batch_size[0], ids, view_data, True)
    predictions = self._model.predict_generator(train_generator, steps=None, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)

    # train linear cca
    self._cca_model = cca.cca(self._emb_dims)
    self._cca_model.fit(predictions[:,:self._view_units[0][-1]], predictions[:,-self._view_units[1][-1]:])

    del view_train, view_test

    return history


  def embed_audio(self, data, batch_size):
    dummy_data = {}
    for key in data.keys():
      dummy_data[key] = np.empty((int(1.5 * 250), 16))

    predict_generator = batch_generator([sorted(list(data.keys())), sorted(list(dummy_data.keys()))], batch_size, True, [data, dummy_data], False)

    deep_embeddings = self._model.predict_generator(predict_generator, steps=None, max_queue_size=1, workers=1, use_multiprocessing=False, verbose=0)[:,:self._view_units[0][-1]]

    embeddings = self._cca_model.project_X(deep_embeddings)

    return embeddings, deep_embeddings
