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
import cca
import utils

def cca_loss(emb_dims):

  def cca_objective(y_true, y_pred):
    reg = 1e-4
    eps = 1e-12
    samples = tf.shape(y_pred)[0]

    audio_emb_dims = emb_dims[0]
    lyrics_emb_dims = emb_dims[1]
    audio_embs = tf.transpose(y_pred[:,:audio_emb_dims])
    lyrics_embs = tf.transpose(y_pred[:,-lyrics_emb_dims:])

    audio_embs_mean = audio_embs - (1.0 / tf.to_float(samples)) * tf.matmul(audio_embs, tf.ones([samples, samples]))
    lyrics_embs_mean = lyrics_embs - (1.0 / tf.to_float(samples)) * tf.matmul(lyrics_embs, tf.ones([samples, samples]))
    audio_embs_cov = (1.0 / (tf.to_float(samples) - 1)) * tf.matmul(audio_embs_mean, tf.transpose(audio_embs_mean)) + reg * tf.eye(audio_emb_dims)
    lyrics_embs_cov = (1.0 / (tf.to_float(samples) - 1)) * tf.matmul(lyrics_embs_mean, tf.transpose(lyrics_embs_mean)) + reg * tf.eye(lyrics_emb_dims)
    embs_cross_cov = (1.0 / (tf.to_float(samples) - 1)) * tf.matmul(audio_embs_mean, tf.transpose(lyrics_embs_mean))

    # eigen decomposition to find root inverse of covariance
    audio_evals, audio_evecs = tf.self_adjoint_eig(audio_embs_cov)
    lyrics_evals, lyrics_evecs = tf.self_adjoint_eig(lyrics_embs_cov)

    # filter out evals and evecs smaller than eps for stability
    idxs = tf.greater(audio_evals, eps)
    audio_evals = tf.boolean_mask(audio_evals, idxs)
    audio_evecs = tf.transpose(tf.boolean_mask(tf.transpose(audio_evecs), idxs))
    idxs = tf.greater(lyrics_evals, eps)
    lyrics_evals = tf.boolean_mask(lyrics_evals, idxs)
    lyrics_evecs = tf.transpose(tf.boolean_mask(tf.transpose(lyrics_evecs), idxs))

    audio_embs_cov_root_inv = tf.matmul(tf.matmul(audio_evecs, tf.diag(audio_evals ** -0.5)), tf.transpose(audio_evecs))
    lyrics_embs_cov_root_inv = tf.matmul(tf.matmul(lyrics_evecs, tf.diag(lyrics_evals ** -0.5)), tf.transpose(lyrics_evecs))

    t = tf.matmul(tf.matmul(audio_embs_cov_root_inv, embs_cross_cov), lyrics_embs_cov_root_inv)

    loss = -tf.sqrt(tf.trace(tf.matmul(tf.transpose(t), t)))

    return loss

  return cca_objective



class audio_lyrics_model(object):

  def __init__(self, cca_dims, view_dims, view_units):
    self._emb_dims, self._view_dims, self._view_units = cca_dims, view_dims, view_units
    self._model, self._cca_model = None, None


  def _build_model(self):
    keras.backend.clear_session()

    x, inputs, emb_dims = [None] * len(self._view_dims), [None] * len(self._view_dims), [None] * len(self._view_dims)
    for view_idx in range(len(self._view_dims)):
      x[view_idx] = inputs[view_idx] = keras.layers.Input(shape=(self._view_dims[view_idx],))

      x[view_idx] = keras.layers.BatchNormalization()(x[view_idx])

      for layer, units in enumerate(self._view_units[view_idx]):
        activation = 'tanh' if layer != (len(self._view_units[view_idx]) - 1) else 'linear'
        x[view_idx] = keras.layers.Dense(units, activation=activation)(x[view_idx])
        x[view_idx] = keras.layers.BatchNormalization()(x[view_idx]) if layer != (len(self._view_units[view_idx]) - 1) else x[view_idx]

      emb_dims[view_idx] = self._view_units[view_idx][-1]

    outputs = keras.layers.Concatenate()(x)
    self._model = keras.models.Model(inputs=inputs, outputs=outputs)
    self._model.compile(loss=cca_loss(emb_dims), optimizer=keras.optimizers.RMSprop(lr=1e-3))


  def _validate_fold(self, view_data, epochs, batch_size, fold, fold_path, model_path):
    view_train, view_test = utils.prepare_simple_multiview_fold(utils.load_fold_ids(fold_path), view_data, False)

    checkpointer = keras.callbacks.ModelCheckpoint(model_path + '_fold-{:0>2}.hdf5'.format(fold), monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min')

    history = self._model.fit(x=view_train, y=np.zeros(len(view_train[0])), batch_size=batch_size[0], epochs=epochs, shuffle=True, verbose=0, callbacks=[checkpointer], validation_data=(view_test, np.zeros(len(view_test[0]))))

    del view_train, view_test

    return history


  def _predict_embeddings(self, view_data, batch_size, folds, fold, fold_path, model_path):
    # initialize embeddings
    view_embs = [[None, None], [None, None]]

    # separate training and testing data
    view_train, view_test = utils.prepare_simple_multiview_fold(utils.load_fold_ids(fold_path), view_data, False)

    self._model.load_weights(model_path + '_fold-{:0>2}.hdf5'.format(fold))

    predictions = self._model.predict(view_train, batch_size=batch_size, verbose=0, steps=None)

    # train linear cca
    self._cca_model = cca.cca(self._emb_dims)
    self._cca_model.fit(predictions[:,:self._view_units[0][-1]], predictions[:,-self._view_units[1][-1]:])

    # predict embeddings for training data
    view_embs[0][0] = self._cca_model.project_X(predictions[:,:self._view_units[0][-1]])
    view_embs[1][0] = self._cca_model.project_Y(predictions[:,-self._view_units[1][-1]:])

    # predict embeddings for testing data
    predictions = self._model.predict(view_test, batch_size=batch_size, verbose=0, steps=None)

    view_embs[0][1] = self._cca_model.project_X(predictions[:,:self._view_units[0][-1]])
    view_embs[1][1] = self._cca_model.project_Y(predictions[:,-self._view_units[1][-1]:])

    del view_train, view_test

    return view_embs


  def cross_validate(self, view_data, epochs, folds, folds_path, batch_size, model_path):
    # view_data is a list of dictionaries mapping instance ids (string) to numpy arrays

    # initialize embeddings arrays
    dataset_size = len(view_data[0])
    test_size = dataset_size // folds
    train_size = dataset_size - test_size
    set_size = [train_size, test_size]
    view_embs = [None] * len(view_data)
    for view_idx in range(len(view_data)):
      view_embs[view_idx] = [None] * 2
      for set_idx in range(2):
        view_embs[view_idx][set_idx] = [None] * folds
        for fold in range(folds):
          view_embs[view_idx][set_idx][fold] = np.full((set_size[set_idx], self._emb_dims), fill_value=np.nan)

    for fold in range(folds):
      self._build_model()

      # create fold data
      fold_path = '{0}/{1:0>2}.txt'.format(folds_path, fold)

      history = self._validate_fold(view_data, epochs, batch_size, fold, fold_path, model_path)

      embs = self._predict_embeddings(view_data, batch_size[1], folds, fold, fold_path, model_path)

      for view_idx in range(len(view_data)):
        for set_idx in range(2):
          view_embs[view_idx][set_idx][fold] = embs[view_idx][set_idx]

    return view_embs
