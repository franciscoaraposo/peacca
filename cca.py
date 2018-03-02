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
import scipy.linalg as la

# X and Y are of shape (N, Fx) and (N, Fy) where N is the number of samples and Fx, Fy are the number of features
class cca(object):

  def __init__(self, components, regularizer=1e-4):
    self._components = components
    self._regularizer = regularizer
    self._mean_x, self._mean_y, self._transform_x, self._transform_y = None, None, None, None


  def fit(self, X, Y):
    if X.shape[0] != Y.shape[0]:
      raise Exception('X and Y must contain the same number of samples')

    samples = X.shape[0]
    dims_x = X.shape[1]
    dims_y = Y.shape[1]

    self._mean_x = np.mean(X, axis=0)
    self._mean_y = np.mean(Y, axis=0)

    x = X - self._mean_x
    y = Y - self._mean_y

    cov_cross = (1.0 / (samples - 1)) * np.dot(x.T, y)
    cov_x = (1.0 / (samples - 1)) * np.dot(x.T, x) + self._regularizer * np.identity(dims_x)
    cov_y = (1.0 / (samples - 1)) * np.dot(y.T, y) + self._regularizer * np.identity(dims_y)

    x_evals, x_evecs = la.eigh(cov_x)
    y_evals, y_evecs = la.eigh(cov_y)
    root_inv_cov_x = np.dot(np.dot(x_evecs, np.diag(x_evals ** -0.5)), x_evecs.T)
    root_inv_cov_y = np.dot(np.dot(y_evecs, np.diag(y_evals ** -0.5)), y_evecs.T)

    t = np.dot(np.dot(root_inv_cov_x, cov_cross), root_inv_cov_y)

    u, s, v_t = la.svd(t)
    v = v_t.T
    self._transform_x = np.dot(root_inv_cov_x, u[:,:self._components])
    self._transform_y = np.dot(root_inv_cov_y, v[:,:self._components])

    return self


  def project_X(self, X):
    return np.dot(X - self._mean_x, self._transform_x)


  def project_Y(self, Y):
    return np.dot(Y - self._mean_y, self._transform_y)
