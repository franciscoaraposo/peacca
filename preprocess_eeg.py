#!/usr/bin/env python3

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

import sys
import argparse
import numpy
import scipy.signal
import utils

argparser = argparse.ArgumentParser(prog='preprocess_eeg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('input_file')
argparser.add_argument('output_file')
argparser.add_argument('--war', type=float, default=None)
argparser.add_argument('--wsd', type=float, default=None)
argparser.add_argument('--ssr', type=float, default=None)
argparser.add_argument('--tsr', type=float, default=None)

args = argparser.parse_args()

input_file = args.input_file
output_file = args.output_file
artifact_threshold = args.war
noise_threshold = args.wsd
source_sampling_rate = args.ssr
target_sampling_rate = args.tsr

data = numpy.load(input_file)

if artifact_threshold is not None:
  # remove artifacts
  data = utils.wavelet_artifact_removal(data, threshold=artifact_threshold, wavelet='db8', mode='symmetric')

# scale to [-1,1]
tmp = data - numpy.min(data, 0)
data = (tmp / numpy.max(tmp, 0) - 0.5) * 2

if noise_threshold is not None:
  # denoise the signal
  data = utils.wavelet_semblance_denoising(data, threshold=noise_threshold, wavelet='db8', mode='symmetric')

if source_sampling_rate is not None and target_sampling_rate is not None:
  # resample
  data = scipy.signal.resample_poly(data, target_sampling_rate, source_sampling_rate, axis=0)

numpy.save(output_file, data)
