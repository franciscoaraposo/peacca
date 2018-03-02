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

import os
import subprocess
import sys

import argparse

import numpy
import scipy.signal

argparser = argparse.ArgumentParser(prog='filter_and_cut_eeg', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
argparser.add_argument('openbci')
argparser.add_argument('subject')
argparser.add_argument('channels', type=int)
argparser.add_argument('stimuli')
argparser.add_argument('output')

args = argparser.parse_args() 

openbci_path = args.openbci
assert os.path.exists(openbci_path), "openbci file not found"
subject_id = args.subject
channels = args.channels
stimuli_path = args.stimuli
assert os.path.exists(stimuli_path), "stimuli info file not found"
output_path = args.output
assert os.path.exists(output_path), "output directory not found"

# read relevant data from opebci file and store stimuli samplestamps
sampling_rate = 0.0 # Hz
sample_id = 0
samplestamps = {}
openbci_file = open(openbci_path, 'r')
tmp_path = output_path + '/everything.tmp'
tmp_file = open(tmp_path, 'w')
for line in openbci_file:
  if line.startswith('%Song '):
    stimulus_id = line[6:].rstrip()
    samplestamps[stimulus_id] = sample_id
  elif line.startswith('%Sample Rate = '):
    sampling_rate = numpy.float32(line[15:-3])
  elif line.startswith('%'):
    continue
  else:
    tmp_file.write(line)
    sample_id += 1
openbci_file.close()
tmp_file.close()

# extract relevant data (channels)
csv_path = output_path + '/everything.csv'
subprocess.Popen('cat {0} | awk -F, \'{{ for (i = 1; i < {1}; ++i) printf("%f ", $(i + 1)); print($({1} + 1)); }}\' > {2}'.format(tmp_path, channels, csv_path), shell=True, stdout=subprocess.PIPE).communicate()
subprocess.Popen('rm {0}'.format(tmp_path), shell=True, stdout=subprocess.PIPE).communicate()

# filter relevant data
samples = numpy.genfromtxt(csv_path)
subprocess.Popen('rm {0}'.format(csv_path), shell=True, stdout=subprocess.PIPE).communicate()
noise_frequency = 50.0 # Hz
relevant_band = (0.5, 49.5) # Hz
order = 5
b, a = scipy.signal.butter(order, [(noise_frequency-1)/(0.5*sampling_rate), (noise_frequency+1)/(0.5*sampling_rate)], btype='bandstop')
samples = scipy.signal.filtfilt(b, a, samples, axis=0)
b, a = scipy.signal.butter(order, [relevant_band[0]/(0.5*sampling_rate), relevant_band[1]/(0.5*sampling_rate)], btype='bandpass')
samples = scipy.signal.filtfilt(b, a, samples, axis=0)

# write each stimuli correlate to its own file
stimuli_file = open(stimuli_path, 'r')
for line in stimuli_file:
  tokens = line.split(' ')
  stimulus_id = tokens[0]
  samplestamp = samplestamps[stimulus_id]
  filename = stimulus_id[:-4]
  if filename == 'custom0':
    filename = '000'
  if filename == 'custom1':
    filename = '001'
  stimulus_duration = float(tokens[1])
  n_samples = int(stimulus_duration * sampling_rate)
  numpy.save(output_path + '/' + filename + '.' + subject_id + '.npy', samples[samplestamp:samplestamp+n_samples,:])
