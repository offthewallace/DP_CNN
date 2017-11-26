# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cPickle
import gzip
import math
import numpy as np
import os
from scipy.io import loadmat as loadmat
from six.moves import urllib
import sys
import tarfile

import tensorflow as tf

FLAGS = tf.flags.FLAGS


def create_dir_if_needed(dest_directory):
  """
  Create directory if doesn't exist
  :param dest_directory:
  :return: True if everything went well
  """
  if not tf.gfile.IsDirectory(dest_directory):
    tf.gfile.MakeDirs(dest_directory)

  return True


def partition_dataset(data, labels, nb_teachers, teacher_id):
  """
  Simple partitioning algorithm that returns the right portion of the data
  needed by a given teacher out of a certain nb of teachers
  :param data: input data to be partitioned
  :param labels: output data to be partitioned
  :param nb_teachers: number of teachers in the ensemble (affects size of each
                      partition)
  :param teacher_id: id of partition to retrieve
  :return:
  """

  # Sanity check
  assert len(data) == len(labels)
  assert int(teacher_id) < int(nb_teachers)

  # This will floor the possible number of batches
  batch_len = int(len(data) / nb_teachers)

  # Compute start, end indices of partition
  start = teacher_id * batch_len
  end = (teacher_id+1) * batch_len

  # Slice partition off
  partition_data = data[start:end]
  partition_labels = labels[start:end]

  return partition_data, partition_labels
