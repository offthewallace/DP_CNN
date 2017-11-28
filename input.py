
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


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
