#Author: Wallace He 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras

from DP_CNN import aggregation
from DP_CNN import partition
from DP_CNN import train_CNN


def ensemble_preds(nb_teachers, stdnt_data):
  """
  Given a dataset, a number of teachers, and some input data, this helper
  function queries each teacher for predictions on the data and returns
  all predictions in a single array. (That can then be aggregated into
  one single prediction per input using aggregation.py (cf. function
  prepare_student_data() below)
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param stdnt_data: unlabeled student training data
  :return: 3d array (teacher id, sample id, probability per class)
  """

  # Compute shape of array that will hold probabilities produced by each
  # teacher, for each training point, and each output class
  result_shape = (nb_teachers, len(stdnt_data), 2)

  # Create array that will hold result
  result = np.zeros(result_shape, dtype=np.float32)

  # Get predictions from each teacher

  #save model to json and reload https://machinelearningmastery.com/save-load-keras-deep-learning-models/
  for teacher_id in xrange(nb_teachers):
    # Compute path of weight file for teacher model with ID teacher_id
      filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.hdf5'
      model, opt = create_six_conv_layer(dataset.shape[1:])
      model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
      model.load_weights(filename, by_name=False)
    # Get predictions on our training data and store in result array
    result[teacher_id] = model.redict_proba(stdnt_data)

    # This can take a while when there are a lot of teachers so output status
    print("Computed Teacher " + str(teacher_id) + "predictions")

  return result


def prepare_student_data(test_data,nb_teachers, save=False,lap_scale,stdnt_share):
  """
  Takes a dataset name and the size of the teacher ensemble and prepares
  training data for the student model, according to parameters indicated
  in flags above.
  :param dataset: string corresponding to mnist, cifar10, or svhn
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :param save: if set to True, will dump student training labels predicted by
               the ensemble of teachers (with Laplacian noise) as npy files.
               It also dumps the clean votes for each class (without noise) and
               the labels assigned by teachers
  :return: pairs of (data, labels) to be used for student training and testing
  """
 
  # Make sure there is data leftover to be used as a test set

  # Prepare [unlabeled] student training data (subset of test set)
  stdnt_data = test_data[:stdnt_share]

  # Compute teacher predictions for student training data
  teachers_preds = ensemble_preds(nb_teachers, stdnt_data)

  # Aggregate teacher predictions to get student training labels
    stdnt_labels = aggregation.noisy_max(teachers_preds,lap_scale)
   

  # Print accuracy of aggregated labels
  ac_ag_labels = metrics.accuracy(stdnt_labels, test_labels[:FLAGS.stdnt_share])
  print("Accuracy of the aggregated labels: " + str(ac_ag_labels))

  # Store unused part of test set for use as a test set after student training
  stdnt_test_data = test_data[stdnt_share:]
  stdnt_test_labels = test_labels[Fstdnt_share:]

'''
  if save:
    # Prepare filepath for numpy dump of labels produced by noisy aggregation
    filepath = FLAGS.data_dir + "/" + str(dataset) + '_' + str(nb_teachers) + '_student_labels_lap_' + str(FLAGS.lap_scale) + '.npy' #NOLINT(long-line)

    # Dump student noisy labels array
    with tf.gfile.Open(filepath, mode='w') as file_obj:
      np.save(file_obj, stdnt_labels)
'''
  return stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels


def train_student(dataset, nb_teachers):
  """
  This function trains a student using predictions made by an ensemble of
  teachers. The student and teacher models are trained using the same
  neural network architecture.
  :param dataset: string corresponding to import data
  :param nb_teachers: number of teachers (in the ensemble) to learn from
  :return: True if student training went well
  """
  # Call helper function to prepare student data using teacher predictions
  stdnt_dataset = prepare_student_data(dataset, nb_teachers, save=True,stdnt_share)

  # Unpack the student dataset
  stdnt_data, stdnt_labels, stdnt_test_data, stdnt_test_labels = stdnt_dataset

  
    filename = + 'student.hdf5'

  # Start student training
	model, opt = create_six_conv_layer(stdnt_data.shape[1:])
  model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  model, hist = training(model, stdnt_data, stdnt_test_data, stdnt_labels, stdnt_test_labels, data_augmentation=True,filename)
  #modify
  # Compute final checkpoint name for student (with max number of steps)
  model.save_weights('student.h5')


  return True


