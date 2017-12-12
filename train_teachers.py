#Author: Wallace He 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import keras

from DP_CNN import partition
from DP_CNN import train_CNN

def train_teacher(nb_teachers, teacher_id,dirc):
  """
  This function trains a single teacher model with responds teacher's ID among an ensemble of nb_teachers
  models for the dataset specified.
  The model will be save in directory. 
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # Load the dataset
  train_data,train_labels,test_data,test_labels = train_CNN.get_dataset(load=False,dirc)
  
  # Retrieve subset of data for this teacher
  data, labels = partition.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path

  filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.hdf5'

  #train directory from the load picture
  ckpt_path = train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training need to modify 
 

  # Create teacher model
  model, opt = create_six_conv_layer(train_data.shape[1:])
  model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  model, hist = training(model, X_train, X_test, y_train, y_test, data_augmentation=True,filename)
  #modify

  #change to my own code
  #https://machinelearningmastery.com/save-load-keras-deep-learning-models/ save model to json 

  #change the prediction function  precision = metrics.accuracy(teacher_preds, test_labels)
  model.save_weights(str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5')

  return True


