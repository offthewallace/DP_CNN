#Author: Wallace He 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model

import partition
import train_CNN

def train_teacher (nb_teachers, teacher_id):
  """
  This function trains a single teacher model with responds teacher's ID among an ensemble of nb_teachers
  models for the dataset specified.
  The model will be save in directory. 
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # Load the dataset
  X_train, X_test, y_train, y_test = train_CNN.get_dataset()
  
  # Retrieve subset of data for this teacher
  data, labels = partition.partition_dataset(X_train,
                                         y_train,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path

  filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.hdf5'
  filename2 = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5'
 
  # Perform teacher training need to modify 
 

  # Create teacher model
  model, opt = train_CNN.create_six_conv_layer(data.shape[1:])
  model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  model, hist = train_CNN.training(model, data, X_test, labels, y_test,filename, data_augmentation=True)
  #modify
  model_json = model.to_json()
  with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
  model.save_weights(filename2)
  print("Saved model to disk")
  return True



