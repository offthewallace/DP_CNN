#Author: Wallace He 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import keras

from differential_privacy.multiple_teachers import deep_cnn
from DP_CNN import partition
from DP_CNN import metrics
from DP_CNN import train_CNN

def train_teacher(nb_teachers, teacher_id):
  """
  This function trains a single teacher model with responds teacher's ID among an ensemble of nb_teachers
  models for the dataset specified.
  The model will be save in directory. 
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # Load the dataset
  train_data,train_labels,test_data,test_labels = train_CNN.get_dataset(load=False)
  
  # Retrieve subset of data for this teacher
  data, labels = partition.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path

  filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.hdf5'

  #train directory from the load picture
<<<<<<< HEAD
  ckpt_path = train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training need to modify 
  #modify  assert deep_cnn.train(data, labels, ckpt_path)
#store part.
    model, opt = create_six_conv_layer(X_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
    model, hist = training(model, data, test_data, data, test_labels, data_augmentation=True)

=======
  # Create teacher model
  model, opt = create_six_conv_layer(train_data.shape[1:])
  model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])
  model, hist = training(model, X_train, X_test, y_train, y_test, data_augmentation=True,filename)
>>>>>>> 9735b7e7ae0c40cc4be1b2654ab8fc831a6ec891
  #modify
  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

  #change to my own code
<<<<<<< HEAD
  teacher_preds = model.prediction()

#change the prediction function  precision = metrics.accuracy(teacher_preds, test_labels)

  print('Precision of teacher after training: ' + str(precision))
=======
  #teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)
  #https://machinelearningmastery.com/save-load-keras-deep-learning-models/ save model to json 

  #change the prediction function  precision = metrics.accuracy(teacher_preds, test_labels)
  model.save_weights(str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5')
>>>>>>> 9735b7e7ae0c40cc4be1b2654ab8fc831a6ec891

  return True


def main(argv=None): 

if __name__ == '__main__':
