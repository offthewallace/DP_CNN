#Author: Wallace He 

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import keras
import aggregation
import partition
import train_CNN


def ensemble_preds(nb_teachers, stdnt_data, num_class):
    """
    Given a dataset, a number of teachers, and some input data, this helper
    function queries each teacher for predictions on the data and returns
    all predictions in a single array. 
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :param stdnt_data: unlabeled student training data
    :return: 3d array (teacher id, sample id, probability per class)
    """

    # Compute shape of array that will hold probabilities produced by each
    # teacher, for each training point, and each output class
    result_shape = (nb_teachers, len(stdnt_data), num_class)

    # Create array that will hold result
    result = np.zeros(result_shape, dtype=np.float32)

    # Get predictions from each teacher

    #save model to json and reload https://machinelearningmastery.com/save-load-keras-deep-learning-models/
    for teacher_id in range(nb_teachers):
        # Compute path of weight file for teacher model with ID teacher_id
        filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.h5'
        model = train_CNN.load_model_from_checkpoint(filename, six_conv=True)
        # Get predictions on our training data and store in result array
        result[teacher_id] = model.predict_proba(stdnt_data)

        # This can take a while when there are a lot of teachers so output status
        print("Computed Teacher " + str(teacher_id) + "predictions")

    return result


def prepare_student_data(test_data,nb_teachers,lap_scale):
    """
    Takes a dataset name and the size of the teacher ensemble and prepares
    training data for the student model
    :param dataset: string corresponding to mnist, cifar10, or svhn
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :Param: lap_scale: scale of the Laplacian noise added for privacy
    :return: pairs of (data, labels) to be used for student training and testing
    """

    # Compute teacher predictions for student training data
    teachers_preds = ensemble_preds(nb_teachers, test_data,2)
    

    # Aggregate teacher predictions to get student training labels
    stdnt_labels = aggregation.noisy_max(teachers_preds,lap_scale)
    print('stdnt_labels')
    stdnt_labels = keras.utils.to_categorical(stdnt_labels, 2)
    print(len(stdnt_labels))
    print(stdnt_labels.shape)

    # Store unused part of test set for use as a test set after student training
    
    return stdnt_labels


def train_student(nb_teachers):
    """
    This function trains a student using predictions made by an ensemble of
    teachers. The student and teacher models are trained using the same
    neural network architecture.
    :param nb_teachers: number of teachers (in the ensemble) to learn from
    :return: True if student training went well
    """
    # you need to change the address of get_dataset() manuly 
    X_train, X_test, y_train, y_test = train_CNN.get_dataset()
    
   # Call helper function to prepare student data using teacher predictions
    y_train= prepare_student_data(X_train, nb_teachers,10)
 
    filename = 'student.hdf5'
    # Start student training
    model, opt = train_CNN.create_six_conv_layer(X_train.shape[1:])
    model.compile(loss='categorical_crossentropy',
                            optimizer=opt,
                            metrics=['accuracy'])
    model, hist = train_CNN.training(model, X_train, X_test, y_train, y_test, filename, data_augmentation=True)
    #modify
    # Compute final checkpoint name for student 
    model.save_weights('student.h5')


    return True
if __name__ == '__main__':
    #modify the dirc first
    train_student(10)
     




