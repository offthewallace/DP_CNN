
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from differential_privacy.multiple_teachers import deep_cnn
from DP_CNN import input
from DP_CNN import metrics
from DP_CNN import train_CNN


""
#tf.flags.DEFINE_string('dataset', 'svhn', 'The name  of the dataset to use')
#tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/tmp','Temporary storage')
tf.flags.DEFINE_string('train_dir','/tmp/train_dir',
                       'Where model ckpt are saved')

tf.flags.DEFINE_integer('max_steps', 3000, 'Number of training steps to run.')
tf.flags.DEFINE_integer('nb_teachers', 50, 'Teachers in the ensemble.')
tf.flags.DEFINE_integer('teacher_id', 0, 'ID of teacher being trained.')

tf.flags.DEFINE_boolean('deeper', False, 'Activate deeper CNN model')

FLAGS = tf.flags.FLAGS


def train_teacher(dataset, nb_teachers, teacher_id):
  """
  This function trains a teacher (teacher id) among an ensemble of nb_teachers
  models for the dataset specified.
  :param dataset: string corresponding to dataset (svhn, cifar10)
  :param nb_teachers: total number of teachers in the ensemble
  :param teacher_id: id of the teacher being trained
  :return: True if everything went well
  """
  # If working directories do not exist, create them
 

  # Load the dataset

  train_data,train_labels,test_data,test_labels = train_CNN.load_pictures(load=False)
  

  # Retrieve subset of data for this teacher
  data, labels = input.partition_dataset(train_data,
                                         train_labels,
                                         nb_teachers,
                                         teacher_id)

  print("Length of training data: " + str(len(labels)))

  # Define teacher checkpoint filename and full path
  if FLAGS.deeper:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '_deep.ckpt'
  else:
    filename = str(nb_teachers) + '_teachers_' + str(teacher_id) + '.ckpt'
  ckpt_path = FLAGS.train_dir + '/' + str(dataset) + '_' + filename

  # Perform teacher training need to modify 
  assert deep_cnn.train(data, labels, ckpt_path)


  ckpt_path_final = ckpt_path + '-' + str(FLAGS.max_steps - 1)

#change to my own code
  teacher_preds = deep_cnn.softmax_preds(test_data, ckpt_path_final)

#change the prediction function  precision = metrics.accuracy(teacher_preds, test_labels)
  print('Precision of teacher after training: ' + str(precision))

  return True


def main(argv=None):  # pylint: disable=unused-argument
  # Make a call to train_teachers with values specified in flags
  assert train_teacher(FLAGS.dataset, FLAGS.nb_teachers, FLAGS.teacher_id)

if __name__ == '__main__':
  tf.app.run()
