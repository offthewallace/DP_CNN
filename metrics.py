from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def accuracy(logits, labels):
  """
  Return accuracy of the array of logits (or label predictions) wrt the labels
  :param logits: this can either be logits, probabilities, or a single label
  :param labels: the correct labels to match against
  :return: the accuracy as a float
  """
  assert len(logits) == len(labels)

  if len(np.shape(logits)) > 1:
    # Predicted labels are the argmax over axis 1
    predicted_labels = np.argmax(logits, axis=1)
  else:
    # Input was already labels
    assert len(np.shape(logits)) == 1
    predicted_labels = logits

  # Check against correct labels to compute correct guesses
  correct = np.sum(predicted_labels == labels.reshape(len(labels)))

  # Divide by number of labels to obtain accuracy
  accuracy = float(correct) / len(labels)

  # Return float value
  return accuracy


