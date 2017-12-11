# DP_CNN

## 1.Model Description 

  The Final project will implement the Private Aggregation of Teacher Ensembles (PATE) by using custom datasets and training CNN model.

  Based on the given code from the paper, the project will be implemented based on  Keras backend instead of the Tensorflow as backend by the given paper. Also Keras is the machine learning backend used by custom CNN model. 

  After implement custom training model and revised the source code, the new student model will be tested by the custom training datasets and comparing with the original training model based on the trade off between the accuracy and the privacy.
  
  
## 2.Final Project Outline
  
  In the sample codes file it has 8 files. The source code files can be found in here   https://github.com/tensorflow/models/tree/master/research/differential_privacy/multiple_teachers

  Aggregation.py


  Analysis.py


  Deep-Cnn.py


  Input.py


  Metrics.py

  Train-students.py

  Train-teachers.py

  Utils.py

  The introduction of each files and functionn from the source code


  1.Input.py: It is the file for simple import the mnist or extract the Cifar10 files as input, so I dont think I will keep this file for my research project since I already had my input function in my own CNN file.


  2.Utils.py: this file  has one function for computes a batch start and end index. 


  3.Aggregation.py: used for aggregating the different teachers models’ vote into one result while applying the laplace noise into those votes for the function “noisymax”. Another function inside of Aggreation.py would be “aggregation-most-frequent”, it’s kind of like the “Above Threshold” but only return the highest  “vote of label”  during the training for student model.


  4.DeepCnn.py： Neural network file


  5.Metrices.py: calculate the accuracy of the array of logits (or label predictions) with the labels. 


  6.Trainteachers.py: Function  “train-teacher”: It would train the teacher model based on the number of teachers and give each teacher model a number. The data set would be partite into # of Teachers parts.


  7.Trainstudents.py: It has three functions. Ensemblepreds: it would return teachers’ model predict results based on the the student’s input, which know as the “public data”.  Basic machine learning predict step.


  8.Prepare-student-data: it would use the functions in “Aggregation.py” to apply the result returned after “Ensemble-preds” to create the privacy result for training the student model


  9.Train-student.py： basic machine learning training for student model by the private data applied DP by Prepare-student-data


## 3.implement details and change

  1.Input.py: I delete most of the functions from the orginal file since I dont need to import the file from the internet like minist or CR10. I added the import functions inside of my training CNN files for import the datapath and parsing the images.
Also I modified the partition-dataset to fit my process.



