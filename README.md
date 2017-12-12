# DP_CNN

## 1.Model Description 

  The Final project will implement the Private Aggregation of Teacher Ensembles (PATE) by using custom datasets and training CNN model.

  Based on the given code from the paper, the project will be implemented based on  Keras backend instead of the Tensorflow as backend by the given paper. Also Keras is the machine learning backend used by custom CNN model. 

  After implement custom training model and revised the source code, the new student model will be tested by the custom training datasets and comparing with the original training model based on the trade off between the accuracy and the privacy.
  
  
## 2.Detail of Implementation
  
   The refernce code of  SEMI-SUPERVISED KNOWLEDGE TRANSFER
FOR DEEP LEARNING FROM PRIVATE TRAINING DATA can be found in here   https://github.com/tensorflow/models/tree/master/research/differential_privacy/multiple_teachers

   The project has 6 files in total includes: trainTeachers.py, trainStudent.py, aggrgation.py, trainCNN.py, andpartition.py. The the flowchart of whole program is listed below.
   
 ![alt text](DP_CNN/Diagram.png)
 
 Description of flowchart
 
 Input of whole program:
 private dataset: P for training teacher model
 public dataset 1: A for training student model
 public dataset 2: B for testing student model
 Number of teacher-models n and teacher-model’s id i .
 
 Step1: We will train a single teacher model based on the given private data-sets. the Inside of trainTeach-ers.py’s trainTeacher() function, import getDataset() function from the trainCNN.py to get the private databy given directory. Then use createSixConvLayer() function from same file to create the a emptymodeli. Then use the partitionDataset() function from the partition.py to partition the p into n part disjoin datasets P nwith same length.  Use Pito trainmodeli.  Then run trainTeacher() n timesto create teacher models:model1...modeln

 
## 3.implement details and change

  1.Delete most of the functions from the original import file since It is unnecessary to import minist or CR10. I added the import functions inside of training CNN files for import the data-path and parsing theimages.
  
  2.Modified the partition-dataset() function inside of partition.py to fit the given datas-ets.
  
  3.Modlify the aggregation.py to fit in Keras version



