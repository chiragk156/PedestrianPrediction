CSL-462 Project: Pedestrain Analysis

Ashish Kumar 2016CSB1033
Chirag Khurana 2016CSB1037

Programming Language: Python-3
==========================================Requirements============================================
Tensorflow Gpu
Keras
Open-CV
Numpy
Scipy
Pandas
Matplotlib

=======================================How to train/run===========================================

To train the teacher model: 
python train1.py teacher

To train the student model:
python train1.py student
Note: Only train the student model once you have trained the teacher model
To do inference for direction using the model:
python train1.py infer

and then input the filepath when asked

To train the large model for both direction and distraction prediction:
python train2.py train
To do direction and distraction prediction:
python train2.py infer

and then input the filepath when asked

models are saved under models/fullmodel_transfer2.h5 for teacher model
models are saved under models/KDmodel.h5 for student model
models are saved under models/distraction.h5 for large model for both distraction and direction prediction
graphs are saved under models/fullmodel_transfer2_history for parent model
graphs are saved under models/KDmodelhistory for parent model


ALL OTHER FILES AND CODE IS THE INTERMEDIATE CODE WHICH WAS GENERATED FOR COMPARING RESULTS/CREATING DATASET/ OTHER WORK
AND IS NOT REQUIRED FOR MAKING INFERENCE OR TRAINING THE MODELS. IT IS JUST INCLUDED AS A PROOF THAT WE HAVE DONE THE WORK OURSELVES.

NOTE: YOU CAN USE SOME IMAGES PRESENT IN THE TEST FOLDER FOR CHECKING INFERENCE

Report is present in Report/Report.pdf