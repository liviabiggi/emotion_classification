import numpy as np
import pandas as pd
import Pipeline as pline
import cleantext as cl

''' train '''
#load train
train_=np.load('../Data/train/train.npy')
info_train=pd.read_csv('../Data/train/Train.csv')

# preprocess, face detection and standardise train set
prep_train= pline.face_recognition(train_)

# predict pic orientation (+ create confusion matrix)
train_prediction, confusion_matrix_train = pline.classify_pose(prep_train, info_train, C=0.9,gamma=0.005)

# results of pose prediction 
train_orientation = pd.DataFrame({'Number':info_train.Number,'Angle': train_prediction})
train_orientation['Angle'] = train_orientation['Angle'].map({0:'Half Left', 1:'Half Right', 2: 'Left', 3:'Right', 4:'Straight'})
train_orientation.to_csv('../Results/Train/train_results.csv', index=False)

# group pics by pose and save Y true
train_pose, y_pose = pline.pose_category(prep_train, info_train)

# compute LDA + SVM of train set
train_svm_straight, train_pred_straight,train_lda_straight = pline.classify_emotion(dataset=train_pose["Straight"],y=y_pose["Straight"], n_comp=4,C=0.05, gamma=0.15)
train_svm_hl, train_pred_hl,train_lda_hl = pline.classify_emotion(dataset=train_pose["Half Left"], y=y_pose["Half Left"], n_comp=4,C=0.9, gamma=0.01)
train_svm_hr, train_pred_hr,train_lda_hr = pline.classify_emotion(dataset=train_pose["Half Right"], y=y_pose["Half Right"], n_comp=4,C=1.13, gamma=0.145)
train_svm_left, train_pred_left,train_lda_left = pline.classify_emotion(dataset=train_pose["Left"], y=y_pose["Left"], n_comp=4,C=0.2, gamma=0.35)
train_svm_right, train_pred_right,train_lda_right = pline.classify_emotion(dataset=train_pose["Right"],y= y_pose["Right"], n_comp=4,C=0.22, gamma=0.027)

# results of emotion detection
final_result={"Straight":train_pred_straight,"Half Left":train_pred_hl,"Half Right":train_pred_hr,"Left":train_pred_left,"Right":train_pred_right}
cl.csv_result(y_pose,final_result,train_orientation,'../Results/Train/train_results.csv')


''' test '''
#load test
test=np.load('../Data/test/test.npy')
info_test=pd.read_csv('../Data/test/Test.csv')

# preprocess, face detection and standardise test set
prep_test= pline.face_recognition(test)

# predict pic orientation (+ create confusion matrix)
test_prediction, confusion_matrix_test = pline.classify_pose(prep_test, info_test, C=0.6, gamma=0.4)

# results of pose prediction 
test_orientation = pd.DataFrame({'Number':info_test.Number,'Angle': test_prediction})
test_orientation['Angle'] = test_orientation['Angle'].map({0:'Half Left', 1:'Half Right', 2: 'Left', 3:'Right', 4:'Straight'})
test_orientation.to_csv('../Results/Test/test_results.csv', index=False)
test_orientation['Encoding_expression'] = info_test.Encoding_expression

# group pics by pose and save Y true
test_pose, y_pose_test = pline.pose_category(prep_test, test_orientation)

# compute PCA + SVM of test set
test_pred_straight = pline.classify_emotion(test_pose["Straight"], y_pose_test["Straight"], svm_=train_svm_straight,lda_=train_lda_straight)
test_pred_hl = pline.classify_emotion(test_pose["Half Left"], y_pose_test["Half Left"], svm_=train_svm_hl,lda_=train_lda_hl)
test_pred_hr = pline.classify_emotion(test_pose["Half Right"], y_pose_test["Half Right"], svm_=train_svm_hr,lda_=train_lda_hr)
test_pred_left = pline.classify_emotion(test_pose["Left"], y_pose_test["Left"], svm_=train_svm_left,lda_=train_lda_left)
test_pred_right = pline.classify_emotion(test_pose["Right"], y_pose_test["Right"],svm_=train_svm_right,lda_=train_lda_right)

# results of emotion detection
final_result={"Straight":test_pred_straight,"Half Left":test_pred_hl,"Half Right":test_pred_hr,"Left":test_pred_left,"Right":test_pred_right}
cl.csv_result(y_pose_test,final_result,test_orientation,'../Results/Test/test_results.csv')
