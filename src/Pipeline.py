from functools import partial
import face_detection as fd
import preprocessing as prep
import feature_detection as featd
import numpy as np
from collections import defaultdict
from sklearn import metrics
from skimage.io import imread_collection
from operator import itemgetter
from sklearn import svm

# SVM 
def apply_SVM(X,y, c_value, gamma_value):
    svm_ = svm.SVC(C = c_value, gamma = gamma_value)
    clf = svm_.fit(X, y)
    return clf

# Preprocessing + face detection + Standardization
def face_recognition(dataset):
    dataset = prep.preprocess_dataset(dataset)
    faces=list(map(fd.detect_face,dataset))
    N = int(np.max([i.shape[0] for i in faces if len(np.unique(i))!=1]))
    M = int(np.max([i.shape[1] for i in faces if len(np.unique(i))!=1]))
    standardise = list(map(partial(featd.standardize_border,N=N,M=M), faces))
    return standardise

# HOG + SVM for pose detection
def classify_pose(dataset, info_dataset, C,gamma):
    gradient_pic=list(map(partial(featd.gradient_histogram,o=8,d_cell=24,d_block=2),dataset))
    y=info_dataset.Encoding_angle
    svm_=apply_SVM(gradient_pic,y,C,gamma)
    prediction=svm_.predict(gradient_pic)
    return prediction, metrics.confusion_matrix(y, prediction)

#  info on photo angle
def info_angle(dataset):
    pose = defaultdict(list)
    for idx in dataset.index:
        pose[dataset.Angle[idx]].append(idx)
    return pose

#    group photos by orientation and create Y true
def pose_category(dataset, dataset_csv):
    pose = info_angle(dataset_csv)
    pose_classes = {"Half Left" : list(imread_collection(itemgetter(*pose["Half Left"])(dataset)).files),
                    "Straight" : list(imread_collection(itemgetter(*pose["Straight"])(dataset)).files),
                    "Half Right" : list(imread_collection(itemgetter(*pose["Half Right"])(dataset)).files),
                    "Right" : list(imread_collection(itemgetter(*pose["Right"])(dataset)).files),
                    "Left" : list(imread_collection(itemgetter(*pose["Left"])(dataset)).files)}
    y_pose = {"Half Left" : dataset_csv.loc[dataset_csv.index.isin(pose["Half Left"])].Encoding_expression,
              "Straight" : dataset_csv.loc[dataset_csv.index.isin(pose['Straight'])].Encoding_expression,
              "Half Right" : dataset_csv.loc[dataset_csv.index.isin(pose["Half Right"])].Encoding_expression,
              "Right" : dataset_csv.loc[dataset_csv.index.isin(pose["Right"])].Encoding_expression,
              "Left" : dataset_csv.loc[dataset_csv.index.isin(pose["Left"])].Encoding_expression}
    return pose_classes, y_pose   


# LDA + SVM for emotion detection
def classify_emotion(dataset, y, n_comp=None,C=None, gamma=None, svm_=None,lda_=None):
    dataset_lda,lda_ = featd.apply_lda(dataset,y,n_comp,lda_)
    if svm_ == None:
        train_svm = apply_SVM(dataset_lda,y,C,gamma)
        return train_svm, train_svm.predict(dataset_lda),lda_
    else:
        dataset_svm = svm_.predict(dataset_lda)
        return dataset_svm
