from skimage.io import imread_collection
from sklearn.model_selection import train_test_split
from os import listdir
from operator import itemgetter
import pandas as pd
import numpy as np
import cleantext as clt
    
#create csv file of the whole dataset
label=pd.DataFrame(columns=['Number','Session','Gender','ID','Expression','Angle'])
name_pic=listdir('../Data/Dataset/pictures')
for pos in range(len(name_pic)):
    label.loc[pos,'Number']=pos
    label.loc[pos,'Session']=name_pic[pos][:1]
    label.loc[pos,'Gender']=name_pic[pos][1:2]
    label.loc[pos,'ID']=name_pic[pos][2:4]
    label.loc[pos,'Expression']=clt.convertion_expression(name_pic[pos][4:6])
    label.loc[pos,'Angle']=clt.convertion_angle(name_pic[pos][6:-4])


label['Encoding_expression'] = label.Expression.astype('category').cat.codes
label['Encoding_angle'] = label.Angle.astype('category').cat.codes
label.to_csv('../Data/Dataset/Label.csv',index=False)

#load pictures
path='../Data/Dataset/pictures/*.jpg'
dataset=imread_collection(path)


#split train and test
train_dataset,test_dataset=train_test_split(label,test_size=0.3,random_state=154)

train_dataset.to_csv('../Data/train/Train.csv',index=False)
train=imread_collection(itemgetter(*list(train_dataset['Number']))(dataset))
np.save('../Data/train/train.npy',np.array(train.files))

test_dataset.to_csv('../Data/test/Test.csv',index=False)
test=imread_collection(itemgetter(*list(test_dataset['Number']))(dataset))
np.save('../Data/test/test.npy',np.array(test.files))
