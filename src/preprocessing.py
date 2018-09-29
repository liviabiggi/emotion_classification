from PIL import Image
import numpy as np
import cv2

'''Contrast stretch'''

#normalize pixel
def normalize(band):
    band=np.array(band)/255
    minOut=0; maxOut=1; minInp=np.min(band); maxInp=np.max(band)
    bandnorm=(band-minInp)*(((maxOut-minOut)/(maxInp-minInp))+minOut)*255
    return Image.fromarray(bandnorm.astype('uint8'))

# Implementation contrast stretching
def contrast_stretching(array):
    pic=Image.fromarray(array)
    multiBands=list(pic.split())
    normalizedImage=Image.merge("RGB",(normalize(multiBands[0]),normalize(multiBands[1]),normalize(multiBands[2])))
    return np.array(normalizedImage)

'''Adaptive Gamma Correction''' 

# split pictures based on their brightness
def Brightness(matrix):
    m=matrix.sum()/(matrix.shape[0]*matrix.shape[1])
    t=(m-112)/112
    if t<-0.3:
        matrix=AGC_truncated(matrix)
    elif t>0.3:
        matrix=AGC_negative(matrix)
    else:
        matrix=matrix
    return matrix

# new pixel intensity
def new_row(mat,g,m):
    return np.round(m*np.power((mat/m),g))

# truncated gamma
def new_row_truncated(mat,g,m):
    gamma=np.maximum(mat,np.array(0.5))
    return np.round(m*np.power((mat/m),gamma))

# AGC for bright images
def AGC_negative(mat,alpha=0.25):
    mat=255-mat
    prob=calculate_weighted(mat,alpha)
    gamma=calculate_gamma(prob,mat)
    mat=new_row(mat,gamma,np.max(mat))
    return np.round(255-mat)

# AGC for dimmed images
def AGC_truncated(mat,alpha=0.75):
    prob=calculate_weighted(mat,alpha)    
    gamma=calculate_gamma(prob,mat)
    mat=new_row_truncated(mat,gamma,np.max(mat))
    return mat

# create the gamma matrix 
def calculate_gamma(prob,mat):
    gamma=1-np.cumsum(prob)
    w,h=(mat.shape[0],mat.shape[1])
    mat=mat.flatten()
    output=np.array([gamma[elem] for elem in mat])
    output=output.reshape(w,h)    
    return output

# calculate the weighted probability for each value of pixel
def calculate_weighted(matrix,alpha):
    pic=Image.fromarray(matrix)
    prob=np.array(pic.histogram())/np.array(pic.histogram()).sum()
    min_=np.amin(prob[prob!=np.amin(prob)]); max_=np.max(prob)
    probweight=max_*np.power(((prob*min_)/(max_-min_)),alpha)
    return probweight

# AGC steps 
def AGC_implementation(image):
    pic=Image.fromarray(cv2.cvtColor(image,cv2.COLOR_RGB2HSV))
    V_matrix=np.array(list(pic.split())[2])
    V_matrix=Brightness(V_matrix)
    V_image=Image.fromarray(V_matrix.astype('uint8'))
    Im=Image.merge("HSV",(list(pic.split())[0],list(pic.split())[1],V_image))
    return cv2.cvtColor(np.array(Im),cv2.COLOR_HSV2RGB)
    
# pipeline preprocessing
def preprocess_dataset(dataset):
    dataset=list(map(contrast_stretching,dataset))
    dataset=list(map(AGC_implementation,dataset))
    return dataset
