import cv2
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from PIL import Image
from skimage.feature import hog


''' FEATURE EXTRACTION (using LDA) '''

# add padding
def standardize_border(pic,N,M):
    if len(pic.shape)==2:
        return pic
    old_im = Image.fromarray(pic)
    old_size = old_im.size
    new_size = (M, N)
    new_im = Image.new("RGB", new_size)   ## luckily, this is
    new_im.paste(old_im, (int((new_size[0]-old_size[0])/2),int((new_size[1]-old_size[1])/2)))
    img = cv2.cvtColor(np.array(new_im),cv2.COLOR_RGB2GRAY)
    return img

# apply LDA for dimensionality reduction
def apply_lda(X,y,n,lda_):
    X=list(map(lambda pic: pic.flatten(),X))
    if lda_==None:
        lda = LDA(n_components=n)
        lda.fit(X,y)
        X_lda= lda.transform(X)
        return X_lda,lda
    else:
        X_lda= lda_.transform(X)
        return X_lda,None

# Calculate HOG histogram 
def gradient_histogram(pic,o,d_cell,d_block):
    gradient_vec=hog(pic,block_norm='L2',orientations=o,pixels_per_cell=(d_cell, d_cell),cells_per_block=(d_block, d_block))
    return gradient_vec
    

    






