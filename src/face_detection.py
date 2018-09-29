from skimage.morphology import disk,opening
from skimage.measure import regionprops
import cv2
from PIL import Image
import numpy as np

''' FACE DETECTION '''

# Identify the regions of the face matching the (mean) skin tone
def identify_skin(mask):
    pic=Image.fromarray(cv2.cvtColor(mask,cv2.COLOR_RGB2HSV))
    H_matrix=np.array(list(pic.split())[0])
    V_matrix=np.array(list(pic.split())[2])
    h_flatten=H_matrix.flatten()
    v_flatten=V_matrix.flatten()
    v_flatten[np.logical_and(h_flatten<=240, h_flatten>19)]=0
    V=np.matrix(v_flatten).reshape(mask.shape[0],mask.shape[1])
    V_image=Image.fromarray(V.astype('uint8'))
    selem=disk(10)
    V_opened=opening(V_image,selem)
    return V_opened

# find the properties of the skin regions 
def face_properties(skin):
    thresh, im_bw = cv2.threshold(np.array(skin), 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    mat=im_bw.astype('uint8')
    labeled_foreground = (mat > thresh).astype(int)
    properties = regionprops(labeled_foreground, mat)
    return properties

#  implement the functions above to detect the face and crop the image accordingly
def detect_face(img):
    skin = identify_skin(img)
    if len(np.unique(skin))==1 and np.unique(skin).item()==0:
        return skin
    properties = face_properties(skin)
    cropped_img = img[properties[0].bbox[0]:properties[0].bbox[2], properties[0].bbox[1]:properties[0].bbox[3]]
    return cropped_img

