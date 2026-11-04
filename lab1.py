import cv2
import numpy as np
from show  import showImg


img=cv2.imread("images/cat.jpg")
if img is None:
    raise FileNotFoundError("image is not found")
h,w=img.shape[:2]
f=img.astype(np.float32)
print("shape",img.shape)
print("shape",img.ndim)
print("data type",img.dtype)
print("range",np.min(img),np.max(img))
# making image gray
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("Grayscale shape: ",gray.shape)
print("Grayscale type: ",gray.dtype)
showImg([img,gray],['original',"grayscale"])

# i thought before running code the image will show me the original color as i implemented cvtcolor in showImg module 
# it show original img and gray scale code
# we should not do the color change it will reverse the change and also shows one image with black and white