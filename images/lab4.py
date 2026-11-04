from show import showImg
import numpy as np
import matplotlib.pyplot as plt
import cv2


img=cv2.imread("images/cat.jpg")

if img is None:
    raise FileNotFoundError("file not found")
fImg=img.astype(np.float32)
gamma=[0.5,1.0,2.0]

images=[img]
titles=["original"]
if any(x <= 0 for x in gamma):
    raise ValueError("Gamma must be greater than 0")
logImg=np.clip(255*np.log1p(fImg)/np.log(256),0,255).astype(np.uint8)
images.append(logImg)
titles.append("Log Image")
for gm in gamma:
    gmImg=np.clip(255*(fImg/255)**(1/gm),0,255).astype(np.uint8)
    images.append(gmImg)
    titles.append(f"gamma : {gm}")

showImg(images=images,titles=titles)