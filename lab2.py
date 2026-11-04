import cv2
import numpy as np
from show import showImg

img=cv2.imread("images/cat.jpg")
if img is None:
    raise FileNotFoundError("file not found")
flt=img.astype(np.float32)
brightness=[-100,0,50,100]
contrast=[0.5,0,2.5,5]
# mid_gray=[]
images=[]
titles=[]
for i in brightness:
    bright=np.clip(flt+i,0,255).astype(np.uint8)
    images.append(bright)
    titles.append(f"Brightness :{i}")

for i in contrast:
    cn=np.clip(i*flt,0,255).astype(np.uint8)
    images.append(cn)
    titles.append(f"Contrast : {i}")

# for i in brightness:
#     alpha=259*(i+255)/(255*(259-i))
#     mid_gray=np.clip((alpha*(flt-128))+128,0,255).astype(np.uint8)
#     images.append(mid_gray)
#     titles.append(f"Mid Gray : {i}")
showImg(images,titles,)

# as i already knew increasing/decreasing brightness will make it color close to black or white but i thought increasing contrast could make it like orange or like dark orange
#  In brightness the increasing make the overall enviroment light and decreasing darken it and same with contrast but it multiples the effect 
