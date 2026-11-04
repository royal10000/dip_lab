import cv2
import matplotlib.pyplot as plt
import numpy as np

original=cv2.imread("images/cat.jpg")
if original is None:
    raise FileNotFoundError("file not found error")
rgb=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
h,w=rgb.shape[:2]
y,x=np.ogrid[:h,:w]

circle_x = 0.47
circle_y = 0.62
circle_radius = 0.28
rect_left = 0.35
rect_right = 0.88

A=((x-circle_x)**2+(y-circle_y)**2<=(min(h,w)*circle_radius)**2)
B=((x>=w*rect_left) & (x<w*rect_right) & (y>=h*0.45) & (y<h*0.85))

fig,axes=plt.subplots(2,4,figsize=(16,8))
mask={
    'NOT A':~A,
    'A OR B':A|B,
    'A AND B':A&B,
    'A XOR B':A^B
}

for col,(name,mask) in enumerate(mask.items()):
    result=np.where(mask[:,:,None],rgb,0)
    axes[0,col].imshow(mask,cmap='gray')
    axes[1,col].imshow(result)
    axes[0,col].axis('off')
    axes[1,col].axis("off")

plt.tight_layout();
plt.show();