import cv2
import matplotlib.pyplot as plt
import numpy as np

original=cv2.imread("images/cat.jpg")
if original is None:
    raise FileNotFoundError("file not found error")
rgb=cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
h,w=rgb.shape[:2]
y,x=np.ogrid[:h,:w]

circle_x=0.23
circle_y=0.25
circle_radius=0.30

rect_left=0.24
rect_right=0.65

A=((x-circle_x)**2+(y-circle_y)**2<=(min(h,w)*circle_radius)**2)
B=((x>=w*rect_left) & (x<w*rect_right) & (y>=h*0.45) & (y<h*0.85))

fig,axes=plt.subplots()