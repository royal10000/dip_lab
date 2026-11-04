import cv2
import numpy as np
import matplotlib.pyplot as plt
original = cv2.imread('images/cat.jpg')
if original is None:
 raise FileNotFoundError("mandrill.tiff was not found.")
rgb = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
R,G,B = [rgb[:,:,i] for i in range(3)]
red_only=np.zeros_like(rgb); red_only[:,:,0]=R
green_only=np.zeros_like(rgb); green_only[:,:,1]=G
blue_only=np.zeros_like(rgb); blue_only[:,:,2]=B
# ===== PARAMETERS TO CHANGE =====
blue_margin_r = 1
blue_margin_g = 10
minimum_blue = 100
# ===============================
gray = cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
gray_rgb = np.repeat(gray[:,:,None],3,axis=2)
Ri,Gi,Bi=R.astype(np.int16),G.astype(np.int16),B.astype(np.int16)
blue_mask=((Bi > Ri + blue_margin_r) &
 (Bi > Gi + blue_margin_g) &
 (Bi > minimum_blue))
selective=np.where(blue_mask[:,:,None],rgb,gray_rgb)
images=[rgb,red_only,green_only,blue_only,selective]
titles=['Original','Red only','Green only','Blue only',
 f'Selective blue, min={minimum_blue}']
fig,axes=plt.subplots(1,5,figsize=(18,5))
for ax,img,title in zip(axes,images,titles):
 ax.imshow(img); ax.set_title(title); ax.axis('off')
plt.tight_layout(); plt.show()
print(f'Selected pixels: {100*blue_mask.mean():.2f}%')