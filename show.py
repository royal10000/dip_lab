import matplotlib.pylab as plt
import numpy as np
import cv2


def showImg(images,titles,color_type="BGR"):
    
    imgCount=len(images)
    cols=4
    rows=np.ceil(imgCount/cols).astype(np.int64)
    fig,axes=plt.subplots(rows,cols,figsize=(4*cols,16),squeeze=False)
    for ax,img,title in zip(axes.flat,images,titles):
        if img.ndim==2:
            ax.imshow(img,cmap="gray",vmin=0,vmax=1 if img.dtype==bool else 255 )#if img.dtype==bool else 255
        else:
            if(color_type=="BGR"):
                ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
            else:
                ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()