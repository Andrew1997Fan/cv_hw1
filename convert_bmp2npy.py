import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np

# load
img = mpimg.imread('./test/bunny/depth_map.bmp') 

# height, width, channel=(360, 480, 3)
#h,w,c = img.shape 

# show
#print(img)
plt.imshow(img) 
plt.axis('off') 
#plt.show()

# save
np.save('./test/bunny/depth_map.npy',img,allow_pickle=True, fix_imports=True)
print('Done')
