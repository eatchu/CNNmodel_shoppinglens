import matplotlib.pyplot as plt
import argparse
from os import listdir
from PIL import Image
import numpy as np



curos = 'C:\\Users\\user\\Desktop\\dataset\\db_final'
files = listdir(curos)

image = Image.open(files[0])
imag_np = np.array(image)
l1 = int(imag_np.shape[0]*0.2)
l2 = int(imag_np.shape[1]*0.2)
imag_np = imag_np[l1:imag_np.shape[0] - l1,l2:imag_np.shape[1]-l2 ,: ]
image3 =image2.reshape((image2.shape[0] * image2.shape[1],3))
clt = KMeans(n_clusters = 1)
clt.fit(image3)

RGB_data = [clt.cluster_centers_[0][0],clt.cluster_centers_[0][1],clt.cluster_centers_[0][2]]
image_box = np.zeros((150,125,3),dtype=int)
image_box[:,:,0] = RGB_data[0]
image_box[:,:,1] = RGB_data[1]
image_box[:,:,2] = RGB_data[2]

fig = plt.figure()
a = fig.add_subplot(1,2,1)
plt.imshow(image2)
a = fig.add_subplot(1,2,2)
plt.imshow(image_box)





