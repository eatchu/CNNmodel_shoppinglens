# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 10:50:55 2020

@author: user
"""


import tensorflow as tf
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from numpy import dot
from numpy.linalg import norm
import matplotlib.pyplot as plt
import argparse
import cv2
from keras.preprocessing import image
import os
from PIL import Image
import keras
import cv2
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator 



def cos_sim(A, B):
       return dot(A, B)/(norm(A)*norm(B))
def euclidean_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += (pt1[i] - pt2[i]) ** 2
  return distance ** 0.5
def manhattan_distance(pt1, pt2):
  distance = 0
  for i in range(len(pt1)):
    distance += abs(pt1[i] - pt2[i])
  return distance

# 1) 바지(크롭위치 : 왼쪽 허벅지)를 위한 색깔 함수
def cluster_color_pants():
   # bgr -> rgb
    image = Image.open(image_dir).resize((130,130))
    imag_np = np.array(image)
    l1 = int(imag_np.shape[0]*0.3)
    l2 = int(imag_np.shape[1]*0.7)
    l3 = int(imag_np.shape[1]*0.1)
    imag_np = imag_np[l3:imag_np.shape[0] - l2,l1:imag_np.shape[1]-l1 ,: ]
    plt.imshow(imag_np)
    image3 =imag_np.reshape((imag_np.shape[0] * imag_np.shape[1],3))
    a = 0
    b = 0
    c = 0
    for j in range(5):
       clt = KMeans(n_clusters = 7)
       clt.fit(image3)
       a += clt.cluster_centers_[0][0]
       b += clt.cluster_centers_[0][1]
       c += clt.cluster_centers_[0][2]
    RGB_data = [a / 5. , b / 5., c / 5.]
    return RGB_data
   
# 2) (크롭위치 : 가운데) , 군집수 5인 색깔 함수
# 3) (크롭위치 : 가운데) , 군집수 1인(줄무늬, 체크무늬를 위한) 색깔 함수             
def cluster_color():
   # bgr -> rgb
    image = Image.open(image_dir).resize((130,130))
    imag_np = np.array(image)
    l1 = int(imag_np.shape[0]*0.3)
    l2 = int(imag_np.shape[1]*0.3)
    imag_np = imag_np[l2:imag_np.shape[0] - l2,l1:imag_np.shape[1]-l1 ,: ]
    plt.imshow(imag_np)
    image3 =imag_np.reshape((imag_np.shape[0] * imag_np.shape[1],3))
    a = 0
    b = 0
    c = 0
    for j in range(5):
       clt = KMeans(n_clusters = 7)
       clt.fit(image3)
       a += clt.cluster_centers_[0][0]
       b += clt.cluster_centers_[0][1]
       c += clt.cluster_centers_[0][2]
    RGB_data = [a / 5. , b / 5., c / 5.]
    return RGB_data
    
##############################################################################################
####### 1. read csv
##############################################################################################
link = pd.read_csv("C:\\ITWILL\\Final_project\\index_link.csv")#, header=None)#, encoding='ms949')
color = pd.read_csv("C:\\ITWILL\\Final_project\\color_db2.csv")#, header=None, encoding='ms949')# (컬럼명 없음)
category = pd.read_csv("C:\\ITWILL\\Final_project\\index_category_num.csv")#, header=None, encoding='ms949')# (컬럼명 없음)
pattern = pd.read_csv("C:\\ITWILL\\Final_project\\index_pattern_num.csv", header=None)#, encoding='ms949')# (컬럼명 없음)

#link
link.head()
link = link.iloc[:,1]
link = list(link)

#color
color.head()
rgb = color.iloc[:,1:4]
rgb.head()

#category
category.head()
category = category.iloc[:,1]
category = list(category)

# pattern
pattern.head()
pattern = pattern.iloc[:,1]
pattern = list(pattern)

model = keras.models.load_model('category_classifier02.h5')
model2 = keras.models.load_model('pattern_classifier02.h5')

os.getcwd()
image_dir = 'C:\\ITWILL\\Flask_part\\Lookus\\static\\imgsave\\8.jpg'
imag = Image.open(image_dir).resize((224,224))
imag_np = np.array(imag)
imag_np = imag_np / 255.
imag_np = np.expand_dims(imag_np, axis = 0)
input_category_num = model.predict_classes(imag_np)[0]


if input_category_num  == 0 :
    input_pattern_num == 0
elif input_category_num  == 2 :
    imag = Image.open(image_dir).resize((150,150))
    imag_np = np.array(imag)
    shape_size = int(imag_np.shape[0]*0.15)
    #test_crop = image[(shape_size*2):-(shape_size*2),(shape_size*3):-(shape_size*1),:3]
    test_crop = imag_np[(shape_size*3):-(shape_size*1),(shape_size*2):-(shape_size*2),:3]
    test_imag = Image.fromarray(test_crop, 'RGB')
    test_imag = test_imag.resize((75,75))
    test_imag_np = np.array(test_imag)
    test_imag_np = test_imag_np / 255.
    test_imag_np = np.expand_dims(test_imag_np, axis = 0)
    input_pattern = model2.predict_classes(test_imag_np)[0] 
else :
            imag = Image.open(image_dir).resize((150,150))
            imag_np = np.array(imag)
            shape_size = int(imag_np.shape[0]*0.3)
            # image slice
            test_crop = imag_np[shape_size:-shape_size,shape_size:-shape_size,:3]
            test_imag = Image.fromarray(test_crop, 'RGB')
            test_imag = test_imag.resize((75,75))
            test_imag_np = np.array(test_imag)
            test_imag_np = test_imag_np / 255.
            test_imag_np = np.expand_dims(test_imag_np, axis = 0)
            input_pattern = model2.predict_classes(test_imag_np)[0] 
    imag = Image.open(image_dir).resize((150,150))
    imag_np = np.array(imag)
    shape_size = int(imag_np.shape[0]*0.3)
    # image slice
    test_crop = imag_np[shape_size:-shape_size,shape_size:-shape_size,:3]
    test_imag = Image.fromarray(test_crop, 'RGB')
    test_imag = test_imag.resize((75,75))
    test_imag_np = np.array(test_imag)
    test_imag_np = test_imag_np / 255.
    test_imag_np = np.expand_dims(test_imag_np, axis = 0)
    input_pattern = model2.predict_classes(test_imag_np)[0] 
    
    
image = Image.open(image_dir).resize((130,130))
if input_category_num == 0  :
    input_color = cluster_color_pants(image,5)
    input_color = list(input_color)  
elif input_pattern in [0,3,4] :
    input_color = cluster_color(image,5)
    input_color = list(input_color) 
elif input_pattern in [1,2,5,6] :
    input_color = cluster_color(image,1)
    input_color = list(input_color) 

image = Image.open(image_dir).resize((130,130))
imag_np = np.array(image)
l1 = int(imag_np.shape[0]*0.3)
l2 = int(imag_np.shape[1]*0.7)
l3 = int(imag_np.shape[1]*0.1)
imag_np = imag_np[l3:imag_np.shape[0] - l2,l1:imag_np.shape[1]-l1 ,: ]
plt.imshow(imag_np)
image3 =imag_np.reshape((imag_np.shape[0] * imag_np.shape[1],3))
a = 0
b = 0
c = 0
for j in range(5):
   clt = KMeans(n_clusters = 7)
   clt.fit(image3)
   a += clt.cluster_centers_[0][0]
   b += clt.cluster_centers_[0][1]
   c += clt.cluster_centers_[0][2]
RGB_data = [a / 5. , b / 5., c / 5.]


if input_category_num == 0  :
    input_color = cluster_color_pants()
    input_color = list(input_color)  
elif input_pattern in [0,3,4] :
    input_color = cluster_color()
    input_color = list(input_color) 
elif input_pattern in [1,2,5,6] :
    input_color = cluster_color()
    input_color = list(input_color) 


dist={}
for i in range(1300) : 
    data =list(rgb.iloc[i])
    d = cos_sim(input_color,data)
    dist[d] = i

a = sorted(dist.items(), reverse = True)
category_dist=[]
#category.head()

for _,i in a :
    if category[i] == input_category_num :
        category_dist.append(i)

# len(category_dist)
# category_dist[:5]

# 3. input_3 = input2_pattern_find
final_index=[]

for i in category_dist :
    if pattern[i] == input_pattern :
        final_index.append(i)

final_index[:5]

import random
random.randint(1,9999)