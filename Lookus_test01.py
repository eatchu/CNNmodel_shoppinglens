from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import time

os.chdir('C:\\ITWILL\\Flask_part\\Lookus')
os.getcwd()
check = False
app = Flask(__name__)


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

result_idx = []

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
def cluster_color_pants(image_dir):
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
def cluster_color(image_dir):
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
link = pd.read_csv("C:\\ITWILL\\Final_project\\item_link.csv")#, header=None)#, encoding='ms949')
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
import random

@app.route('/')
def index():
    return render_template('/index.html')
@app.route('/detail1')
def detail1():
    return render_template('/detail1.html')
@app.route('/detail2')
def detail2():
    return render_template('/detail2.html')
@app.route('/detail3')
def detail3():
    return render_template('/detail3.html')
@app.route('/detail4')
def detail4():
    return render_template('/detail4.html')
@app.route('/detail5')
def detail5():
    return render_template('/detail5.html')

@app.route('/detail6')
def detail6():
    return render_template('/detail6.html')

@app.route('/detail7')
def detail7():
    return render_template('/detail7.html')

@app.route('/detail8')
def detail8():
    return render_template('/detail8.html')

@app.route('/detail9')
def detail9():
    return render_template('/detail9.html')

@app.route('/detail10')
def detail10():
    return render_template('/detail10.html')

@app.route('/detail11')
def detail11():
    return render_template('/detail11.html')

@app.route('/detail12')
def detail12():
    return render_template('/detail12.html')

@app.route('/detail13')
def detail13():
    return render_template('/detail13.html')

@app.route('/detail14')
def detail14():
    return render_template('/detail14.html')

@app.route('/detail15')
def detail15():
    return render_template('/detail15.html')

@app.route('/detail16')
def detail16():
    return render_template('/detail16.html')

@app.route('/detail17')
def detail17():
    return render_template('/detail17.html')

@app.route('/fileUpload', methods = ['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']
        global num
        num = random.randint(1,9999)
        f.save('C:\\ITWILL\\Flask_part\\Lookus\\static\\imgsave\\' + secure_filename(str(num) + '.jpg'))
        imag = Image.open('C:\\ITWILL\\Flask_part\\Lookus\\static\\imgsave\\' + str(num) + '.jpg').resize((224,224))
        imag_np = np.array(imag)
        imag_np = imag_np / 255.
        imag_np = np.expand_dims(imag_np, axis = 0)
        input_category_num = model.predict_classes(imag_np)[0]                  
        image_dir = 'C:\\ITWILL\\Flask_part\\Lookus\\static\\imgsave\\' + str(num) + '.jpg'                                 
        if input_category_num  == 0 :
            input_pattern = 0
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
        if input_category_num == 0  :
            input_color = cluster_color_pants(image_dir)
            input_color = list(input_color)  
        elif input_pattern in [0,3,4] :
            input_color = cluster_color(image_dir)
            input_color = list(input_color) 
        elif input_pattern in [1,2,5,6] :
            input_color = cluster_color(image_dir)
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
        global item1
        item1 = final_index[0]  
        global item2
        item2 = final_index[1] 
        global item3
        item3 = final_index[2] 
        global item4
        item4 = final_index[3] 
        global item5
        item5 = final_index[4] 
        global item6
        item6 = final_index[5] 
        global link1
        link1 = link[item1]
        global link2
        link2 = link[item2]
        global link3
        link3 = link[item3]
        global link4
        link4 = link[item4]
        global link5
        link5 = link[item5]
        global link6
        link6 = link[item6]
        return render_template('/processing.html')
        # return render_template('/result.html' , num = num, item1 = result_idx[0] , item2 = result_idx[1], item3 = result_idx[2], item4 = result_idx[3], item5 = result_idx[4])

@app.errorhandler(500) 
def page_not_found(error): 
    return render_template('error_handle.html'), 500

@app.route('/result')
def result():
    print(result_idx)
    return render_template('/result.html' , num = num, item1 = item1 , item2 = item2, item3 = item3, item4 = item4, item5 = item5, item6 = item6, link1 = link1 , link2 = link2, link3 = link3 , link4 = link4 , link5 =link5, link6 = link6)
    # return render_template('/result.html' , num = num)

@app.route('/intro') 
def intro(): 
    return render_template('/intro.html')


if __name__ == "__main__":
    app.run("192.168.12.21")

link[800]