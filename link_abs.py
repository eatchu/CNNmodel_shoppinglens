import os
from os import listdir
from PIL import Image
import numpy as np
os.getcwd()
curos = os.chdir('C:\\Users\\user\\Desktop\\dataset\\outer')
curos2 = os.chdir('C:\\Users\\user\\Desktop\\dataset\\outer2')

files = listdir('C:\\Users\\user\\Desktop\\dataset\\outer')


len(files)
files_link = []
for i in range(len(files)):
   files_link.append(files[i][:-10])
files_list = []
files_list = list(map(int, files_link))

files_list.sort()
files_list = list(map(str, files_list))

url_link = []
url_link2 = []
for i in range(len(files_list)):
   url_link.append( 'https://store.musinsa.com/app/product/detail/' + files_list[i] + '/0')
url_link
len(url_link)
# url_link 에 오름차순으로 정렬완료
image_id=[]
image_RGB=[]
image_id2=[]
image_RGB2=[]
for i, f in enumerate(files):
    img = Image.open(f)
    img = np.array(img)
    image_RGB.append(img)
    image_id.append(i + 401)

image_id
len(image_RGB)
url_link
image_id2
url_link2