import os
from PIL import Image
import string
import numpy as np

os.getcwd()
os.chdir('C:\\Users\\user\\Desktop\\dataset\\category\\category_pattern')
file = os.listdir('C:\\Users\\user\\Desktop\\dataset\\category\\category_pattern')
file2 = os.listdir('C:\\Users\\user\\Desktop\\dataset\\category\\temp')
os.chdir('C:\\Users\\user\\Desktop\\dataset\\category\\temp')
len(file)
imag = Image.open(file[0])
imag = Image.open(file[22]).convert('RGB')
imag.show()
imag_np = np.array(imag)
imag_np.shape
imag_np2 = np.expand_dims(imag_np, 2)
imag_np2.shape
imag = Image.fromarray(imag_np2, 'RGB')

imag_np = imag_np[120:380, 100:400, :]
imag = Image.fromarray(imag_np, 'L')
imag.save('new_image.jpg')
imag.show()
imag_np = imag_np[40:110, 30:95, :]

for i, f in enumerate(file):
    imag = Image.open(f).convert('L')
    imag_np = np.array(imag)
    imag_np = imag_np[120:380, 100:400]
    imag = Image.fromarray(imag_np, 'L')
    imag.save(str(i) + '.jpg')

file.sort()
int(file)

for i, f in enumerate(file):
    imag = Image.open(f)
    imag.save(str(i) + '.jpg')
a = file[0].lstrip('[포맷변환]400 ')
a = a.rstrip('.jpg')
a = str(a)
a.lstrip('(')

file[0]
number = []

os.chdir('C:\\Users\\user\\Desktop\\dataset\\category\\temp2')
file = os.listdir('C:\\Users\\user\\Desktop\\dataset\\category\\temp2')

# 숫자만 추출하여 index 행으로 사용
number = []
for f in file2:
    f = f.lower()
    f = f.rstrip('.jpg')
    a = int(f)
    number.append(a)
# dataframe 생성, file 이름에 index 부여 후 정렬
import pandas as pd

df_file = pd.DataFrame(file2, number)
df_file = df_file.sort_index()
df_file
file_sort = np.array(df_file[0])

for i, f in enumerate(file_sort):
    imag = Image.open(f).convert('L')
    imag = imag.convert('RGB')
    imag_np = np.array(imag)
    imag = Image.fromarray(imag_np, 'RGB')
    imag.save(str(i + 200) + '.jpg')

os.chdir('C:\\Users\\user\\Desktop\\dataset\\category\\temp2')
file = os.listdir('C:\\Users\\user\\Desktop\\dataset\\category\\temp2')
number = []
for f in file:
    f = f.lstrip('J')
    f = f.rstrip('.jpg')
    a = int(f)
    number.append(a)
# dataframe 생성, file 이름에 index 부여 후 정렬
import pandas as pd

df_file = pd.DataFrame(file, number)
df_file = df_file.sort_index()
df_file
file_sort = np.array(df_file[0])

for i, f in enumerate(file_sort):
    imag = Image.open(f).convert('RGB')
    imag_np = np.array(imag)
    imag = Image.fromarray(imag_np, 'RGB')
    imag.save(str(i + 600) + '.jpg')

import cv2
