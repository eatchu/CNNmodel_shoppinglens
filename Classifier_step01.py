# -*- coding: utf-8 -*-
"""
- celeb image classifier
- imageDataGenerator 클래스 이용
"""
from tensorflow.keras import Sequential  # keras model
from tensorflow.keras.layers import Conv2D, MaxPool2D  # Convolution layer
from tensorflow.keras.layers import Dense, Flatten  # Affine layer
from tensorflow.keras.layers import Dropout
import os

# dir setting
base_dir = "C:\\Users\\user\\Desktop\\dataset\\data_set"

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')

# Hyper parameters
img_h = 224  # height
img_w = 224  # width
input_shape = (img_h, img_w, 3)

# 1. CNN Model layer
print('model create')
model = Sequential()

# Convolution layer1
model.add(Conv2D(96, kernel_size=(11, 11), activation='relu', strides=4, padding='same',
                 input_shape=input_shape))
model.add(MaxPool2D(pool_size=(3, 3) , strides=2, padding='valid'))
# Convolution layer2
model.add(Conv2D(256, kernel_size=(5, 5), activation='relu', strides=1, padding='same'))
model.add(MaxPool2D(pool_size=(3, 3) , strides=2, padding='valid'))
# Convolution layer3 : maxpooling() 제외
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(Conv2D(384, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', strides=1, padding='same'))

# Flatten layer : 3d -> 1d
model.add(Flatten())

# DNN hidden layer(Fully connected layer)
model.add(Dense(4096, activation='relu'))
model.add(Dense(1000, activation='relu'))

# DNN Output layer
model.add(Dense(5, activation='softmax'))

# model training set : Adam or RMSprop
model.compile(optimizer='adam',
              # loss = 'binary_crossentropy', # integer(generator가 integer로 읽어옴) + 이항분류
              # loss = 'categorical_crossentropy' # y:원핫인코딩
              loss='sparse_categorical_crossentropy',  # Y=integer + 다항분류
              metrics=['sparse_categorical_accuracy'])

# 2. image file preprocessing : image 제너레이터 이용
from tensorflow.keras.preprocessing.image import ImageDataGenerator

print("image preprocessing")

# 특정 폴더의 이미지를 분류하기 위해서 학습시킬 데이터셋 생성
train_data = ImageDataGenerator(rescale=1. / 255)  # 0~1 정규화

# 검증 데이터
validation_data = ImageDataGenerator(rescale=1. / 255)  # 0~1 정규화

train_generator = train_data.flow_from_directory(
    train_dir,
    target_size=(224, 224),  # image reshape
    batch_size=20,  # batch size
    class_mode='binary')  # binary label
# Found 2000 images belonging to 2 classes.

validation_generator = validation_data.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='binary')
# Found 1000 images belonging to 2 classes.

# 3. model training : image제너레이터 이용 모델 훈련
model_fit = model.fit_generator(
    train_generator,
    steps_per_epoch=40,  # 20(배치사이즈:이미지 공급)* 100(steps 1에폭내에서 반복수)
    epochs=50,
    validation_data=validation_generator,
    validation_steps=10)  # 1000 = 20*50

# 4. model history graph
import matplotlib.pyplot as plt

print(model_fit.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

loss = model_fit.history['loss']  # train
acc = model_fit.history['sparse_categorical_accuracy']
val_loss = model_fit.history['val_loss']  # validation
val_acc = model_fit.history['val_sparse_categorical_accuracy']

epochs = range(1, len(acc) + 1)

# acc vs val_acc
plt.plot(epochs, acc, 'bo', label='train acc')
plt.plot(epochs, val_acc, 'r', label='val acc')
plt.title('Training vs validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuray')
plt.legend(loc='best')
plt.show()

# loss vs val_loss
plt.plot(epochs, loss, 'bo', label='train loss')
plt.plot(epochs, val_loss, 'r', label='val loss')
plt.title('Training vs validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()



