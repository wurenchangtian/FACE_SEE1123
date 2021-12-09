# 训练模型
from __future__ import print_function

import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import utils as np_utils
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from faceout import extract_data, resize_with_pad, IMAGE_SIZE


path_NUM='./face_data/'
peo_num = len(os.listdir(path_NUM))
# 定义数据集类


class Dataset(object):
    def __init__(self, path_name):         # 定义构造函数，包括了训练集，验证集，测试集
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.Y_train = None
        self.Y_valid = None
        self.Y_test = None

        self.path_name = path_name

        self.user_num = len(os.listdir(path_name))

    # 定义数据读取函数
    def read(self, img_rows=IMAGE_SIZE, img_cols=IMAGE_SIZE, img_channels=3):
        images, labels = extract_data('./face_data/')    # 数据的路径
        nb_classes = self.user_num
        labels = np.reshape(labels, [-1])           # 随机分割数据集
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=random.randint(0,100))
        X_valid, X_test, y_valid, y_test = train_test_split(images, labels, test_size=0.5, random_state=random.randint(0,100))

        if K.image_data_format() == 'channels_first':           # 这是theano的数据格式
            X_train = X_train.reshape(X_train.shape[0], 3, img_rows, img_cols)
            X_valid = X_valid.reshape(X_valid.shape[0], 3, img_rows, img_cols)
            X_test = X_test.reshape(X_test.shape[0], 3, img_rows, img_cols)
            input_shape = (3, img_rows, img_cols)
        else:
            X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 3)
            X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 3)
            X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 3)
            input_shape = (img_rows, img_cols, 3)

        # 数据集随机分为测试和训练集
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples(训练样本)')
        print(X_valid.shape[0], 'valid samples(验证样本)')
        print(X_test.shape[0], 'test samples(测试样本)')
        # 类别标签进行one-hot编码使其向量化，在这里我们的类别只有两种，经过转化后标签数据变为二维
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_valid = np_utils.to_categorical(y_valid, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)

        X_train = X_train.astype('float32')
        X_valid = X_valid.astype('float32')
        X_test = X_test.astype('float32')

        X_train /= 255
        X_valid /= 255
        X_test /= 255

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.Y_train = Y_train
        self.Y_valid = Y_valid
        self.Y_test = Y_test

# 定义卷积神经网络模型


class Model(object):

    def __init__(self):
        self.model = None

    def build_model(self, dataset, nb_classes=peo_num):   # 构建网络,4个分类

        self.model = Sequential()                   # 添加卷积层,32个3*3卷积核，pad=1
        self.model.add(Convolution2D(32, 3, 3, padding='same', input_shape=dataset.X_train.shape[1:]))
        self.model.add(Activation('relu'))          # 激活函数relu
        self.model.add(Convolution2D(32, 3, 3, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))      # 最大池化2*2模板
        self.model.add(Dropout(0.25))               # dropout层，比例为0.25
        self.model.add(Convolution2D(64, 3, 3, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(Convolution2D(64, 3, 3, padding='same'))
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))  # 最大池化2*2模板
        self.model.add(Dropout(0.25))
        self.model.add(Flatten())                   # 拉直
        self.model.add(Dense(512))                  # 全连接层512个神经元
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(nb_classes))
        self.model.add(Activation('softmax'))       # 最后用softmax进行分类
        self.model.summary()

# 定义训练函数


    def train(self, dataset, batch_size=32, nb_epochs=40, data_augmentation=True):
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # 采用交叉熵以及SGD进行网络训练，学习率定位0.01
        if not data_augmentation:
            print('没有使用数据扩充')
            self.model.fit(dataset.X_train, dataset.Y_train, batch_size=batch_size, nb_epoch=nb_epochs,
                       validation_data=(dataset.X_valid, dataset.Y_valid),shuffle=True)
        else:
            print('使用数据扩充')
            datagen = ImageDataGenerator(
                featurewise_center=False,           # 数据得均值
                samplewise_center=False,
                featurewise_std_normalization=False,    # 用方差进行划分
                samplewise_std_normalization=False,
                zca_whitening=False,                    # ZCA数据增白
                rotation_range=20,                      # 旋转角度
                width_shift_range=0.2,                  # 变换宽比例
                height_shift_range=0.2,
                horizontal_flip=True,                   # 随机翻转
                vertical_flip=False)
            # 计算标准化
            datagen.fit(dataset.X_train)

            self.model.fit_generator(datagen.flow(dataset.X_train, dataset.Y_train,
                                                  batch_size=batch_size),
                                     steps_per_epoch=dataset.X_train.shape[0],
                                     epochs=nb_epochs,
                                     validation_data=(dataset.X_valid, dataset.Y_valid))




# 保存模型函数
    FILE_PATH = './store/face1.h5'

    def save(self, file_path=FILE_PATH):
        self.model.save(file_path)
        print('模型保存完毕')

# 模型载入函数


    def load(self, file_path=FILE_PATH):
        print('模型载入')
        self.model = load_model(file_path)

# 调用模型


    def predict(self, image):
        if K.image_data_format() == 'channels_first' and image.shape != (1, 3, IMAGE_SIZE, IMAGE_SIZE):
            image = resize_with_pad(image)
            image = image.reshape((1, 3, IMAGE_SIZE, IMAGE_SIZE))

        elif K.image_data_format() == 'channels_last' and image.shape != (1, IMAGE_SIZE, IMAGE_SIZE, 3):
            image = resize_with_pad(image)
            image = image.reshape((1, IMAGE_SIZE, IMAGE_SIZE, 3))
        image = image.astype('float32')
        image /= 255
        result = self.model.predict_proba(image)        # 预测
        print(result)
        result = self.model.predict_classes(image)       # 分类
        return result[0]

# 定义评估函数


    def evaluate(self, dataset):
        score = self.model.evaluate(dataset.X_test, dataset.Y_test, verbose=0)
        print("%s: %.2f%%" % (self.model.metrics_names[1], score[1] * 100))



'''

# 执行流程
dataset = Dataset('./face_data/')     # 构建数据集
dataset.read()          # 数据集读取
model = Model()         # 构建模型
print('构建完毕')
model.build_model(dataset)
model.train(dataset, nb_epochs=10)
model.save()
model = Model()
model.load()
model.evaluate(dataset)
'''
