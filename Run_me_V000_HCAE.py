#import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from pretty_confusion_matrix import pp_matrix
import copy
import tensorflow as tf
import cv2
from tensorflow import keras
from matplotlib import pyplot as plt
import scipy.io
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import mat73
from datetime import datetime
from sklearn import preprocessing
##region networks

class Encod_2(tf.keras.Model):
    def __init__(self, filters_1x1=15, Dropout_rate=0.2):
        super(Encod_2, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters_1x1, (5, 5), padding='same')
        self.conv1x1_2 = tf.keras.layers.Conv2D(filters_1x1, (5, 5), padding='same')
        self.max_pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2), padding='same')
        self.act = tf.keras.layers.Activation('relu')
        self.dropout = tf.keras.layers.Dropout(Dropout_rate)
        self.norm = tf.keras.layers.BatchNormalization()
        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        b1 = self.conv1x1(inputs)
        b1_norm = self.norm(b1)
        b1_act = self.act(b1_norm)
        b1_droped = self.dropout(b1_act)
        b2 = self.conv1x1_2(b1_droped)
        b2_norm = self.norm2(b2)
        b2_act = self.act(b2_norm)
        b3 = self.max_pool2(b2_act)
        return b3


class endi(tf.keras.Model):
    def __init__(self, numb_labels=8):
        super(endi, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(numb_labels, (2, 2), padding='same', activation='sigmoid')

    def call(self, inputs):
        lout = self.conv1x1(inputs)
        return lout


class Same_net(tf.keras.Model):
    def __init__(self,
                 filters_1x1=10,
                 filters_3x3_reduce=10,
                 filters_3x3=10,
                 filters_5x5_reduce=10,
                 filters_5x5=10,
                 filters_pool_proj=32):
        super(Same_net, self).__init__()
        self.conv1x1 = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same')

        self.conv3x3_reduce = tf.keras.layers.Conv2D(filters_3x3_reduce, (1, 1), padding='same')
        self.conv3x3 = tf.keras.layers.Conv2D(filters_3x3, (3, 3), padding='same')

        self.conv5x5_reduce = tf.keras.layers.Conv2D(filters_5x5_reduce, (1, 1), padding='same')
        self.conv5x5 = tf.keras.layers.Conv2D(filters_5x5, (5, 5), padding='same')

        self.convpool = tf.keras.layers.Conv2D(filters_1x1, (1, 1), padding='same')

        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3, 3), strides=(1, 1), padding='same')
        self.Concat = tf.keras.layers.Concatenate()

    def call(self, inputs):
        b1 = self.conv1x1(inputs)
        b1_act = self.act(b1)

        b2 = self.conv3x3_reduce(inputs)
        b2_act = self.act(b2)
        b2_2 = self.conv3x3(b2_act)
        b2_2_act = self.act(b2_2)

        b3 = self.conv5x5_reduce(inputs)
        b3_act = self.act(b3)
        b3_2 = self.conv5x5(b3_act)
        b3_2_act = self.act(b3_2)

        b4 = self.max_pool(inputs)
        b4_2 = self.convpool(b4)
        b4_2_act = self.act(b4_2)

        output = self.Concat([b1_act, b2_2_act, b3_2_act, b4_2_act])
        return output


class CustomDense(tf.keras.Model):
    def __init__(self, filters_1x1=16):
        super(CustomDense, self).__init__()

    def call(self, inputs):
        output = K.softmax(inputs, axis=-1)
        return output


class CustomDense_sig(tf.keras.Model):
    def __init__(self, filters_1x1=16):
        super(CustomDense_sig, self).__init__()

    def call(self, inputs):
        output = K.sigmoid(inputs)
        return output


class Decoder(tf.keras.Model):
    def __init__(self, filters_1x1=12, same_pad=0, up_sample=0, kernels=(2, 2)):
        super(Decoder, self).__init__()
        if (same_pad == 1):
            self.conv1x1 = tf.keras.layers.Conv2DTranspose(filters_1x1, kernels, padding='same')
            self.conv1x1_2 = tf.keras.layers.Conv2DTranspose(filters_1x1, kernels, padding='same')
        elif (same_pad == 0):
            self.conv1x1 = tf.keras.layers.Conv2DTranspose(filters_1x1, kernels, padding='valid')
            self.conv1x1_2 = tf.keras.layers.Conv2DTranspose(filters_1x1, kernels, padding='valid')
        else:
            self.conv1x1 = tf.keras.layers.Conv2D(filters_1x1, kernels)
            self.conv1x1_2 = tf.keras.layers.Conv2DTranspose(filters_1x1, kernels, padding='same')
        if up_sample:
            self.UpSampling2D = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        else:
            self.UpSampling2D = tf.keras.layers.UpSampling2D((1, 1), interpolation='bilinear')

        self.act = tf.keras.layers.Activation('sigmoid')


        self.conv_reduce = tf.keras.layers.Conv2D(filters_1x1, (3, 3))

    def call(self, inputs):
        b1 = self.conv1x1(inputs)
        b1_act = self.act(b1)
        b2 = self.conv1x1_2(b1_act)
        b2_act = self.act(b2)
        b3 = self.UpSampling2D(b2_act)
        return b3



class Decod_comp(tf.keras.Model):
    def __init__(self, filters_1x1=8):
        super(Decod_comp, self).__init__()
        self.b1 = Same_net()
        self.b12 = Same_net()
        self.b13 = Same_net()
        self.b14 = Same_net()
        self.b15 = Same_net()

        self.b2 = Decoder(same_pad=1, up_sample=1)
        self.b22 = Decoder(same_pad=1, up_sample=1)
        self.b23 = Decoder(same_pad=1, up_sample=1)
        self.b24 = Decoder(same_pad=1, up_sample=1)
        self.b25 = Decoder(filters_1x1=1, same_pad=1, up_sample=1)

    def call(self, inputs):
        l1 = self.b1(inputs)
        l2 = self.b2(l1)
        l3 = self.b12(l2)
        l4 = self.b22(l3)
        l5 = self.b13(l4)
        l6 = self.b23(l5)
        l7 = self.b14(l6)
        l8 = self.b24(l7)
        l9 = self.b15(l8)
        l10 = self.b25(l9)

        return l10


class Encod_comp(tf.keras.Model):
    def __init__(self, numb_labels=8):
        super(Encod_comp, self).__init__()
        self.b1 = Same_net()
        self.b12 = Same_net()
        self.b13 = Same_net()
        self.b14 = Same_net()
        self.b15 = Same_net()
        self.b16 = Same_net()

        self.b2 = Encod_2()
        self.b22 = Encod_2()
        self.b23 = Encod_2()
        self.b24 = Encod_2()
        self.b25 = Encod_2()

    def call(self, inputs):
        l1 = self.b1(inputs)
        l2 = self.b2(l1)
        l3 = self.b12(l2)
        l4 = self.b22(l3)
        l5 = self.b13(l4)
        l6 = self.b23(l5)
        l7 = self.b14(l6)
        l8 = self.b24(l7)
        l9 = self.b15(l8)
        l10 = self.b25(l9)
        l11 = self.b16(l10)
        return l11


class labels_comp(tf.keras.Model):
    def __init__(self, numb_labels=8):
        super(labels_comp, self).__init__()

        self.b1 = Same_net()
        #self.b2 = Encod_2()
        self.out = endi(numb_labels=numb_labels)
        self.outact = CustomDense()
        self.outact_sig = CustomDense_sig()

    def call(self, inputs):
        #l1 = self.b2(inputs)
        l2 = self.b1(inputs)
        lout = self.out(l2)
        #loutact = self.outact_sig(lout)

        return lout


class HCAE_comp(tf.keras.Model):
    def __init__(self, numb_labels=3):
        super(HCAE_comp, self).__init__()
        self.encoding = Encod_comp()
        self.labeler = labels_comp(numb_labels=numb_labels)
        self.decoding = Decod_comp()

    def call(self, inputs):
        l1 = self.encoding(inputs)
        lout_labels = self.labeler(l1)
        lout_AE = self.decoding(l1)

        return [lout_AE, lout_labels]

## endregion
##region data proccesing
IMG_HEIGHT = 288
IMG_WIDTH = 288
Numb_labels = 4
Numb_grids = 9
filepath = 'D:\Javid\Paper_V2\Matlab\From_other_PC\dataset2.mat'


data_dict = mat73.loadmat(filepath)
a = data_dict['dataset']
part_num = a['part_num']
layer_num = a['layer_num']
Labeled = a['Labeled']
labels = a['Labels']
img = a['img_uncolor']

#data_dict = scipy.io.loadmat('C:\Javid\Paper_V2\Matlab\data_2.mat')
#a = data_dict['dataset']
#part_num = a[0]['part_num'][()][:]
#layer_num = a[0]['layer_num'][()][:]
#Labeled = a[0]['Labeled'][()][:]
#labels = a[0]['Labels'][()][:]
#img = a[0]['img_uncolor'][()][:]

# img = np.array(img)
img2 = []
labels2 = []
labels3 = []
name2 = []
img3 = []
name3 = []
for i in range(Labeled.__len__()):
    if Labeled[i] == 1:
        labels2.append(labels[i])
        img2.append(img[i])
        name2.append("P{0}_L{1}".format(part_num[i], layer_num[i]))
    elif Labeled[i] != 2:
        img3.append(img[i])
        labels3.append(np.zeros([Numb_grids, Numb_grids, Numb_labels]))
        name3.append("P{0}_L{1}".format(part_num[i], layer_num[i]))

        # print(Labeled[i][0][0])
        # labels2 = np.stack(labels[i], axis=0)

#for i in range(Labeled.shape[0]):
#    if Labeled[i][0][0] == 1:
#        labels2.append(labels[i])
#        img2.append(img[i])
#        name2.append("P{0}_L{1}".format(part_num[i][0], layer_num[i][0][0]))
#    elif Labeled[i] == 0:
#        img3.append(img[i])
#        labels3.append(np.zeros([Numb_grids,Numb_grids,Numb_labels]))
#        name3.append("P{0}_L{1}".format(part_num[i][0], layer_num[i][0][0]))
#labels2 = np.stack(labels2, axis=0)
#labels3 = np.stack(labels3, axis=0)

def single_multply(a, flag=1):
    if flag:
        a = a.numpy()
    temp = [[]]
    temp.append(a)
    temp.append(a[:, :, ::-1, :])
    temp.append(a[:, ::-1, :, :])
    temp.append(np.rot90(a[:, ::-1, :, :], 1, (2, 1)))
    temp.append(np.rot90(a, 1, (1, 2))[:, :, ::-1, :])
    temp.append(np.rot90(a, 1, (1, 2)))
    temp.append(np.rot90(temp[-1], 1, (1, 2)))
    temp.append(np.rot90(temp[-1], 1, (1, 2)))
    del temp[0]
    return temp

def img_proc(img_in, labels_, IMG_HEIGHT_, IMG_WIDTH_, Numb_labels = 8, Numb_grids = 9):
    img_data_array = []
    for i_ in range(len(img_in)):
        image_def = img_in[i_]
        image_def = cv2.resize(image_def, (IMG_HEIGHT_, IMG_WIDTH_), interpolation=cv2.INTER_AREA)
        image_def = np.array(image_def)
        image_def = image_def.astype('float16')
        #    image /= 255
        img_data_array.append(image_def)
    Full_imgages_ = []
    Full_labels_ = []
    try:
        if  (labels_.shape).__len__() == 4:
            labels_ = np.zeros([img_in.__len__(),Numb_grids,Numb_grids,Numb_labels])
    except:
        if labels_.__len__() == 4:
            labels_ = np.zeros([img_in.__len__(), Numb_grids, Numb_grids, Numb_labels])

    for iter_ in range(img_data_array.__len__()):
        ## original image
        Full_imgages_.append(img_data_array[iter_])
        Full_labels_.append(labels_[iter_])
        ## Left Right
        temp_Img = np.fliplr(img_data_array[iter_])
        temp_Lab = np.fliplr(labels_[iter_])
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)
        ## Up Down
        temp_Img = np.flipud(img_data_array[iter_])
        temp_Lab = np.flipud(labels_[iter_])
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)
        ## Diag normal
        temp_Img = np.rot90(np.flipud(img_data_array[iter_]))
        temp_Lab = np.rot90(np.flipud(labels_[iter_]))
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)
        ## Diag Inverse
        temp_Img = np.fliplr(np.rot90(img_data_array[iter_]))
        temp_Lab = np.fliplr(np.rot90(labels_[iter_]))
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)
        ## Rot -90
        temp_Img = np.rot90(img_data_array[iter_])
        temp_Lab = np.rot90(labels_[iter_])
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)
        ## Rot -180
        temp_Img = np.rot90(temp_Img)
        temp_Lab = np.rot90(temp_Lab)
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)
        ## Rot -270
        temp_Img = np.rot90(temp_Img)
        temp_Lab = np.rot90(temp_Lab)
        Full_imgages_.append(temp_Img)
        Full_labels_.append(temp_Lab)

    return [Full_imgages_, Full_labels_]


[img2, labels2] = img_proc(img2, labels2, IMG_HEIGHT, IMG_WIDTH)
img2 = np.stack(img2, axis=0)
img2 = img2.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
labels2 = np.stack(labels2, axis=0)
[img3, labels3]= img_proc(img3, labels3, IMG_HEIGHT, IMG_WIDTH)
img3 = np.stack(img3, axis=0)
img3 = img3.reshape(-1, IMG_HEIGHT, IMG_WIDTH, 1)
labels3 = np.stack(labels3, axis=0)

img2 = (img2+1)/2
img3 = (img3+1)/2

# img_data = np.array(img_data_array)
# img_data = np.stack(Full_imgages, axis=0)
# labels2 = np.stack(Full_labels, axis=0)
del data_dict
del a
del labels
del img


label_encoder = LabelEncoder()


def onehotesh(c):
    # define example
    data = c.T
    values = array(data)
    values = values.reshape(values.shape[0], )
    # print(values)
    # integer encode
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    # print(inverted)
    return onehot_encoded


def inverse_coding(d, Numb_grids_, Numb_labels_):
    output = []
    if len(d.shape) == 4:
        for iter in range(len(d)):
            e = d[iter]
            f = e.reshape(Numb_grids_ * Numb_grids_, Numb_labels_)
            inverted = []
            for iii in range(Numb_grids_ * Numb_grids_):
                inverted.append(label_encoder.inverse_transform([argmax(f[iii, :])]))
            output.append(np.array(inverted).reshape(Numb_grids_, Numb_grids_))
    else:
        e = d
        f = e.reshape(Numb_grids_ * Numb_grids_, Numb_labels_)
        inverted = []
        for i in range(Numb_grids_ * Numb_grids_):
            inverted.append(label_encoder.inverse_transform([argmax(f[i, :])]))
        output.append(np.array(inverted).reshape(Numb_grids_, Numb_grids_))
    return output


c = labels2.reshape(1, -1)
onehot_encoded = onehotesh(c)
del c
labels2_onehot = onehot_encoded.reshape(-1, Numb_grids, Numb_grids, Numb_labels)
g = inverse_coding(labels2_onehot[0], Numb_grids, Numb_labels)
if np.all(g == labels2[0]):
    print("eyval")
del g


## endregion data Augmentation
##region functions
def f1(C):
    num_classes = np.shape(C)[0]
    f1_score = np.zeros(shape=(num_classes,), dtype='float32')
    f1_score_w = np.zeros(shape=(num_classes,), dtype='float32')

    weights = np.sum(C, axis=0) / np.sum(C)

    for j in range(num_classes):
        tp = np.sum(C[j, j])
        fp = np.sum(C[j, np.concatenate((np.arange(0, j), np.arange(j + 1, num_classes)))])
        fn = np.sum(C[np.concatenate((np.arange(0, j), np.arange(j + 1, num_classes))), j])
        #         tn = np.sum(C[np.concatenate((np.arange(0, j), np.arange(j+1, num_classes))), np.concatenate((np.arange(0, j), np.arange(j+1, num_classes)))])

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score_w[j] = 2 * precision * recall / (precision + recall) * weights[j] if (precision + recall) > 0 else 0
        f1_score[j] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # f1_score = np.sum(f1_score)
    return f1_score, np.sum(f1_score_w)
def integerization(img__):
    return (img__*255).reshape(288,288).astype('int32')
def my_resize(img,size,mode_img=0):
    L = np.floor(size/img.shape[0]).astype('int32')
    temp = np.zeros([size, size, img.shape[-1]+1])
    for step_y in range(1, 1+img.shape[0]):
        for step_x in range(1, 1+img.shape[0]):
            start_x = L * (step_x - 1)
            end_x = L * step_x
            start_y = L * (step_y - 1)
            end_y = L * step_y
            temp[start_x:end_x, start_y:end_y, 0:3] = img[step_x-1,step_y-1,0:3]
    if mode_img == 2:
        temp[:, :, 3] = temp[:, :, 0]
        temp[:, :, 0] = temp[:, :, 2]
        temp[:, :, 2] = temp[:, :, 3]
    return temp[:, :, 0:3]
def label_floorer(pred_img_,upper_ther = 0.40375,lower_ther = 0.2):
    temp = copy.deepcopy(pred_img_)
    a = np.sort(pred_img_,axis=-1)[:,:,:,-2]
    a=np.stack([a,a,a,a],axis=-1)
    temp = temp * (temp >= a).astype(int) # taking only max 2 elements
    temp = temp.reshape(-1,1) # making it linear

    temp[temp>=upper_ther] = 1
    temp[temp<=lower_ther] = 2
    temp[temp<0.85 ] = .5
    temp[temp==2 ] = 0
    temp = temp.reshape(-1, 4)
    sum_temp = np.sum(temp,axis=-1)
    temp[sum_temp == 0] = [0,0,0,1.] # trying to eliminate all zeros

    temp =  temp.reshape(pred_img_.shape)
    er = temp[:,:,:,3] == 0.5
    temp[:][:][:][er] = 0
    a = np.sort(temp,axis=-1)[:,:,:,-1]
    a=np.stack([a,a,a,a],axis=-1)
    temp = temp * (temp >= a).astype(int)
    a = np.sum(temp,axis=-1)
    b = temp*0+1
    b[a==0.5] = 2
    temp = temp*b
    return temp
def softmax(x):
    max = np.max(x, axis=-1, keepdims=True)  # returns max of each row and keeps same dims
    e_x = np.exp(x - max)  # subtracts each row with its max value
    sum = np.sum(e_x, axis=-1, keepdims=True)  # returns sum of each row and keeps same dims
    f_x = e_x / sum
    return f_x
def my_confusion(y_true,y_pred,n_labels=None,partial=0):


    u_labels = np.unique(y_true)
    n_labels = u_labels.shape[0]
    mat = np.zeros([n_labels,n_labels])
    for iter in range(y_true.shape[0]):
        if y_true[iter] == y_pred[iter]:
            mat[y_true[iter],y_true[iter]] += 1
        else:
            if partial == 1:
                #temp[np.all(labels == [W],axis=-1)] = 0
                #temp[np.all(labels == [Bl],axis=-1)] = 0
                #temp[np.all(labels == [R],axis=-1)] = 1
                #temp[np.all(labels == [G],axis=-1)] = 2
                #temp[np.all(labels == [B],axis=-1)] = 3
                #temp[np.all(labels == [RB],axis=-1)] = 4 (1-3)
                #temp[np.all(labels == [GB],axis=-1)] = 5 (2-3)
                #temp[np.all(labels == [RG],axis=-1)] = 6 (1-2)
                if y_true[iter]==1 and (y_pred[iter]==4 or y_pred[iter]==6):
                    mat[int(y_true[iter]), int(y_true[iter])] += 0.75
                    mat[int(y_true[iter]), int(y_pred[iter])] += 0.25
                elif y_true[iter]==2 and (y_pred[iter]==5 or y_pred[iter]==6):
                    mat[int(y_true[iter]), int(y_true[iter])] += 0.75
                    mat[int(y_true[iter]), int(y_pred[iter])] += 0.25
                elif y_true[iter]==3 and (y_pred[iter]==5 or y_pred[iter]==4):
                    mat[int(y_true[iter]), int(y_true[iter])] += 0.75
                    mat[int(y_true[iter]), int(y_pred[iter])] += 0.25
                elif y_true[iter]==4 and (y_pred[iter]==1 or y_pred[iter]==3):
                    #mat[int(y_true[iter]), int(y_true[iter])] += 1

                    mat[int(y_true[iter]), int(y_true[iter])] += 0.75
                    mat[int(y_true[iter]), int(1 if y_pred[iter] == 3 else 3)] += 0.25
                elif y_true[iter]==4 and (y_pred[iter]==5 or y_pred[iter]==6):
                    #mat[int(y_true[iter]), int(y_true[iter])] += 0.5
                    #mat[int(y_true[iter]), int(y_pred[iter])] += 0.5
                    mat[int(y_true[iter]), int(y_true[iter])] += 0.5
                    mat[int(y_true[iter]), int(2)] += 0.25
                    mat[int(y_true[iter]), int(y_pred[iter])] += 0.25
                elif y_true[iter]==5 and (y_pred[iter]==2 or y_pred[iter]==3):
                    #at[int(y_true[iter]), int(y_true[iter])] += 1

                    mat[int(y_true[iter]), int(y_true[iter])] += 0.75
                    mat[int(y_true[iter]), int(2 if y_pred[iter] == 3 else 3)] += 0.25
                elif y_true[iter]==5 and (y_pred[iter]==4 or y_pred[iter]==6):
                    #mat[int(y_true[iter]), int(y_true[iter])] += 0.5
                    #mat[int(y_true[iter]), int(y_pred[iter])] += 0.5
                    mat[int(y_true[iter]), int(y_true[iter])] += 0.5
                    mat[int(y_true[iter]), int(1)] += 0.25
                    mat[int(y_true[iter]), int(y_pred[iter])] += 0.25
                elif y_true[iter]==6 and (y_pred[iter]==1 or y_pred[iter]==2):
                    #mat[int(y_true[iter]), int(y_true[iter])] += 1

                    mat[int(y_true[iter]), int(y_true[iter])] += 0.75
                    mat[int(y_true[iter]), int(2 if y_pred[iter] == 1 else 1)] += 0.25
                elif y_true[iter]==6 and (y_pred[iter]==4 or y_pred[iter]==5):
                    #mat[int(y_true[iter]), int(y_true[iter])] += 0.5
                    #mat[int(y_true[iter]), int(y_pred[iter])] += 0.5
                    mat[int(y_true[iter]), int(y_true[iter])] += 0.5
                    mat[int(y_true[iter]), int(3)] += 0.25
                    mat[int(y_true[iter]), int(y_pred[iter])] += 0.25
                else:
                    mat[int(y_true[iter]), int(y_pred[iter])] += 1
            elif partial == 0:
                if y_true[iter]==1 and (y_pred[iter]==4 or y_pred[iter]==6):
                    mat[int(y_true[iter]), int(y_true[iter])] += 1
                elif y_true[iter]==2 and (y_pred[iter]==5 or y_pred[iter]==6):
                    mat[int(y_true[iter]), int(y_true[iter])] += 1
                elif y_true[iter]==3 and (y_pred[iter]==5 or y_pred[iter]==4):
                    mat[int(y_true[iter]), int(y_true[iter])] += 1
                elif y_true[iter]==4 and (y_pred[iter]==1 or y_pred[iter]==3 or y_pred[iter]==5 or y_pred[iter]==6):
                    mat[int(y_true[iter]), int(y_true[iter])] += 1
                elif y_true[iter]==5 and (y_pred[iter]==2 or y_pred[iter]==3 or y_pred[iter]==6 or y_pred[iter]==4):
                    mat[int(y_true[iter]), int(y_true[iter])] += 1
                elif y_true[iter]==6 and (y_pred[iter]==1 or y_pred[iter]==2 or y_pred[iter]==5 or y_pred[iter]==4):
                    mat[int(y_true[iter]), int(y_true[iter])] += 1
                else:
                    mat[int(y_true[iter]), int(y_pred[iter])] += 1
    return mat
def label_solidify(labels):
    R = [0, 0, 1., 0]
    G = [0, 1., 0, 0]
    B = [1., 0, 0, 0]
    RB = [0.5, 0, 0.5, 0]
    GB = [0.5, 0.5, 0, 0]
    RG = [0, 0.5, 0.5, 0]
    W = [0, 0, 0, 1]
    Bl = [0, 0, 0, 0.]

    labels = labels.reshape(-1,4)
    temp = np.zeros(labels.shape[0])
    temp[np.all(labels == [R],axis=-1)] = 1
    temp[np.all(labels == [G],axis=-1)] = 2
    temp[np.all(labels == [B],axis=-1)] = 3
    temp[np.all(labels == [RB],axis=-1)] = 4
    temp[np.all(labels == [GB],axis=-1)] = 5
    temp[np.all(labels == [RG],axis=-1)] = 6
    temp[np.all(labels == [W],axis=-1)] = 0
    temp[np.all(labels == [Bl],axis=-1)] = 0
    return temp
## endregion
##region setup network
def plothist(history):
    plt.figure(100)
    plt.ion()
    plt.plot(history.history['output_2_accuracy'], label='output_2_accuracy')
    plt.plot(history.history['val_output_2_accuracy'], label='val_output_2_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(101)
    plt.ion()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()


checkpoint_filepath = './tmp'
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True,
                                                                monitor='val_output_2_accuracy', mode='max',
                                                                save_freq='epoch', save_best_only=True)


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        # clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.show()


plot_losses = PlotLosses()
model_1 = HCAE_comp(numb_labels=Numb_labels)


def ssim_loss(y_true, y_pred):
    eval_ = K.square(y_pred - y_true)
    #print(eval_.shape)
    eval_ = K.mean(eval_, axis=-1)
    #print(eval_.shape)
    return eval_

##endregion
del name2
del name3
del labels2_onehot
del onehot_encoded
import gc

gc.collect()
model_1.load_weights("model_withGOOD_training")
model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001 / (1 + 1) ** 3.5), loss=['mean_squared_error', 'mean_squared_error'], loss_weights=[1, 1], metrics=['accuracy'])

def training_loop(imgs, lbls,load="0", dual_mode=0, epoch_iter=3, l_w_AE=1, l_w_class=1, ep_1=1, ep_2=1):

    if load != "0":
        model_1.load_weights(load)

    if dual_mode == 1:
        model_1.layers[0].trainable = False
        model_1.layers[1].trainable = True
        model_1.layers[2].trainable = False
        l_w_AE = 0
    elif dual_mode == 0:
        model_1.layers[0].trainable = True
        model_1.layers[1].trainable = False
        model_1.layers[2].trainable = True
        l_w_class = 0
    elif dual_mode == 2:
        model_1.layers[0].trainable = True
        model_1.layers[1].trainable = True
        model_1.layers[2].trainable = True

    history = []
    for iterr in range(epoch_iter):
        model_1.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001 / ((iterr + 1) ** 3.5)),
                        loss=['mean_squared_error', 'mean_squared_error'],
                        loss_weights=[l_w_AE, l_w_class], metrics=['accuracy'])

        history.append(model_1.fit(x=imgs,
                              y=[imgs, lbls],
                              batch_size=5 * iterr,
                              epochs=ep_1,
                              verbose=1,
                              validation_split=0.2,
                              callbacks=[ model_checkpoint_callback],
                              shuffle=True))

        history.append(model_1.fit(x=imgs,
                              y=[imgs, lbls],
                              batch_size=20 + (5 ** iterr + 1),
                              epochs=ep_2,
                              verbose=1,
                              validation_split=0.2,
                              callbacks=[ model_checkpoint_callback],
                              shuffle=True))
    return history

loading_flag = "Full_mat_model_no_drop_04_10_2021_12_49_58_weights.tf"
history_comp = []
for iter_2 in range(1,5):
    print("round: ", iter_2)
    history_comp=(training_loop(img3, labels3, load=loading_flag, dual_mode=0, epoch_iter=iter_2, l_w_AE=1, l_w_class=0, ep_1=1, ep_2=1))
    #history_comp = training_loop(img2, labels2, load="0", dual_mode=1, epoch_iter=iter_2, l_w_AE=0, l_w_class=1, ep_1=2, ep_2=2)
    history_comp=(training_loop(img2, labels2, load="0", dual_mode=2, epoch_iter=iter_2, l_w_AE=1, l_w_class=1, ep_1=2, ep_2=2))
    loading_flag = "0"
    # plt.plot(history_comp3[0].history['loss'])
    # plt.plot(history_comp3[0].history['val_output_2_loss'])
    # plt.title('model loss')
    # plt.ylabel('val_output_2_loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()
    # # summarize history for loss
    # plt.plot(history_comp3[0].history['loss'])
    # plt.plot(history_comp3[0].history['val_output_2_accuracy'])
    # plt.title('model loss')
    # plt.ylabel('val_output_2_accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'test'], loc='upper left')
    # plt.show()

model_1.evaluate(x=img2, y=[img2, labels2])
now = datetime.now()  # current date and time
date_time = now.strftime("Full_mat_model_no_drop_%m_%d_%Y_%H_%M_%S_weights.tf")
model_1.save_weights(filepath=date_time)

for mode_conf in range(2):
    ##region result visualisation

    fig, axs = plt.subplots(4, 7, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})
    inte = np.random.random_integers(1,len(img2),50)
    img_t = img2[inte]
    real_labels = labels2[inte]
    pred = np.array(model_1(img_t)[0])
    pred_img = np.array(model_1(img_t)[1])
    pred_img_floored = label_floorer(tf.nn.softmax(pred_img,axis=-1).numpy(),upper_ther=upper_ther,lower_ther=lower_ther)

    for j in range(0,7):
        for jj in range(0, 1):
            #axs[0+3*jj][j].matshow(integerization(img_t[2*j]), cmap=plt.cm.gray)
            #axs[0+3*jj][j].set_title("Original Image")
            #axs[1+3*jj][j].matshow(integerization(pred[j*2]), cmap=plt.cm.gray)
            #axs[1+3*jj][j].set_title("AE Image")
            # axs[2+3*jj][j].imshow(my_resize(pred_img[j*2, :, :, 0:3], pred.shape[1], mode_img=2))
            # axs[2+3*jj][j].set_title("Label Image")

            axs[0+4*jj][j].matshow(integerization(img_t[2*j]), cmap=plt.cm.gray)
            #axs[0+4*jj][j].set_title("Original Image")
            axs[1+4*jj][j].imshow(my_resize(pred_img[j*2, :, :, 0:3], pred.shape[1], mode_img=2))
            #axs[1+4*jj][j].set_title("Label Image")
            axs[2 + 4 * jj][j].imshow(my_resize(pred_img_floored[j * 2, :, :, 0:3], pred.shape[1], mode_img=2))
            #axs[2 + 4 * jj][j].set_title("Label Image floored")
            axs[3+4*jj][j].imshow(my_resize(real_labels[j*2, :, :, 0:3], pred.shape[1], mode_img=2))
            #axs[3+4*jj][j].set_title("real Label Image floored")
    pred = np.array(model_1.predict(img2)[1])
    pred = label_floorer(tf.nn.softmax(pred).numpy(),upper_ther=upper_ther,lower_ther=lower_ther)
    pred = label_solidify(pred).reshape([-1, 1])
    labels2_solid = label_solidify(labels2).reshape([-1, 1])

    #confusion = confusion_matrix(labels2_solid, pred)
    #a = f1_score(labels2_solid, pred,zero_division=1,average=None)
    target_names = ['Empty', 'Over printing', 'Normal','Under Printing','Over and Under printing','Under and normal printing','Over and normal printing']
    #a=preprocessing.normalize(confusion,axis=1)
    #plt.matshow(a)
    #plt.title("partial off")
    confusion = my_confusion(pred,labels2_solid,n_labels=None,partial=mode_conf)
    df_cm = pd.DataFrame(confusion, index=target_names, columns=target_names)
    # colormap: see this and choose your more dear
    pp_matrix(df_cm,fmt=".1f",fz=16,lw=2.5,figsize=[8, 8],show_null_values=1)
    ##endregion
    print('f1 score for method{}   '.format('relaxed on ' if mode_conf==1 else 'relaxed off ' ), f1(confusion))
print('f1 score for method{}   '.format('normal f1 ' ), f1_score(labels2_solid, pred,zero_division=1,average=None))
## region Post Process



#
# # Confusion Matrix and Classification Report
# y_pred = np.array(inverse_coding(model_1.predict(img2)[1], Numb_grids_=Numb_grids, Numb_labels_=Numb_labels))
# y_pred = y_pred.reshape(1, -1)[0]
# y_true = labels2
# y_true = y_true.reshape(1, -1)[0]
# print('Confusion Matrix')
# cm1 = confusion_matrix(y_true=y_true, y_pred=y_pred)
# print(cm1)
# print('Classification Report')
# target_names = ['under', 'over', 'ok','Un_Ok','Un_Ov','Ok_Ov','empty', 'ALL']
# print(classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names))
#
# TruePositive = np.diag(cm1)
# print('TruePositive')
# print(TruePositive)
#
# FalsePositive = []
# for i in range(3):
#     FalsePositive.append(sum(cm1[:, i]) - cm1[i, i])
# print('FalsePositive')
# print(FalsePositive)
#
# FalseNegative = []
# for i in range(3):
#     FalseNegative.append(sum(cm1[i, :]) - cm1[i, i])
# print('FalseNegative')
# print(FalseNegative)



#image_path = r'D:\Javid\Conference\Test_data_labeled\For_test\Helix.png'
#image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
#image = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
#image = np.array(image)
#image = image.astype('float32')
#image /= 255
#a = []
#a.append(image)
#a.append(image)
#a.append(image)
#a.append(image)
#img_data = np.array(a).reshape(-1, IMG_HEIGHT, IMG_WIDTH, 3)

#k = 0
#fig, axs = plt.subplots(10, 10, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0})



upper_ther = np.linspace(0.005,0.6,60)
lower_ther = np.linspace(0.005,0.6,60)
results = []
pred_main = np.array(model_1.predict(img2)[1])
pred_main = tf.nn.softmax(pred_main).numpy()
labels2_solid = label_solidify(labels2).reshape([-1, 1])
for upper_ther_iter in upper_ther:
    for lower_ther_iter in lower_ther:
        #if lower_ther_iter >= upper_ther_iter:
            #continue
        pred = label_floorer(pred_main,upper_ther_iter,lower_ther_iter)
        pred = label_solidify(pred).reshape([-1,1])
        results.append([f1_score(labels2_solid, pred, zero_division=1, average=None),f1_score(labels2_solid, pred, zero_division=1, average='weighted'),upper_ther_iter,lower_ther_iter])
flat_list =  np.stack([item[0] for item in results])
flat_list0 =  np.stack([item[1] for item in results])
flat_list1 =  np.stack([item[2] for item in results])
flat_list2 =  np.stack([item[3] for item in results])
for i__ in range(0,7):
    plt.plot(flat_list[:,i__],markersize=.2)
plt.plot(flat_list0[:])
plt.plot(flat_list1[: ],'c*',markersize=2)
plt.plot(flat_list2[:],'r*',markersize=2)
upper_ther = flat_list1[np.where(np.max(flat_list0) == flat_list0)[0][0]]
lower_ther = flat_list2[np.where(np.max(flat_list0) == flat_list0)[0][0]]



