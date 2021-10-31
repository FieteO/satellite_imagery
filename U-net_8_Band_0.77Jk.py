import gc
import os
import random
import warnings
from collections import defaultdict
#from tensorflow.keras import backend as keras
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.wkt
import tensorflow as tf
import tifffile as tiff
from shapely.geometry import MultiPolygon, Polygon
# from shapely.wkt import loads as wkt_loads
from sklearn.metrics import jaccard_score
from tensorflow.keras import Input
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

#from sklearn.metrics import jaccard_score


warnings.filterwarnings("ignore")

from helpers import (calc_jacc, generate_mask_for_image_and_class, generate_training_files, get_patches,
                     get_rgb_from_m_band, get_scalers, jaccard_coef,
                     jaccard_coef_int, mask_for_polygons, mask_to_polygons,
                     stretch_n)

N_Cls = 10
inDir = 'dataset'
DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
ISZ = 160
smooth = 1e-12

# path to train file that contains the images
x_train_path = Path(f'{inDir}/x_trn_{N_Cls}.npy')
# path to train file that contains the mask of the image
y_train_path = Path(f'{inDir}/y_trn_{N_Cls}.npy')


datadir = Path(f'./{inDir}/unet_8_band/data/')
if not datadir.exists():
    os.makedirs(f'./{inDir}/unet_8_band/data/')
    os.makedirs(f'./{inDir}/unet_8_band/msk/')
    os.makedirs(f'./{inDir}/unet_8_band/weights/')
    os.makedirs(f'./{inDir}/unet_8_band/subm/')


# def stick_all_train():
#     print ("let's stick all imgs together")
#     s = 835

#     x = np.zeros((5 * s, 5 * s, 8))
#     y = np.zeros((5 * s, 5 * s, N_Cls))

#     ids = sorted(DF.ImageId.unique())
#     print (len(ids))
#     for i in range(5):
#         for j in range(5):
#             id = ids[5 * i + j]

#             img = get_rgb_from_m_band(id)
#             img = stretch_n(img)
#             print (img.shape, id, np.amax(img), np.amin(img))
#             x[s * i:s * i + s, s * j:s * j + s, :] = img[:s, :s, :]
#             for z in range(N_Cls):
#                 y[s * i:s * i + s, s * j:s * j + s, z] = generate_mask_for_image_and_class(
#                     (img.shape[0], img.shape[1]), id, z + 1)[:s, :s]

#     print (np.amax(y), np.amin(y))

#     np.save(f'{inDir}/unet_8_band/data/x_trn_%d' % N_Cls, x)
#     np.save(f'{inDir}/unet_8_band/data/y_trn_%d' % N_Cls, y)


def make_val():
    print ("let's pick some samples for validation")
    img = np.load(f'{inDir}/unet_8_band/data/x_trn_%d.npy' % N_Cls)
    msk = np.load(f'{inDir}/unet_8_band/data/y_trn_%d.npy' % N_Cls)
    x, y = get_patches(img, msk, amt=2000)

    np.save(f'{inDir}/unet_8_band/data/x_tmp_%d' % N_Cls, x)
    np.save(f'{inDir}/unet_8_band/data/y_tmp_%d' % N_Cls, y)
    
    img = None
    msk = None
    x = None
    y = None
    del(img,msk,x,y)
    gc.collect()



def jaccard_loss(y_true, y_pred):
    return -jaccard_coef(y_true, y_pred)

def get_unet():
    inputs = Input((8, ISZ, ISZ))
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(conv3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2),data_format='channels_first')(drop4)

    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(drop5))
    merge6 = concatenate([drop4,up6], axis = 1)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv6))
    merge7 = concatenate([conv3,up7], axis = 1)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv7))
    merge8 = concatenate([conv2,up8], axis = 1)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(UpSampling2D(size = (2,2),data_format='channels_first')(conv8))
    merge9 = concatenate([conv1,up9], axis = 1)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal',data_format='channels_first')(conv9)
    conv10 = Conv2D(N_Cls, (1, 1),strides=1, activation = 'sigmoid',data_format='channels_first')(conv9)

    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    #model.compile(optimizer=Adam(lr=1e-4), loss = jaccard_loss, metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])

    return model

def train_net():
    print ("start train net")
    x_val, y_val = np.load(f'{inDir}/unet_8_band/data/x_tmp_%d.npy' % N_Cls), np.load(f'{inDir}/unet_8_band/data/y_tmp_%d.npy' % N_Cls)
    img = np.load(f'{inDir}/unet_8_band/data/x_trn_%d.npy' % N_Cls)
    msk = np.load(f'{inDir}/unet_8_band/data/y_trn_%d.npy' % N_Cls)

    x_trn, y_trn = get_patches(img, msk)

    model = get_unet()
    #model.load_weights('../input/trained-weight/unet_10_jk0.7565')
    model_checkpoint = ModelCheckpoint('unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for i in range(1):
        model.fit(x_trn, y_trn, batch_size=64, epochs=10, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(x_val, y_val))
        del x_trn
        del y_trn
        #x_trn, y_trn = get_patches(img, msk)
        score, trs = calc_jacc(model)
        print ('val jk'+ str(score))
        model.save_weights('unet_10_jk%.4f' % score)
    
    x_val = None
    y_val = None
    x_trn = None
    y_trn = None
    del(x_val,y_val,x_trn,y_trn)
    gc.collect()
    
    return model, score, trs

def predict_id(id, model, trs):
    img = get_rgb_from_m_band(id)
    x = stretch_n(img)

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * ISZ:(i + 1) * ISZ, j * ISZ:(j + 1) * ISZ] = tmp[j]

    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]



def predict_test(model, trs):
    print ("predict test")
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save(f'{inDir}/unet_8_band/msk/10_%s' % id, msk)
        #if i % 100 == 0: print (i, id)



def make_submit():
    print ("make submission file")
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print (df.head())
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load(f'{inDir}/unet_8_band/msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 1000 == 0: print (idx)
    print (df.head())
    df.to_csv(f'{inDir}/unet_8_band/subm/1.csv', index=False)



def check_predict(id='6120_2_3'):
    model = get_unet()
    model.load_weights(f'{inDir}/unet_8_band/working/unet_10_jk%.4f' % score)

    msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    m = get_rgb_from_m_band(id)
    print(m.shape)
    class_list = ["Buildings", "Misc.Manmade structures" ,"Road",\
                  "Track","Trees","Crops","Waterway","Standing water",\
                  "Vehicle Large","Vehicle Small"]
    
    img = np.zeros((m.shape[0],m.shape[1],3))
    img[:,:,0] = m[:,:,4] #red
    img[:,:,1] = m[:,:,2] #green
    img[:,:,2] = m[:,:,1] #blue
    for i in range(10):
        plt.figure(figsize=(20,20))
        ax1 = plt.subplot(131)
        ax1.set_title('image ID:6120_2_3')
        ax1.imshow(stretch_n(img))
        ax2 = plt.subplot(132)
        ax2.set_title("predict "+ class_list[i] +" pixels")
        ax2.imshow(msk[i], cmap=plt.get_cmap('gray'))
        ax3 = plt.subplot(133)
        ax3.set_title("predict " + class_list[i] + " polygones")
        ax3.imshow(mask_for_polygons(mask_to_polygons(msk[i], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))
        plt.show()



generate_training_files(x_train_path, y_train_path)
DF = None
x = None
y = None
img = None
ids = None
del(DF,x,y,img,ids)
gc.collect()



make_val()



model, score, trs = train_net()



check_predict('6120_2_2')



predict_test(model, trs)

img = None
cnv = None
del(img,cnv)
gc.collect()

