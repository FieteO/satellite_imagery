# https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
import os
from pathlib import Path

from collections import defaultdict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.wkt
# from shapely.wkt import loads as wkt_loads
# from shapely.geometry import MultiPolygon, Polygon
# from sklearn.metrics import jaccard_score

from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.backend import concatenate
from tensorflow.python.keras.engine.training import concat

from helpers import (generate_mask_for_image_and_class, get_rgb_from_m_band,
                     jaccard_coef, jaccard_coef_int, mask_for_polygons,
                     mask_to_polygons, stretch_n, calc_jacc, get_scalers, subset_in_folder, get_patches)


def generate_training_files():
    """Saves the images and masks of the 25 train files as numpy arrays.

    The images initially have slightly different sizes and are therefore reshaped to `(835, 835, 8)`.
    This is marginally smaller then the smallest real image dimension of `837` pixels
    """

    if not x_train_path.exists() or not y_train_path.exists():
        image_size = 835
        # code becomes unreadable with image_size
        s = image_size

        x = np.zeros((5 * image_size, 5 * image_size, 8))
        y = np.zeros((5 * image_size, 5 * image_size, N_Cls))

        ids = sorted(DF.ImageId.unique())
        print(f'Number of images in train: {len(ids)}')
        print(f'Shape of x: {x.shape}')
        print(f'Shape of y: {y.shape}')
        print(f'Resulting uniform image size: (835, 835, 8)')
        print(ids)
        for i in range(5):
            for j in range(5):
                id = ids[5 * i + j]
                img = get_rgb_from_m_band(id)
                img = stretch_n(img)
                print(f'img id: {id}, shape: {img.shape}, max val: {np.amax(img)}, min val: {np.amin(img)}')
                i_pos = s * i
                j_pos = s * j
                # write images of size (835,835, 8) to the array
                x[i_pos:i_pos + s, j_pos:j_pos + s, :] = img[:s, :s, :]
                
                # Save image masks in y
                for z in range(N_Cls):
                    img_mask = generate_mask_for_image_and_class(
                        raster_size=(img.shape[0], img.shape[1]),
                        imageId=id,
                        class_type=z + 1,
                        grid_sizes_panda=GS,
                        wkt_list_pandas=DF
                    )
                    y[i_pos:i_pos + s, j_pos:j_pos + s, z] = img_mask[:s, :s]

        np.save(x_train_path, x)
        np.save(y_train_path, y)
    else:
        print('Traing dataset already exists, skipping.')

def make_val():
    """Create a validation set"""
    if not x_val_path.exists() or not y_val_path.exists():
        img = np.load(x_train_path)
        msk = np.load(y_train_path)
        x, y = get_patches(img, msk, amt=3000)

        np.save(x_val_path, x)
        np.save(y_val_path, y)
    else:
        print('Validation dataset already exists, skipping.')

def train_net():
    x_val = np.load(x_val_path),
    y_val = np.load(y_val_path)
    img = np.load(x_train_path)
    msk = np.load(y_train_path)

    x_trn, y_trn = get_patches(img, msk)

    model = get_unet()
    model.load_weights('weights/unet_10_jk0.7878')
    model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for i in range(1):
        model.fit(x_trn, y_trn, batch_size=64, nb_epoch=1, verbose=1, shuffle=True,
                  callbacks=[model_checkpoint], validation_data=(x_val, y_val))
        del x_trn
        del y_trn
        x_trn, y_trn = get_patches(img, msk)
        score, trs = calc_jacc(model)
        print('val jk', score)
        model.save_weights('weights/unet_10_jk%.4f' % score)

    return model


def get_unet_seq():
    model = Sequential([
        layers.Input((8,ISZ,ISZ)),
        layers.Conv2D(32,3,3, activation='relu', padding='same'),
        layers.Conv2D(32,3,3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2), padding='same'),

        layers.Conv2D(64,3,3, activation='relu', padding='same'),
        layers.Conv2D(64,3,3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),
        
        layers.Conv2D(128,3,3, activation='relu', padding='same'),
        layers.Conv2D(128,3,3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(256,3,3, activation='relu', padding='same'),
        layers.Conv2D(256,3,3, activation='relu', padding='same'),
        layers.MaxPooling2D(pool_size=(2,2)),

        layers.Conv2D(512,3,3, activation='relu', padding='same'),
        layers.Conv2D(512,3,3, activation='relu', padding='same'),
    ])


def get_unet():
    inputs = Input((8, ISZ, ISZ))
    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv1)

    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv2)

    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv3)

    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), padding='same')(conv4)

    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, 3, 3, activation='relu', padding='same')(conv5)
    # Dimension 0 in both shapes must be equal, but are 2 and 1. Shapes are [2,512] and [1,256]. for '{{node tf.keras.backend.concatenate/concat}} = ConcatV2[N=2, T=DT_FLOAT, Tidx=DT_INT32](Placeholder, Placeholder_1, tf.keras.backend.concatenate/concat/axis)'
    # with input shapes: [?,2,2,512], [?,1,1,256], [] and with computed input tensors: input[2] = <1>.
    up6 = concat([UpSampling2D(size=(2, 2))(conv5), conv4], axis=0)
    # up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=1)
    # up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, 3, 3, activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=1)
    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, 3, 3, activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=1)
    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, 3, 3, activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=1)
    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, 3, 3, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(N_Cls, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    return model

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

    # trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :img.shape[0], :img.shape[1]]


def predict_test(model, trs):
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0: print(i, id)


def make_submit():
    df = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    print(df.head())
    for idx, row in df.iterrows():
        id = row[0]
        kls = row[1] - 1

        msk = np.load('msk/10_%s.npy' % id)[kls]
        pred_polygons = mask_to_polygons(msk)
        x_max = GS.loc[GS['ImageId'] == id, 'Xmax'].as_matrix()[0]
        y_min = GS.loc[GS['ImageId'] == id, 'Ymin'].as_matrix()[0]

        x_scaler, y_scaler = get_scalers(msk.shape, x_max, y_min)

        scaled_pred_polygons = shapely.affinity.scale(pred_polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler,
                                                      origin=(0, 0, 0))

        df.iloc[idx, 2] = shapely.wkt.dumps(scaled_pred_polygons)
        if idx % 100 == 0: print(idx)
    print(df.head())
    df.to_csv('subm/1.csv', index=False)


def check_predict(id='6120_2_3'):
    model = get_unet()
    model.load_weights('weights/unet_10_jk0.7878')

    msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
    img = get_rgb_from_m_band(id)

    plt.figure()
    ax1 = plt.subplot(131)
    ax1.set_title('image ID:6120_2_3')
    ax1.imshow(img[:, :, 5], cmap=plt.get_cmap('gist_ncar'))
    ax2 = plt.subplot(132)
    ax2.set_title('predict bldg pixels')
    ax2.imshow(msk[0], cmap=plt.get_cmap('gray'))
    ax3 = plt.subplot(133)
    ax3.set_title('predict bldg polygones')
    ax3.imshow(mask_for_polygons(mask_to_polygons(msk[0], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))

    plt.show()


if __name__ == '__main__':
    N_Cls = 10
    inDir = 'dataset'
    band = 'sixteen_band'
    DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
    DF = subset_in_folder(DF, f'{inDir}/{band}')
    GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))
    ISZ = 160
    smooth = 1e-12

    # path to train file that contains the images
    x_train_path = Path(f'{inDir}/x_trn_{N_Cls}.npy')
    # path to train file that contains the mask of the image
    # that is: ...
    y_train_path = Path(f'{inDir}/y_trn_{N_Cls}.npy')
    x_val_path   = Path(f'{inDir}/x_val_{N_Cls}.npy')
    y_val_path   = Path(f'{inDir}/y_val_{N_Cls}.npy')

    generate_training_files()
    make_val()
    model = train_net()
    score, trs = calc_jacc(model)
    predict_test(model, trs)
    make_submit()

    # # bonus
    # check_predict()
