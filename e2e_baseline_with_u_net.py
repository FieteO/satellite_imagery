# https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.wkt
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers.merge import concatenate

from helpers import (calc_jacc, generate_training_files, get_patches,
                     get_rgb_from_m_band, get_scalers, jaccard_coef,
                     jaccard_coef_int, mask_for_polygons, mask_to_polygons,
                     stretch_n, subset_in_folder)


def make_val():
    """Create a validation set"""
    if not x_val_path.exists() or not y_val_path.exists():
        print('Creating a validation dataset...')
        img = np.load(x_train_path)
        msk = np.load(y_train_path)
        x, y = get_patches(img, msk, amt=3000)

        np.save(x_val_path, x)
        np.save(y_val_path, y)
    else:
        print('Validation dataset already exists, skipping.')

def train_net(epochs=2):
    print('Loading validation and training dataset...')
    img = np.load(x_train_path)
    msk = np.load(y_train_path)
    x_val = np.load(x_val_path),
    y_val = np.load(y_val_path)
    print('Done loading validation and training dataset.')

    x_trn, y_trn = get_patches(img, msk)

    model = get_unet(image_size)
    model_checkpoint = ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True)
    for _ in range(1):
        model.fit(x=x_trn, y=y_trn, batch_size=64, epochs=epochs, shuffle=True,
                    callbacks=[model_checkpoint], validation_data=(x_val, y_val))

        del x_trn
        del y_trn
        # x_trn, y_trn = get_patches(img, msk)
        # x_val = np.load(x_val_path)
        # y_val = np.load(y_val_path)
        score, _ = calc_jacc(model, x_val, y_val)
        print(f'Validation Jaccard Score: {score}')
        model.save_weights(f'weights/unet_10_jk{score}')

    return model

# https://www.kaggle.com/kmader/data-preprocessing-and-unet-segmentation-gpu
def get_unet(image_size):
    input_shape = [image_size,image_size, 8]
    inputs = Input(input_shape)
    s = BatchNormalization()(inputs)
    s = Dropout(0.5)(s)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

    outputs = Conv2D(N_Cls, (1, 1), activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard_coef, jaccard_coef_int, 'accuracy'])
    plot_model(model, to_file='e2e_unet_plot.png')
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
            line.append(cnv[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = tmp[j]

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


def check_predict(id='6120_2_3', image_size=160):
    model = get_unet(image_size)
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
    image_size = 160
    smooth = 1e-12
    train_epochs = 5

    DF = pd.read_csv(inDir + '/train_wkt_v4.csv')
    DF = subset_in_folder(DF, f'{inDir}/{band}')
    GS = pd.read_csv(inDir + '/grid_sizes.csv', names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    SB = pd.read_csv(os.path.join(inDir, 'sample_submission.csv'))

    # path to train file that contains the images
    x_train_path = Path(f'{inDir}/x_trn_{N_Cls}.npy')
    # path to train file that contains the mask of the image
    y_train_path = Path(f'{inDir}/y_trn_{N_Cls}.npy')
    x_val_path   = Path(f'{inDir}/x_val_{N_Cls}.npy')
    y_val_path   = Path(f'{inDir}/y_val_{N_Cls}.npy')

    generate_training_files(x_train_path, y_train_path)
    make_val()
    model = train_net(epochs=train_epochs)
    # score, trs = calc_jacc(model, np.load(x_val_path), np.load(y_val_path))
    # predict_test(model, trs)
    # make_submit()

    # # bonus
    # check_predict()
