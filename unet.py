# https://www.kaggle.com/drn01z3/end-to-end-baseline-with-u-net-keras
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.affinity
import shapely.wkt
from sklearn.model_selection import train_test_split
from tensorflow.keras import models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dropout, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.metrics import MeanIoU

from helpers import (calc_jacc, get_metric_plot, get_patches, get_scalers,
                     jaccard, jaccard_int, mask_for_polygons, mask_to_polygons,
                     read_image, read_images_masks, stretch_n)


def make_val():
    """Create a validation set"""
    if not x_val_path.exists() or not y_val_path.exists():
        print('Creating a validation dataset...')
        img = np.load(x_train_path)
        msk = np.load(y_train_path)
        x, y = get_patches(img, msk, num_patches=3000)

        np.save(x_val_path, x)
        np.save(y_val_path, y)
    else:
        print('Validation dataset already exists, skipping.')

def train_net(epochs=2):
    x_train, x_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.2, random_state=42)

    # random patches in correct model input size
    x_train, y_train = get_patches(x_train, y_train, num_patches=5000)
    x_test, y_test   = get_patches(x_test, y_test, num_patches=1000)

    model = get_unet(image_size)
    callbacks = [
        ModelCheckpoint('weights/unet_tmp.hdf5', monitor='loss', save_best_only=True),
        EarlyStopping(monitor='val_jaccard', patience=8)
    ]
    for _ in range(1):
        history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=epochs, shuffle=True,
                    callbacks=callbacks, validation_data=(x_test, y_test))

        del x_train
        del y_train
        # x_trn, y_trn = get_patches(img, msk, num_patches=5000)
        # x_val = np.load(x_val_path)
        # y_val = np.load(y_val_path)
        score, _ = calc_jacc(model, x_test, y_test)
        print(f'Validation Jaccard Score: {score}')
        # model.save_weights(f'weights/unet_10_jk{score}', save_format='h5')
        model.save(f'models/unet_jk_score_{round(score,3)}.h5')
        np.save(f'models/unet_jk_score_{round(score,3)}_history.npy',history.history)
        
        plot = get_metric_plot(history, 'accuracy')
        plot.savefig(f'models/plots/unet_jk_score_{round(score,3)}_accuracy.png')
        plot.show()
        plot = get_metric_plot(history, 'loss')
        plot.savefig(f'models/plots/unet_jk_score_{round(score,3)}_loss.png')
        plot.show()
        plot = get_metric_plot(history, 'jaccard')
        plot.savefig(f'models/plots/unet_jk_score_{round(score,3)}_jaccard.png')
        plot.show()
        plot = get_metric_plot(history, 'jaccard_int')
        plot.savefig(f'models/plots/unet_jk_score_{round(score,3)}_jaccard_int.png')
        plot.show()
    return model

# https://www.kaggle.com/kmader/data-preprocessing-and-unet-segmentation-gpu
def get_unet(image_size, N_Cls=10):
    input_shape = [image_size,image_size, 8]
    inputs = Input(input_shape)
    s = BatchNormalization()(inputs)
    s = Dropout(0.5)(s)

    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(s)
    c1 = Conv2D(8, (3, 3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2, 2)) (c1)

    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(p1)
    c2 = Conv2D(16, (3, 3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2, 2)) (c2)

    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(p2)
    c3 = Conv2D(32, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2)) (c3)

    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(p3)
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(p4)
    c5 = Conv2D(128, (3, 3), activation='relu', padding='same')(c5)

    u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(u6)
    c6 = Conv2D(64, (3, 3), activation='relu', padding='same')(c6)

    u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(u7)
    c7 = Conv2D(32, (3, 3), activation='relu', padding='same')(c7)

    u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(u8)
    c8 = Conv2D(16, (3, 3), activation='relu', padding='same')(c8)

    u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(u9)
    c9 = Conv2D(8, (3, 3), activation='relu', padding='same')(c9)

    outputs = Conv2D(N_Cls, (1, 1), activation='sigmoid')(c9)
    print(inputs.shape)
    print(outputs.shape)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[jaccard, jaccard_int, 'accuracy'])
    # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/MeanIoU
    # model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[MeanIoU, 'accuracy'])
    plot_model(model, to_file='models/unet_plot.png')
    model.summary()
    return model

def predict_id(id, model, trs):
    img = read_image(id)
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
    img = read_image(id)

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
    data_dir = Path('dataset')
    image_folder = data_dir.joinpath('sixteen_band')
    model_outdir = Path('App/unet')
    model_version = 2

    image_size = 160
    smooth = 1e-12
    train_epochs = 40

    MASK_DF = pd.read_csv(data_dir.joinpath('train_wkt_v4.csv'))
    GRID_DF = pd.read_csv(data_dir.joinpath('grid_sizes.csv'),
                     names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
    SB = pd.read_csv(data_dir.joinpath('sample_submission.csv'))

    images_path  = data_dir.joinpath('images.npy')
    masks_path   = data_dir.joinpath('masks.npy')
    x_train_path = data_dir.joinpath(f'x_trn_{N_Cls}.npy')
    y_train_path = data_dir.joinpath(f'y_trn_{N_Cls}.npy')
    x_val_path   = data_dir.joinpath(f'x_val_{N_Cls}.npy')
    y_val_path   = data_dir.joinpath(f'y_val_{N_Cls}.npy')

    if not images_path.exists() or not masks_path.exists():
        print('Creating base dataset...')
        images, masks = read_images_masks(MASK_DF, GRID_DF, image_folder)
        np.save(images_path, images)
        np.save(masks_path, masks)
    else:
        print('Base dataset already exists, skipping.')
        images = np.load(images_path)
        masks  = np.load(masks_path)

    model = train_net(epochs=train_epochs)
    
    model_outdir = model_outdir.joinpath(str(model_version))
    models.save_model(
        model,
        model_outdir,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
    )
    print(f'Saved model to ${model_outdir}')

    # score, trs = calc_jacc(model, np.load(x_val_path), np.load(y_val_path))
    # predict_test(model, trs)
    # make_submit()

    # # bonus
    # check_predict()
