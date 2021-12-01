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
from tensorflow.keras.layers import (Activation, BatchNormalization, Conv2D,
                                     Conv2DTranspose, Dense, Dropout, Input,
                                     MaxPooling2D, Reshape, UpSampling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from helpers import (calc_jacc, read_images_masks, get_metric_plot,
                     get_patches, get_scalers, jaccard, jaccard_int,
                     mask_for_polygons, mask_to_polygons, read_image,
                     stretch_n)


def make_val():
    """Create a validation set"""
    if not x_val_path.exists() or not y_val_path.exists():
        print('Creating a validation dataset...')
        images = np.load(x_train_path)
        masks = np.load(y_train_path)
        x, y = get_patches(images, masks, num_patches=3000)

        np.save(x_val_path, x)
        np.save(y_val_path, y)
    else:
        print('Validation dataset already exists, skipping.')


def train_model(images: np.ndarray, masks: np.ndarray, epochs=2):

    x_train, x_test, y_train, y_test = train_test_split(
        images, masks, test_size=0.2, random_state=42)

    # random patches in correct model input size
    x_train, y_train = get_patches(x_train, y_train, num_patches=5000)
    x_test, y_test   = get_patches(x_test, y_test, num_patches=1000)

    model = get_segnet(image_size)
    callbacks = [
        ModelCheckpoint('weights/segnet.hdf5',
                        monitor='loss', save_best_only=True),
        EarlyStopping(monitor='val_jaccard', patience=8)
    ]

    history = model.fit(x=x_train, y=y_train, batch_size=64, epochs=epochs, shuffle=True,
                        callbacks=callbacks, validation_data=(x_test, y_test))

    del x_train
    del y_train

    score, _ = calc_jacc(model, x_test, y_test)
    print(f'Validation Jaccard Score: {score}')
    # model.save_weights(f'weights/unet_10_jk{score}', save_format='h5')
    model.save(f'models/segnet_jk_score_{round(score,3)}.h5')
    np.save(
        f'models/segnet_jk_score_{round(score,3)}_history.npy', history.history)

    plot_outdir = Path('models/plots')
    plot_outdir.mkdir(exist_ok=True)
    plot = get_metric_plot(history, 'accuracy')
    plot.savefig(plot_outdir.joinpath(f'segnet_jk_score_{round(score,3)}_accuracy.png'))
    plot.show()
    plot = get_metric_plot(history, 'loss')
    plot.savefig(plot_outdir.joinpath(f'segnet_jk_score_{round(score,3)}_loss.png'))
    plot.show()
    plot = get_metric_plot(history, 'jaccard')
    plot.savefig(plot_outdir.joinpath(f'segnet_jk_score_{round(score,3)}_jaccard.png'))
    plot.show()
    plot = get_metric_plot(history, 'jaccard_int')
    plot.savefig(plot_outdir.joinpath(f'segnet_jk_score_{round(score,3)}_jaccard_int.png'))
    plot.show()
    return model

# https://www.kaggle.com/hashbanger/skin-lesion-segmentation-using-segnet
# https://www.kaggle.com/ksishawon/segnet-dstl


def get_segnet(image_size):
    input_shape = [image_size, image_size, 8]
    inputs = Input(input_shape)
    print(inputs.shape)

    # Encoding layers
    c1 = Conv2D(32, (3, 3), padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    c1 = Conv2D(32, (3, 3), padding='same')(c1)
    c1 = BatchNormalization()(c1)
    c1 = Activation('relu')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, (3, 3), padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    c2 = Conv2D(64, (3, 3), padding='same')(c2)
    c2 = BatchNormalization()(c2)
    c2 = Activation('relu')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, (3, 3), padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(128, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    c3 = Conv2D(128, (3, 3), padding='same')(c3)
    c3 = BatchNormalization()(c3)
    c3 = Activation('relu')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(256, (3, 3), padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(256, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    c4 = Conv2D(256, (3, 3), padding='same')(c4)
    c4 = BatchNormalization()(c4)
    c4 = Activation('relu')(c4)
    p4 = MaxPooling2D()(c4)

    c5 = Conv2D(256, (3, 3), padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(256, (3, 3), padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    c5 = Conv2D(256, (3, 3), padding='same')(c5)
    c5 = BatchNormalization()(c5)
    c5 = Activation('relu')(c5)
    p5 = MaxPooling2D()(c5)

    d1 = Dense(512, activation='relu')(p5)
    d1 = Dense(512, activation='relu')(d1)

    # Decoding Layers
    u1 = UpSampling2D()(d1)
    c6 = Conv2DTranspose(256, (3, 3), padding='same')(u1)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2DTranspose(256, (3, 3), padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)
    c6 = Conv2DTranspose(256, (3, 3), padding='same')(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('relu')(c6)

    u2 = UpSampling2D()(c6)
    c7 = Conv2DTranspose(256, (3, 3), padding='same')(u2)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2DTranspose(256, (3, 3), padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)
    c7 = Conv2DTranspose(128, (3, 3), padding='same')(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('relu')(c7)

    u3 = UpSampling2D()(c7)
    c8 = Conv2DTranspose(128, (3, 3), padding='same')(u3)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2DTranspose(128, (3, 3), padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)
    c8 = Conv2DTranspose(64, (3, 3), padding='same')(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('relu')(c8)

    u4 = UpSampling2D()(c8)
    c9 = Conv2DTranspose(64, (3, 3), padding='same')(u4)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)
    c9 = Conv2DTranspose(32, (3, 3), padding='same')(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('relu')(c9)

    u5 = UpSampling2D()(c9)
    c10 = Conv2DTranspose(32, (3, 3), padding='same')(u5)
    c10 = BatchNormalization()(c10)
    c10= Activation('relu')(c10)
    c10 = Conv2DTranspose(N_Cls, (3, 3), padding='same',activation='sigmoid')(c10)
    c10 = BatchNormalization()(c10)
    c10 = Activation('relu')(c10)
    # outputs = Reshape([image_size, image_size])(c10)
    outputs = Conv2D(N_Cls, (1, 1), activation='sigmoid') (c10)
    print(inputs.shape)
    print(outputs.shape)
    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer=Adam(), loss='binary_crossentropy',
                  metrics=[jaccard, jaccard_int, 'accuracy'])
    plot_outdir = Path('models')
    plot_outdir.mkdir(exist_ok=True)
    plot_model(model, to_file=plot_outdir.joinpath('segnet_plot.png'))
    model.summary()
    return model


def predict_id(id, model, trs):
    image = read_image(id)
    x = stretch_n(image)

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:image.shape[0], :image.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * image_size:(i + 1) * image_size,
                        j * image_size:(j + 1) * image_size])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1
        tmp = model.predict(x, batch_size=4)
        for j in range(tmp.shape[0]):
            prd[:, i * image_size:(i + 1) * image_size,
                j * image_size:(j + 1) * image_size] = tmp[j]

    # trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    return prd[:, :image.shape[0], :image.shape[1]]


def predict_test(model, trs):
    for i, id in enumerate(sorted(set(SB['ImageId'].tolist()))):
        msk = predict_id(id, model, trs)
        np.save('msk/10_%s' % id, msk)
        if i % 100 == 0:
            print(i, id)


def make_submit():
    df = pd.read_csv(os.path.join(data_dir, 'sample_submission.csv'))
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
        if idx % 100 == 0:
            print(idx)
    print(df.head())
    df.to_csv('subm/1.csv', index=False)


def check_predict(id='6120_2_3', image_size=160):
    model = get_segnet(image_size)
    # model.load_weights('weights/unet_10_jk0.7878')

    msk = predict_id(
        id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])
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
    ax3.imshow(mask_for_polygons(mask_to_polygons(
        msk[0], epsilon=1), img.shape[:2]), cmap=plt.get_cmap('gray'))

    plt.show()


if __name__ == '__main__':
    N_Cls = 10
    data_dir = Path('dataset')
    image_folder = data_dir.joinpath('sixteen_band')
    model_outdir = Path('App/segnet')
    model_version = 1
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

    model = train_model(images, masks, train_epochs)

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
