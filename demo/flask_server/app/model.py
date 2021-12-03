from pathlib import Path
import sys # added!

import numpy as np
import json
import requests
import os
import re

from modelhelpers import mask_to_polygons, mask_for_polygons, read_image, stretch_n, read_image_rgb
import matplotlib
# matplotlib.use('macOSX')

import matplotlib.pyplot as plt

N_Cls = 10
# MODEL_URI = "http://localhost:8501/v1/models/unet:predict"
SERVING_CONTAINER = os.getenv('SERVING_CONTAINER')
SERVING_PORT = os.getenv('SERVING_PORT')
model = 'unet'
MODEL_URI = f'http://{SERVING_CONTAINER}:{SERVING_PORT}/v1/models/{model}:predict'

headers = {"content-type": "application/json"}
CLASSES = {
    1: 'Bldg',
    2: 'Struct',
    3: 'Road',
    4: 'Track',
    5: 'Trees',
    6: 'Crops',
    7: 'Fast H20',
    8: 'Slow H20',
    9: 'Truck',
    10: 'Car',
}

def get_prediction(image_path, image_size=160):

    print(f'imagepath: {image_path}')
    data_dir = Path('dataset')

    # image_path is splitted to recieve the id
    id_full = os.path.basename(image_path)
    id = re.split('_?[A-Z]?\.', id_full)[0]

    sys.path.append("../..")  # added!
    image_three = read_image_rgb(image_id=id, inDir=data_dir.joinpath('three_band'))

    image_sixteen = read_image(image_id=id, inDir=data_dir.joinpath('sixteen_band'))

    x = stretch_n(image_sixteen)

    print(f'images.shape: {image_sixteen.shape}')
    print('--------------------------------------------------------')
    print(f'x.shape: {x.shape}')
    print('--------------------------------------------------------')

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    pred = np.zeros((960, 960, N_Cls)).astype(np.float32)
    cnv[:image_sixteen.shape[0], :image_sixteen.shape[1], :] = x

    for i in range(0, 6):
        line = []
        for j in range(0, 6):
            line.append(cnv[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size])

        x = 2 * np.transpose(line, (0, 1, 2,3)) - 1

        """ Ab hier der Ausschnitt aus predict_id"""

        test_image = x
        print(f'test_image_shape: {test_image.shape}')

        # add dimension to satisfy model input shape
        #x = test_image.reshape(1, image_size, image_size, 8)
        data = json.dumps({"signature_name": "serving_default", "instances": x.tolist()})
        print('Data: {} ... {}'.format(data[:50], data[len(data) - 52:]))
        print('--------------------------------------------------------')
        print(f'x.shape: {x.shape}')
        print('--------------------------------------------------------')

        response = requests.post(url=MODEL_URI, data=data, headers=headers)
        predictions = json.loads(response.text)
        print(f'prediction_shape: {np.array(predictions["predictions"]).shape}')
        print('--------------------------------------------------------')

        tmp = np.array(predictions['predictions'])
        #print(f'prediction for first pixel: {tmp[0, 1, 1, :]}')

        for j in range(tmp.shape[0]):
            pred[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size,:] = tmp[j]

    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]
    for i in range(N_Cls):
        #print(f'pred_i.shape: {pred[:,:,i].shape}')
        pred[:,:,i] = pred[:,:,i] > trs[i]

    msk = pred[:image_sixteen.shape[0], :image_sixteen.shape[1], :]
    print(f'msk.shape: {msk[1,1,:]}')

    plt.figure()
    fig, axArr = plt.subplots(figsize=(10,10), nrows=3, ncols=5)

    ax = axArr[0][0]
    ax.set_title(f'Ground Truth: RGB')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(image_three)

    ax = axArr[0][1]
    ax.set_title(f'Near IR Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow((image_sixteen[:, :, 6]).astype(np.uint8))

    ax = axArr[0][2]
    ax.set_title(f'IR Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow((image_sixteen[:, :, 7]).astype(np.uint8))

    ax = axArr[0][3]
    ax.set_title(f'Yellow - Near IR - IR Image')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow((image_sixteen[:, :, [5,6, 7]]).astype(np.uint8))



    ax = axArr[1][0]
    ax.set_title(f'pred for class: {CLASSES[1]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:,:,0], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[1][1]
    ax.set_title(f'pred for class: {CLASSES[2]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:,:,1], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[1][2]
    ax.set_title(f'pred for class: {CLASSES[3]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:,:,2], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[1][3]
    ax.set_title(f'pred for class: {CLASSES[4]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:,:,3], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[1][4]
    ax.set_title(f'pred for class: {CLASSES[5]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:,:,4], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[2][0]
    ax.set_title(f'pred for class: {CLASSES[6]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:,:,5], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[2][1]
    ax.set_title(f'pred for class: {CLASSES[7]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:, :, 6], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[2][2]
    ax.set_title(f'pred for class: {CLASSES[8]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:, :, 7], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[2][3]
    ax.set_title(f'pred for class: {CLASSES[9]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:, :, 8], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    ax = axArr[2][4]
    ax.set_title(f'pred for class: {CLASSES[10]}')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(mask_for_polygons(mask_to_polygons(msk[:, :, 9], epsilon=1),
                                image_sixteen.shape[:2]), cmap=plt.get_cmap('gray'))

    # plt.savefig(Path('./new_plot.png'))

    return msk, plt
