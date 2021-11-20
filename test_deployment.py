import os
import re

import numpy as np
import json
import requests
import os
import re
import matplotlib.pyplot as plt

from helpers import get_rgb_from_m_band, mask_to_polygons, mask_for_polygons
from helpers import stretch_n

N_Cls = 10
SIZE = 128
MODEL_URI = "http://localhost:8501/v1/models/unet:predict"
image_path = './data.nosync/three_band/6010_0_0.tif'

headers = {"content-type": "application/json"}


def get_prediction(image_path, image_size=160):

    # image_path is splitted to recieve the id
    id_full = os.path.basename(image_path)
    id = re.split('_?[A-Z]?\.', id_full)[0]

    """ Ab hier der Ausschnitt aus predict_id"""

    img = get_rgb_from_m_band(image_id=id, inDir='data.nosync')
    x = stretch_n(img)

    cnv = np.zeros((960, 960, 8)).astype(np.float32)
    prd = np.zeros((N_Cls, 960, 960)).astype(np.float32)
    cnv[:img.shape[0], :img.shape[1], :] = x

    for i in range(0, 6):
        if i == 1:
            break
        line = []
        for j in range(0, 6):
            line.append(cnv[i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size])

        x = 2 * np.transpose(line, (0, 3, 1, 2)) - 1

        """ Bis hier der Ausschnitt aus predict_id"""

        """ TF-Modelserver Ansprechen"""

        data = json.dumps(x.tolist())

        print(f'x.shape: {x.shape}')
        print('--------------------------------------------------------')

        response = requests.post(url=MODEL_URI, json=data, headers=headers)

        #print(response.text)
        result = json.loads(response.text)
        print(type(result))
        #prediction = np.squeeze(result)

        #print(result.keys())
        result = list(result['error'].split('"'))[1]
        #print(type(result))

        json_input = f'{{"numbers" : {result}}}'


        dict = json.loads(json_input)
        numpy_2d_arrays = [np.array(number) for number in dict["numbers"]]

        print(numpy_2d_arrays[0].shape)

        """ Ab hier wieder der Ausschnitt aus der predict_id function """

        tmp = numpy_2d_arrays[0]
        #print (tmp[0])

        print(f'prd shape: {prd.shape}')
        for j in range(tmp.shape[0]):
            print(tmp[j].shape)
            prd[:, i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = tmp[j]

    msk =  prd[:, :img.shape[0], :img.shape[1]]

    fig = plt.figure()
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
    #return msk, fig



print(get_prediction(image_path=image_path))