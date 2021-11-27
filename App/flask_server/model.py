import tensorflow as tf
import numpy as np
import json
import requests
import os
import re
import matplotlib.pyplot as plt

from helpers_II import get_rgb_from_m_band, mask_to_polygons, mask_for_polygons
from helpers_II import stretch_n

N_Cls = 10
SIZE=128
MODEL_URI='http://localhost:8501/v1/models/unet:predict'
#CLASSES = ['Cat', 'Dog']

def get_prediction(image_path, image_size=160):

    #image_path is splitted to recieve the id
    id_full = os.path.basename(image_path)
    id = re.split('_?[A-Z]?\.',id_full)[0]
    print (f'id: {id} & imagepath: {image_path} & id_full: {id_full}')
    """ Ab hier der Ausschnitt aus predict_id"""

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

        """ Bis hier der Ausschnitt aus predict_id"""

        """ TF-Modelserver Ansprechen"""

        data = json.dumps({
            'instances': x.tolist()
        })
        response = requests.post(MODEL_URI, data=data.encode('utf-8'))
        result = json.loads(response.text)


        for j in range(result.shape[0]):
            prd[:, i * image_size:(i + 1) * image_size, j * image_size:(j + 1) * image_size] = result[j]


    trs = [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1]

    for i in range(N_Cls):
        prd[i] = prd[i] > trs[i]

    msk =  prd[:, :img.shape[0], :img.shape[1]]
    #msk = predict_id(id, model, [0.4, 0.1, 0.4, 0.3, 0.3, 0.5, 0.3, 0.6, 0.1, 0.1])


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

    #plt.show()
    return msk, fig