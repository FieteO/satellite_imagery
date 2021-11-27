# https://www.tensorflow.org/tfx/tutorials/serving/rest_simple#make_a_request_to_your_model_in_tensorflow_serving
import json
from pathlib import Path
import numpy as np
import requests

import sys # added!
sys.path.append("../..") # added!
from helpers import get_patch

model_name = 'unet'
inDir = '../../dataset'
x_train_path = Path(f'{inDir}/x_trn_10.npy')
y_train_path = Path(f'{inDir}/y_trn_10.npy')

images = np.load(x_train_path)
masks = np.load(y_train_path)

test_image, test_mask = get_patch(images, masks, 160)
# add dimension to satisfy model input shape
test_image = test_image.reshape(1,160,160,8)
data = json.dumps({"signature_name": "serving_default", "instances": test_image.tolist()})
print('Data: {} ... {}'.format(data[:50], data[len(data)-52:]))


headers = {"content-type": "application/json"}
json_response = requests.post(f'http://localhost:8501/v1/models/{model_name}:predict', data=data, headers=headers)
predictions = json.loads(json_response.text)
print(predictions)
# print(predictions[:50])
