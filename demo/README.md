# Tensorflow serving

[Tensorflow Serving REST Tutorial](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)

## Getting started

### Environment setup
Create an `.env` file that contains the environment specific container configuration.
``` bash
# Run frontend server in debug mode
DEBUG=True
# Directory that the frontend container will mount as a volume
DATASET_DIR=dataset
```

``` bash
docker-compose up
```

``` bash
(satellite_imagery)fiete@ubu:/ssd2/development/satellite_imagery/App/scripts$ python predict_image.py 
Data: {"signature_name": "serving_default", "instances": ... 29275513, 0.9463414549827576, 0.9793103337287903]]]}
```