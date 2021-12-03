# Tensorflow serving

[Tensorflow Serving REST Tutorial](https://www.tensorflow.org/tfx/tutorials/serving/rest_simple)

## Getting started

### Directory structure
``` bash
fiete@ubu /ssd2/development/satellite_imagery/demo$ tree -d
.
├── flask_server
│   └── app
│       ├── static
│       └── template
├── scripts
├── segnet
│   └── 1
│       └── variables
└── unet
    └── 1
        └── variables

11 directories
```

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