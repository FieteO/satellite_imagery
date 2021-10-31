# [Dstl Satellite Imagery Feature Detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/overview)

- [Upsampling and Image Segmentation with Tensorflow and TF-Slim](http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/)

[Keras UNet implementations (Github)](https://github.com/karolzak/keras-unet)

## Semantic segmentation
![Semantic segmentation](readme_images/pixel_wise_classifier.png)

## [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index)
The Jaccard coefficient measures similarity between finite sample sets, and is defined as the size of the intersection divided by the size of the union of the sample sets.
![Ground truth vs. predicted bounding boxes on a stop sign](readme_images/jaccard.jpeg)  
![Areas of overlap and union](readme_images/area.png)  
![Comparison of results](readme_images/intersection.png)  

## Getting started

### Creating a virtual environment
``` bash
pipenv install
```

## About the dataset
For image segmentation we need images to work on and masks that provide the ground truth for segments in the images.
A mask is an area in the image belonging to a specific class. In this dataset they are provided in the `train_wkt_v4.csv` file that has a `MultipolygonWKT` column which contains `MultiPolygon` objects of the [Shapely](https://shapely.readthedocs.io/en/stable/project.html) library.