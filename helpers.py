from typing import Tuple
import numpy as np
import os
import cv2
import random
import tensorflow as tf
import pandas as pd

from PIL import Image

def can_be_concatenated(layer1, layer2):
    """Check wether the first three layer dimensions (i.e `[None,1,1]`) are the same and can thus be concatenated."""
    return layer1.shape.as_list()[:3] == layer2.shape.as_list()[:3]

def _convert_coordinates_to_raster(coords, img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    Xmax, Ymax = xymax
    H, W = img_size
    W1 = 1.0 * W * W / (W + 1)
    H1 = 1.0 * H * H / (H + 1)
    xf = W1 / Xmax
    yf = H1 / Ymax
    coords[:, 1] *= yf
    coords[:, 0] *= xf
    coords_int = np.round(coords).astype(np.int32)
    return coords_int

def calc_jacc(model, img, msk, N_Cls = 10):
    from sklearn.metrics import jaccard_score

    prd = model.predict(img, batch_size=4)
    print(prd.shape, msk.shape)
    avg, trs = [], []

    for i in range(N_Cls):
        y_true = msk[:, i, :, :]
        t_prd = prd[:, i, :, :]
        y_true = y_true.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])
        t_prd = t_prd.reshape(msk.shape[0] * msk.shape[2], msk.shape[3])

        m, b_tr = 0, 0
        for j in range(10):
            tr = j / 10.0
            y_pred = t_prd > tr

            jk = jaccard_score(y_true, y_pred, average='micro')
            if jk > m:
                m = jk
                b_tr = tr
        print(i, m, b_tr)
        avg.append(m)
        trs.append(b_tr)

    score = sum(avg) / 10.0
    return score, trs

def get_patch(images: np.ndarray, masks: np.ndarray, image_size):
    """
    Returns an image patch from a random position in the given array
    """
    x_length = images.shape[0]
    y_length = images.shape[1]
    # random number + image_size may not be larger then length
    # this is to prevent an array out of bounds in the following steps
    random_x_pos = random.randint(0, x_length - image_size)
    random_y_pos = random.randint(0, y_length - image_size)
    
    x_lower = random_x_pos
    y_lower = random_y_pos
    x_upper = random_x_pos + image_size
    y_upper = random_y_pos + image_size
    image_patch = images[x_lower:x_upper, y_lower:y_upper,:]
    mask_patch  =  masks[x_lower:x_upper, y_lower:y_upper,:]
    return image_patch, mask_patch


def get_x_patch(images: np.ndarray, image_size):
    """
    Returns an image patch from a random position in the given array
    """
    x_length = images.shape[0]
    y_length = images.shape[1]
    # random number + image_size may not be larger then length
    # this is to prevent an array out of bounds in the following steps
    random_x_pos = random.randint(0, x_length - image_size)
    random_y_pos = random.randint(0, y_length - image_size)

    x_lower = random_x_pos
    y_lower = random_y_pos
    x_upper = random_x_pos + image_size
    y_upper = random_y_pos + image_size
    image_patch = images[x_lower:x_upper, y_lower:y_upper, :]

    return image_patch

def get_patches(images, masks, amt=10000, aug=True, image_size=160, N_Cls=10, labels_start=False):
    """
    Splits up the given numpy array into patches of [number_of_images,8,160,160]
    In: (4175,4175,8)
    Out: (3547, 8, 160, 160)

    :param bool labels_start: have labels in the second dimension of the output array
    """
    assert image_size == int(1.0 * image_size), 'image_size should conform to a specific format'
    x, y = [], []

    tr = [0.4, 0.1, 0.1, 0.15, 0.3, 0.95, 0.1, 0.05, 0.001, 0.005]
    for _ in range(amt):
        x_patch, y_patch = get_patch(images, masks, image_size)

        im = x_patch
        ms = y_patch

        for class_index in range(N_Cls):
            sm = np.sum(ms[:, :, class_index])
            magic_number        = 1.0 * sm / image_size ** 2
            other_magic_number  = tr[class_index]
            if magic_number > other_magic_number:
                if aug:
                    if random.uniform(0, 1) > 0.5:
                        # print(f'Before: {im[:10,0,0]}')
                        im = im[::-1]
                        # print(f'After: {im[:10,0,0]}')
                        ms = ms[::-1]
                    if random.uniform(0, 1) > 0.5:
                        im = im[:, ::-1]
                        ms = ms[:, ::-1]
                x.append(im)
                y.append(ms)

    x = np.array(x)
    y = np.array(y)
    if labels_start:
        # reshape the array into another column order from (160,160,8) to (8, 160, 160)
        column_order = (0,3,1,2)
        x = np.transpose(x, column_order),
        y = np.transpose(y, column_order)
    x = 2 * x - 1
    return x, y


def _get_xmax_ymin(grid_sizes_panda, imageId):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xmax, ymin = grid_sizes_panda[grid_sizes_panda.ImageId == imageId].iloc[0, 1:].astype(float)
    return (xmax, ymin)

def _get_polygon_list(wkt_list_pandas, imageId, cType):
    from shapely.wkt import loads as wkt_loads
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    df_image = wkt_list_pandas[wkt_list_pandas.ImageId == imageId]
    multipoly_def = df_image[df_image.ClassType == cType].MultipolygonWKT
    polygonList = None
    if len(multipoly_def) > 0:
        assert len(multipoly_def) == 1
        polygonList = wkt_loads(multipoly_def.values[0])
    return polygonList

def _get_and_convert_contours(polygonList, raster_img_size, xymax):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    perim_list = []
    interior_list = []
    if polygonList is None:
        return None
    for k in range(len(polygonList)):
        poly = polygonList[k]
        perim = np.array(list(poly.exterior.coords))
        perim_c = _convert_coordinates_to_raster(perim, raster_img_size, xymax)
        perim_list.append(perim_c)
        for pi in poly.interiors:
            interior = np.array(list(pi.coords))
            interior_c = _convert_coordinates_to_raster(interior, raster_img_size, xymax)
            interior_list.append(interior_c)
    return perim_list, interior_list

def _plot_mask_from_contours(raster_img_size, contours, class_value=1):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    img_mask = np.zeros(raster_img_size, np.uint8)
    if contours is None:
        return img_mask
    perim_list, interior_list = contours
    cv2.fillPoly(img_mask, perim_list, class_value)
    cv2.fillPoly(img_mask, interior_list, 0)
    return img_mask

def generate_mask_for_image_and_class(raster_size, imageId, class_type, grid_sizes_panda, wkt_list_pandas):
    # __author__ = visoft
    # https://www.kaggle.com/visoft/dstl-satellite-imagery-feature-detection/export-pixel-wise-mask
    xymax = _get_xmax_ymin(grid_sizes_panda, imageId)
    polygon_list = _get_polygon_list(wkt_list_pandas, imageId, class_type)
    contours = _get_and_convert_contours(polygon_list, raster_size, xymax)
    mask = _plot_mask_from_contours(raster_size, contours, 1)
    return mask

def stretch_n(bands, lower_percent=5, higher_percent=95):
    out = np.zeros_like(bands, dtype=np.float32)
    n = bands.shape[2]
    for i in range(n):
        a = 0  # np.min(band)
        b = 1  # np.max(band)
        c = np.percentile(bands[:, :, i], lower_percent)
        d = np.percentile(bands[:, :, i], higher_percent)
        t = a + (bands[:, :, i] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[:, :, i] = t

    return out.astype(np.float32)

# https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric#standalone_usage_2
# Not done yet
# class Jaccard(tf.keras.metrics.Metric):
#     """
#     A custom Keras metric to compute the running average of the confusion matrix
#     """
#     def __init__(self, name='jaccard', **kwargs):
#         super(Jaccard, self).__init__(name=name, **kwargs)
#         self.jaccard_score = self.add_weight(name='jc', initializer='zeros')

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.jaccard_score.assign_add()

#     def result(self):
#         return self.jaccard_score

def jaccard(y_true, y_pred, smooth = 1e-12):
    from tensorflow.keras import backend
    # __author__ = Vladimir Iglovikov
    intersection = backend.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = backend.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return backend.mean(jac)

def jaccard_int(y_true, y_pred, smooth = 1e-12):
    from tensorflow.keras import backend
    # __author__ = Vladimir Iglovikov
    y_pred_pos = backend.round(backend.clip(y_pred, 0, 1))

    intersection = backend.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = backend.sum(y_true + y_pred_pos, axis=[0, -1, -2])
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return backend.mean(jac)

def get_scalers(im_size, x_max, y_min):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    h, w = im_size  # they are flipped so that mask_for_polygons works correctly
    h, w = float(h), float(w)
    w_ = 1.0 * w * (w / (w + 1))
    h_ = 1.0 * h * (h / (h + 1))
    return w_ / x_max, h_ / y_min

def mask_for_polygons(polygons, im_size):
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly
    img_mask = np.zeros(im_size, np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons
                 for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask

def mask_to_polygons(mask, epsilon=5, min_area=1.):
    from shapely.geometry import MultiPolygon, Polygon
    from collections import defaultdict
    # __author__ = Konstantin Lopuhin
    # https://www.kaggle.com/lopuhin/dstl-satellite-imagery-feature-detection/full-pipeline-demo-poly-pixels-ml-poly

    # first, find contours with cv2: it's much faster than shapely

    contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8),
                                                    cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    approx_contours = [cv2.approxPolyDP(cnt, epsilon, True)
                       for cnt in contours]
    if not contours:
        return MultiPolygon()
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy[0]):
        if parent_idx != -1:
            child_contours.add(idx)
            cnt_children[parent_idx].append(approx_contours[idx])
    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    for idx, cnt in enumerate(approx_contours):
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            poly = Polygon(
                shell=cnt[:, 0, :],
                holes=[c[:, 0, :] for c in cnt_children.get(idx, [])
                       if cv2.contourArea(c) >= min_area])
            all_polygons.append(poly)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons

def read_image(image_id, inDir='data.nosync/sixteen_band'):
    """Returns a three channel rgb image from a 16 band image"""
    import tifffile as tiff
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, f'{image_id}_M.tif')
    img = tiff.imread(filename)
    print(f'img_before_roll: {img.shape}')
    img = np.rollaxis(img, 0, 3)
    print(f'img_after_roll: {img.shape}')
    return img

def subset_in_folder(dataframe, folder):
    """Filter out dataframe rows that do not exist as a file"""
    files_in_folder = os.listdir(folder)
    ids = [file[:8] for file in files_in_folder]
    unique_ids = set(ids)
    subset = dataframe[dataframe['ImageId'].isin(unique_ids)]
    return subset

def generate_training_files(masks: pd.DataFrame, grid_sizes: pd.DataFrame, x_train_path, y_train_path) -> Tuple[np.ndarray, np.ndarray]:
    """Saves the images and masks of the 25 train files as numpy arrays.

    The images initially have slightly different sizes and are therefore reshaped to `(835, 835, 8)`.
    This is marginally smaller then the smallest real image dimension of `837` pixels
    """

    if not x_train_path.exists() or not y_train_path.exists():
        print('Generating a training dataset...')
        image_size  = 835
        ids         = sorted(masks['ImageId'].unique())
        n_classes   = masks['ClassType'].nunique()
        # code becomes unreadable with image_size
        s = image_size

        x = np.zeros((5 * image_size, 5 * image_size, 8))
        y = np.zeros((5 * image_size, 5 * image_size, n_classes))

        
        # print(f'Number of images in train: {len(ids)}')
        # print(f'Shape of x: {x.shape}')
        # print(f'Shape of y: {y.shape}')
        # print(f'Resulting uniform image size: (835, 835, 8)')
        for i in range(5):
            for j in range(5):
                id = ids[5 * i + j]
                img = read_image(id)
                img = stretch_n(img)
                # print(f'img id: {id}, shape: {img.shape}, max val: {np.amax(img)}, min val: {np.amin(img)}')
                i_pos = s * i
                j_pos = s * j
                # write images of size (835,835, 8) to the array
                x[i_pos:i_pos + s, j_pos:j_pos + s, :] = img[:s, :s, :]
                
                # Save image masks in y
                for z in range(n_classes):
                    img_mask = generate_mask_for_image_and_class(
                        raster_size=(img.shape[0], img.shape[1]),
                        imageId=id,
                        class_type=z + 1,
                        grid_sizes_panda=grid_sizes,
                        wkt_list_pandas=masks
                    )
                    y[i_pos:i_pos + s, j_pos:j_pos + s, z] = img_mask[:s, :s]
        np.save(x_train_path, x)
        np.save(y_train_path, y)
        return (x, y)
    else:
        print('Training dataset already exists, skipping.')
        x = np.load(x_train_path)
        y = np.load(y_train_path)
        return (x, y)

from tensorflow.keras.callbacks import History
def get_metric_plot(history: History, metric):
    import matplotlib.pyplot as plt
    train_metrics = history.history[metric]
    val_metrics = history.history['val_'+metric]
    epochs = range(1, len(train_metrics) + 1)
    plt.plot(epochs, train_metrics)
    plt.plot(epochs, val_metrics)
    plt.title('Training and validation '+ metric)
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend(["train_"+metric, 'val_'+metric])
    return plt


def generate_deploy_data(id) -> Tuple[np.ndarray, np.ndarray]:
    """Saves the images and masks of the 25 train files as numpy arrays.

    The images initially have slightly different sizes and are therefore reshaped to `(835, 835, 8)`.
    This is marginally smaller then the smallest real image dimension of `837` pixels
    """

    print('Generating a static_dataset...')
    image_size = 835

    # code becomes unreadable with image_size
    s = image_size
    i = 0
    j = 0
    x = np.zeros((5 * image_size, 5 * image_size, 8))


    # print(f'Number of images in train: {len(ids)}')
    # print(f'Shape of x: {x.shape}')
    # print(f'Shape of y: {y.shape}')
    # print(f'Resulting uniform image size: (835, 835, 8)')

    img = read_image(id)
    img = stretch_n(img)
    # print(f'img id: {id}, shape: {img.shape}, max val: {np.amax(img)}, min val: {np.amin(img)}')
    i_pos = s * i
    j_pos = s * j
    # write images of size (835,835, 8) to the array
    x[i_pos:i_pos + s, j_pos:j_pos + s, :] = img[:s, :s, :]

    #np.save(x_dep_path, x)

    return (x)

def read_image_rgb(image_id, inDir='data.nosync/three_band'):
    """Returns a three channel rgb image from a 16 band image
    import tifffile as tiff
    # __author__ = amaia
    # https://www.kaggle.com/aamaia/dstl-satellite-imagery-feature-detection/rgb-using-m-bands-example
    filename = os.path.join(inDir, f'{image_id}.tif')
    img = tiff.imread(filename)
    print(f'img_before_roll: {img.shape}')
    img = np.rollaxis(img, 0, 3)
    print(f'img_after_roll: {img.shape}')
    return img"""
    filename = os.path.join(inDir, f'{image_id}.tif')
    img = Image.open(filename)
    return img

