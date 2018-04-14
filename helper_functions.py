import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
from sklearn.utils import shuffle
from scipy import ndimage as ndi
import pickle
from lenet5 import *

def distribution(n_cl, y):
    """Return data distribution"""
    dist = np.zeros(n_cl, dtype=np.int32)
    for l in y:
        dist[l] += 1
    return dist


def visualize_data_distribution(y_train, y_valid, y_test, n_classes):
    """Plot data distribution"""
    dist_train = distribution(n_classes, y_train)
    dist_valid = distribution(n_classes, y_valid)
    dist_test = distribution(n_classes, y_test)

    plt.figure(2, figsize=(15, 5))

    plt.subplot(131)
    plt.bar(np.arange(n_classes), dist_train)
    plt.text(n_classes / 3.0, np.max(dist_train), 'Train Data')

    plt.subplot(132)
    plt.bar(np.arange(n_classes), dist_valid)
    plt.text(n_classes / 3.0, np.max(dist_valid), 'Valid Data')

    plt.subplot(133)
    plt.bar(np.arange(n_classes), dist_test)
    plt.text(n_classes / 3.0, np.max(dist_test), 'Test Data')

    plt.suptitle('Data Distribution')
    plt.show()


def display_images_by_class(x, y, cls):
    imgs = []
    for i in range(len(y)):
        if y[i] == cls:
            imgs.append(x[i])

    col, row = 20, 10
    for k in range(0, int(np.ceil(len(imgs) / float(col*row)))):
        for i in range(row):
            plt.figure(1, figsize=(col, row))
            for j in range(col):
                img_idx = k*col*row + (i * col) + j
                if img_idx < len(imgs):
                    plt.subplot(row, col, (i * col) + (j + 1))
                    plt.imshow(imgs[img_idx])
                    plt.axis('off')
        plt.show()


def display_img_types(x, y, cls):
    imgs = {}
    acc = set()
    while len(acc) < cls:
        index = random.randint(0, len(y))
        if y[index] not in acc:
            imgs[y[index]] = x[index]
            acc.add(y[index])

    col, row = 8, 6
    plt.figure(5, figsize=(row, col))
    k = 0
    for i in range(row):
        for j in range(col):
            plt.subplot(row, col, (i * col) + (j + 1))
            plt.imshow(imgs[k])
            plt.axis('off')
            plt.title(str(k))
            k += 1
            if k >= cls:
                break
        if k >= cls:
            break

    plt.show()


def display_random_img(x, y):
    """This can be improved to read appropriate number of images when x is small"""
    col, row = 10, 5
    if len(x) < col*row*10:
        print("Too few data points to select images")
    plt.figure(1, figsize=(col, row))
    img_indx = set()
    for i in range(row):
        for j in range(col):
            index = random.randint(0, len(x))
            while index in img_indx:
                index = random.randint(0, len(x))
            img_indx.add(index)
            image = x[index]

            plt.subplot(row, col, (i*col) + (j+1))
            plt.imshow(image)
            plt.text(5, 0, y[index])
            plt.axis('off')
    plt.show()

def normalize_img_data(x):
    """Quick image data norm"""
    orig_shape = x.shape
    pixels = np.reshape(x, [orig_shape[0] * orig_shape[1] * orig_shape[2], orig_shape[3]])
    u = np.mean(pixels, axis=0)
    min = np.min(pixels, axis=0)
    max = np.max(pixels, axis=0)
    pixels_norm = np.ndarray.astype((pixels - u) / (max - min), np.float32)

    return np.reshape(pixels_norm, orig_shape)


def convert_to_grayscale(x):
    ret = np.zeros([len(x), x.shape[1], x.shape[2]])
    for i in range(len(x)):
        np.append(ret, cv.cvtColor(x[i], cv.COLOR_RGB2GRAY))
    return ret


def synthesize_data(x, y, n_classes):
    class_img_map = {}
    for i in range(len(x)):
        cls = y[i]
        if cls not in class_img_map:
            class_img_map[cls] = list()
        class_img_map[cls].append(x[i])

    syn_features = []
    syn_labels = []
    y_dist = distribution(n_classes, y)
    for cls in range(n_classes):
        # if y_dist[cls] < 500:
        #     cls_list = class_img_map[cls]
        #     for im in cls_list:
                # syn_features.append(ndi.rotate(im, 15.0, reshape=False)), syn_labels.append(cls)
                # syn_features.append(ndi.rotate(im, -15.0, reshape=False)), syn_labels.append(cls)

        if y_dist[cls] < 850:
            cls_list = class_img_map[cls]
            for im in cls_list:
                zm = ndi.zoom(im, (0.75, 0.75, 1))
                zm = np.pad(zm, ((4, 4), (4, 4), (0, 0)), 'constant')
                syn_features.append(zm), syn_labels.append(cls)
                if y_dist[cls] < 600:
                    syn_features.append(ndi.rotate(zm, 10.0, reshape=False)), syn_labels.append(cls)
                if y_dist[cls] < 500:
                    syn_features.append(ndi.rotate(zm, -10.0, reshape=False)), syn_labels.append(cls)

    with open('traffic-signs-data/syn_train.pickle', 'wb') as f:
        pickle.dump({"syn_features": np.array(syn_features), "syn_labels": np.array(syn_labels)}, f)
