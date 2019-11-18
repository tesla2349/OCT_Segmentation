from model import *
from data import *
from losses import *
import numpy as np
import segmentation_models as sm
from keras.models import load_model
from keras.optimizers import SGD
import tensorflow as tf
import keras.backend as K
import cv2 as cv
import os
import glob
import keras

os.environ["CUDA_VISIBLE_DEVICES"]="0"
keras.losses.focal_dice_loss = focal_dice_loss
model = load_model('results/unet+rgb/models/baseline_1000.h5')

samples = sorted(glob.glob('data/all_data/test/images/*'))

for image in samples:
    img = cv.imread(image, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (256,256))
    im = img.reshape(1,256,256,1)
    p=model.predict(im/255)
    result = p[0]*255
    result=result.reshape(256,256)
    result = cv.normalize(result, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)*255
    im_name = image.split('/')[-1]
    ground_truth = 'data/all_data/test/labels/' + im_name
    ground_truth = cv.imread(ground_truth, cv.IMREAD_GRAYSCALE)
    ground_truth = cv.resize(ground_truth, (256,256))
    save_dir = 'results/unet+rgb/model1000/'
    final_img = np.hstack((img, result, ground_truth))
    final_img = final_img.reshape(final_img.shape[0], final_img.shape[1], 1)
    cv.imwrite(save_dir + im_name, final_img)
