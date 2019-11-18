import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from model import *
from data import *
import tensorflow as tf
import keras.backend as K
from losses import *
import numpy as np
import cv2 as cv
import segmentation_models as sm
from keras.models import load_model
from keras.optimizers import SGD, Adam

batch_size=1
aug_dict=dict(width_shift_range=0.1,
height_shift_range=0.1,
shear_range=0.01,
zoom_range=[0.9, 1.25],
horizontal_flip=True)

model = unet(n_ch=3,patch_height=256,patch_width=256)
sgd = SGD(lr=0.001)

model.compile(sgd, loss=focal_dice_loss, metrics=[sm.metrics.iou_score])
myGene = trainGenerator(batch_size,'data/all_data/train','images_rgb','labels', aug_dict=aug_dict)
sample = cv.imread('data/all_data/test/sample_image_rgb/Lasered_501.png')
sample = cv.resize(sample, (256,256), interpolation = cv.INTER_AREA)
inp = sample.reshape(1,256,256,3)

gt = cv.imread('data/all_data/test/sample_label/Lasered_501.png', cv.IMREAD_GRAYSCALE)
gt = cv.resize(gt,(256,256))
gt=gt.reshape(256,256,1)

for k in range(1001):
    p = model.predict(inp/255)
    result = p[0]*255
    result=result.reshape(256,256)
    result = cv.normalize(result, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)*255
    result = result.reshape(256,256,1)
    cv.imwrite('results/phase2/unet+rgb/vis/sample_%d.png'%(k), result)
    if k>0 and k%10==0:
        model.save('results/phase2/unet+rgb/models/baseline_%d.h5'%(k))
    model.fit_generator(myGene,steps_per_epoch=32,epochs=100)
