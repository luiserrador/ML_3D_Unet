import unittest
from utils.layers_func import create_unet3d
from utils.train import *


def testTrainer():
    model = create_unet3d(input_shape=[128, 128, 128, 2],
                          n_convs=2,
                          n_filters=[8, 16, 32, 64],
                          ksize=[3, 3, 3],
                          padding='same',
                          pooling='avg',
                          norm='batch_norm',
                          dropout=[0.25, 0.5, 0.5],
                          upsampling=True,
                          activation='relu',
                          depth=4)

    trainer = Trainer(model, tf.keras.optimizers.Adam, 1e-3, 'tf_ckpt')
