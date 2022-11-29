import os
import shutil

import tensorflow as tf
from utils.layers_func import create_unet3d
from utils.layers import create_unet3d_class
from utils.train import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Comment to use GPU


def training_create_unet3d():
    """Training example of 3D-Unet using Trainer class"""

    tf.random.set_seed(10)

    model_dir = 'tf_ckpt'  # directory where to save checkpoints

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    model = create_unet3d(input_shape=[32, 32, 32, 1],
                          n_convs=1,
                          n_filters=[8, 16],
                          ksize=[3, 3, 3],
                          padding='same',
                          pooling='avg',
                          norm='batch_norm',
                          dropout=[0],
                          upsampling=True,
                          activation='relu',
                          depth=2)  # create U-Net 3D model

    trainer = Trainer(model, tf.keras.optimizers.Adam, 1e-3, model_dir)  # create Trainer

    train_data = tf.data.Dataset.from_tensor_slices((tf.zeros([10, 32, 32, 32, 1]),
                                                     tf.ones([10, 32, 32, 32, 1])))  # mock training dataset

    valid_data = tf.data.Dataset.from_tensor_slices((tf.zeros([10, 32, 32, 32, 1]),
                                                     tf.ones([10, 32, 32, 32, 1])))  # mock validation dataset

    batch_size = 1

    train_ds = tf.distribute.get_strategy().experimental_distribute_dataset(
        train_data.batch(batch_size).repeat())  # batch and repeat training dataset
    valid_ds = tf.distribute.get_strategy().experimental_distribute_dataset(
        valid_data.batch(batch_size).repeat())  # batch and repeat valid dataset

    loss_fn = tf.keras.losses.binary_crossentropy  # define loss function
    accuracy_fn = tf.keras.metrics.binary_accuracy  # define accuracy function

    print('\nTrain 2 epochs and stop')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=batch_size, EPOCHS=2, save_step=1)

    print('\nResume training from 2nd epoch')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=batch_size, EPOCHS=4, save_step=1)

    print('\nTry to resume training when achieved number epochs')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=batch_size, EPOCHS=4, save_step=1)

    shutil.rmtree(model_dir)

    return


def training_create_unet3d_class():
    """Training example of 3D-Unet using Trainer class"""

    tf.random.set_seed(1)

    model_dir = 'tf_ckpt'  # directory where to save checkpoints

    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    model = create_unet3d_class(input_shape=[32, 32, 32, 1],
                                n_convs=1,
                                n_filters=[8, 16],
                                ksize=[3, 3, 3],
                                padding='same',
                                pooling='avg',
                                norm='batch_norm',
                                dropout=[0],
                                upsampling=True,
                                activation='relu',
                                depth=2)  # create U-Net 3D model

    trainer = Trainer(model, tf.keras.optimizers.Adam, 1e-3, model_dir)  # create Trainer

    train_data = tf.data.Dataset.from_tensor_slices((tf.zeros([10, 32, 32, 32, 1]),
                                                     tf.ones([10, 32, 32, 32, 1])))  # mock training dataset

    valid_data = tf.data.Dataset.from_tensor_slices((tf.zeros([10, 32, 32, 32, 1]),
                                                     tf.ones([10, 32, 32, 32, 1])))  # mock validation dataset

    batch_size = 1

    train_ds = tf.distribute.get_strategy().experimental_distribute_dataset(
        train_data.batch(batch_size).repeat())  # batch and repeat training dataset
    valid_ds = tf.distribute.get_strategy().experimental_distribute_dataset(
        valid_data.batch(batch_size).repeat())  # batch and repeat valid dataset

    loss_fn = tf.keras.losses.binary_crossentropy  # define loss function
    accuracy_fn = tf.keras.metrics.binary_accuracy  # define accuracy function

    print('\nTrain 2 epochs and stop')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=batch_size, EPOCHS=2, save_step=1)

    print('\nResume training from 2nd epoch')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=batch_size, EPOCHS=4, save_step=1)

    print('\nTry to resume training when achieved number epochs')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=batch_size, EPOCHS=4, save_step=1)

    shutil.rmtree(model_dir)

    return


if __name__ == '__main__':
    training_create_unet3d()
    training_create_unet3d_class()
