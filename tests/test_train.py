import os
import shutil

import tensorflow as tf
from utils.layers_func import create_unet3d
from utils.train import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Comment to use GPU


def testTrainer():
    """Testing Trainer Class"""

    model_dir = 'tf_ckpt'

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
                          depth=2)

    trainer = Trainer(model, tf.keras.optimizers.Adam, 1e-3, model_dir)

    train_data = tf.data.Dataset.from_tensor_slices((tf.random.uniform([10, 32, 32, 32, 1]),
                                                     tf.random.uniform([10, 32, 32, 32, 1])))

    valid_data = tf.data.Dataset.from_tensor_slices((tf.random.uniform([10, 32, 32, 32, 1]),
                                                     tf.random.uniform([10, 32, 32, 32, 1])))

    train_ds = tf.distribute.get_strategy().experimental_distribute_dataset(train_data.batch(1).repeat())
    valid_ds = tf.distribute.get_strategy().experimental_distribute_dataset(valid_data.batch(1).repeat())

    loss_fn = tf.keras.losses.binary_crossentropy
    accuracy_fn = tf.keras.metrics.binary_accuracy

    print('\nTrain 2 epochs and stop')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=1, EPOCHS=2, save_step=1)

    print('\nResume training from 2nd')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=1, EPOCHS=4, save_step=1)

    print('\nTry to resume training when achieved number epochs')

    trainer.train(train_ds=train_ds, valid_ds=valid_ds, train_size=10, validation_size=10, loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn, BATCH_SIZE=1, EPOCHS=4, save_step=1)

    shutil.rmtree(model_dir)
