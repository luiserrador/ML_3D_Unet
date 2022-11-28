import os.path

import tensorflow as tf
import numpy as np


class Trainer:
    def __init__(self, model, optimizer, learning_rate, model_dir):
        self.model = model

        with tf.distribute.get_strategy().scope():
            self.train_accuracy = tf.keras.metrics.Sum()
            self.valid_accuracy = tf.keras.metrics.Sum()
            self.train_loss = tf.keras.metrics.Sum()
            self.valid_loss = tf.keras.metrics.Sum()

            self.optimizer = optimizer(learning_rate=learning_rate)

            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.model)
            manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
            ckpt.restore(manager.latest_checkpoint)

        if manager.latest_checkpoint:
            print("Restored from {}".format(manager.latest_checkpoint))
            self.step = np.load(os.path.join(model_dir, "step_scratch.npy"))
        else:
            print("Initializing from scratch.")
            self.step = 0

    @tf.function
    def train_step(self, data_iter):
        def train_step_fn(images, labels):
            with tf.GradientTape() as tape:
                probabilities = self.model(images, training=True)
                loss = self.loss_fn(tf.cast(labels, tf.float32), probabilities, epoch=tf.cast(self.epoch, tf.float32),
                                    EPOCHS=tf.cast(self.EPOCHS, tf.float32))
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
            accuracy = self.accuracy_fn(tf.cast(labels, tf.float32), probabilities)
            self.train_accuracy.update_state(accuracy)
            # train_binary_accuracy.update_state(labels, probabilities)
            self.train_loss.update_state(loss)

        for _ in tf.range(self.STEPS_PER_CALL):
            tf.distribute.get_strategy().run(train_step_fn, next(data_iter))

    @tf.function
    def valid_step(self, data_iter):
        def valid_step_fn(images, labels):
            probabilities = self.model(images, training=False)
            loss = self.loss_fn(tf.cast(labels, tf.float32), probabilities, epoch=tf.cast(self.epoch, tf.float32),
                                EPOCHS=tf.cast(self.EPOCHS, tf.float32))
            accuracy = self.accuracy_fn(tf.cast(labels, tf.float32), probabilities)
            self.valid_accuracy.update_state(accuracy)
            self.valid_loss.update_state(loss)
        for _ in tf.range(self.VALIDATION_STEPS_PER_CALL):
            tf.distribute.get_strategy().run(valid_step_fn, next(data_iter))
