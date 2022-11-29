import os.path
import time
import os

import tensorflow as tf
from tensorflow.keras.layers import Lambda, Activation, concatenate
from tensorflow.keras.models import Model
import numpy as np

from utils.train import Trainer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Comment to use GPU


class Trainer_KD:
    """ Trainer for Knowledge Distillation

    Parameters
    -----------
    model : Tensorflow Model
        Model to train
    teacher_model : Tensorflow Model
        Teacher model from which to distil knowledge
    temperature : int
        Temperature to use to soften logits
    optimizer : Tensorflow Optimizer
        Optimizer to use
    learning_rate : float
        Learning rate
    model_dir : str
        Directory where to save the checkpoints / models
    """

    def __init__(self, student_model, teacher_model, temperature, optimizer, learning_rate, model_dir):

        self.student_model = _get_student_soften(student_model, temperature)
        self.teacher_model = _get_teacher_soften(teacher_model, temperature)
        self.student_model_scratch = student_model
        self.teacher_model_scratch = teacher_model

        with tf.distribute.get_strategy().scope():
            self.train_accuracy = tf.keras.metrics.Sum()
            self.valid_accuracy = tf.keras.metrics.Sum()
            self.train_loss = tf.keras.metrics.Sum()
            self.valid_loss = tf.keras.metrics.Sum()

            self.optimizer = optimizer(learning_rate=learning_rate)

            ckpt = tf.train.Checkpoint(optimizer=self.optimizer, net=self.student_model)
            self.manager = tf.train.CheckpointManager(ckpt, model_dir, max_to_keep=3)
            ckpt.restore(self.manager.latest_checkpoint)

        check_dir = os.path.exists(model_dir)
        if not check_dir:
            os.makedirs(model_dir)

        self.step_dir = os.path.join(model_dir, "step.npy")
        self.student_model_dir = model_dir

        self.student_model_dir_scratch = model_dir + '_student_scratch'
        self.teacher_model_dir_scratch = model_dir + '_teacher_scratch'

        self.trainer_student_scratch = Trainer(self.student_model_scratch, optimizer, learning_rate, self.student_model_dir_scratch)
        self.trainer_teacher_scratch = Trainer(self.student_model_scratch, optimizer, learning_rate, self.teacher_model_dir_scratch)

        if self.manager.latest_checkpoint:
            print("Restored from {}".format(self.manager.latest_checkpoint))
            self.step = np.load(self.step_dir)
        else:
            print("Initializing from scratch.")
            self.step = 0

    def train(self, train_ds, valid_ds, train_size, validation_size, loss_fn, accuracy_fn, BATCH_SIZE, EPOCHS,
              save_step=1):
        """ Train the model

        Parameters
        -----------
        train_ds : tf.data.Dataset
            Training dataset
        valid_ds : tf.data.Dataset
            Validation dataset
        train_size : scalar
            Size of the training dataset
        validation_size : scalar
            Size of the validation dataset
        loss_fn : function
            Loss function
        accuracy_fn : function
            Accuracy function
        BATCH_SIZE : int
            Batch size
        EPOCHS : int
            Number of epochs to train
        """

        print('Train Teacher')

        self.trainer_teacher_scratch.train(train_ds, valid_ds, train_size, validation_size, loss_fn, accuracy_fn,
                                           BATCH_SIZE, EPOCHS, save_step)

        print('Train Student')

        self.trainer_student_scratch.train(train_ds, valid_ds, train_size, validation_size, loss_fn, accuracy_fn,
                                           BATCH_SIZE, EPOCHS, save_step)

        self.EPOCHS = EPOCHS

        with tf.distribute.get_strategy().scope():
            self.loss_fn = loss_fn
            self.accuracy_fn = accuracy_fn

        self.STEPS_PER_CALL = STEPS_PER_EPOCH = train_size // BATCH_SIZE
        self.VALIDATION_STEPS_PER_CALL = validation_size // BATCH_SIZE
        self.epoch = self.step // STEPS_PER_EPOCH
        epoch_steps = 0
        epoch_start_time = time.time()

        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        train_data_iter = iter(train_ds)
        valid_data_iter = iter(valid_ds)

        print('KD from Teacher to Student')

        if self.epoch < self.EPOCHS:

            while True:
                # run training step
                print('\nEPOCH {:d}/{:d}'.format(self.epoch + 1, self.EPOCHS))
                self.train_step(train_data_iter)
                epoch_steps += self.STEPS_PER_CALL
                self.step += self.STEPS_PER_CALL
                print(epoch_steps, '/', STEPS_PER_EPOCH)

                # validation run at the end of each epoch
                if (self.step // STEPS_PER_EPOCH) > self.epoch:
                    # validation run
                    valid_epoch_steps = 0
                    self.valid_step(valid_data_iter)
                    valid_epoch_steps += self.VALIDATION_STEPS_PER_CALL

                    # compute metrics
                    history['acc'].append(self.train_accuracy.result().numpy() / (BATCH_SIZE * epoch_steps))
                    history['val_acc'].append(self.valid_accuracy.result().numpy() / (BATCH_SIZE * valid_epoch_steps))
                    history['loss'].append(self.train_loss.result().numpy() / (BATCH_SIZE * epoch_steps))
                    history['val_loss'].append(self.valid_loss.result().numpy() / (BATCH_SIZE * valid_epoch_steps))

                    # report metrics
                    epoch_time = time.time() - epoch_start_time
                    print('time: {:0.1f}s'.format(epoch_time),
                          'loss: {:0.4f}'.format(history['loss'][-1]),
                          'acc: {:0.4f}'.format(history['acc'][-1]),
                          'val_loss: {:0.4f}'.format(history['val_loss'][-1]),
                          'val_acc: {:0.4f}'.format(history['val_acc'][-1]))

                    # save checkpoint and training_step
                    if save_step and self.epoch % save_step == 0:
                        model_path = (os.path.join(self.student_model_dir, 'model_epoch_%s.h5' % (self.epoch + 1)))
                    self.student_model.save(model_path)
                    self.manager.save()
                    np.save(self.step_dir, self.step)

                    # set up next epoch
                    self.epoch = self.step // STEPS_PER_EPOCH
                    epoch_steps = 0
                    epoch_start_time = time.time()
                    self.train_accuracy.reset_states()
                    self.valid_accuracy.reset_states()
                    self.valid_loss.reset_states()
                    self.train_loss.reset_states()
                    if self.epoch >= self.EPOCHS:
                        print('Training done, {} epochs'.format(self.epoch))
                        break
        else:
            print('\nAlready trained!')

    @tf.function
    def train_step(self, data_iter):
        def train_step_fn(images, labels):
            with tf.GradientTape() as tape:
                probabilities = self.student_model(images, training=True)
                probabilities_teacher = self.teacher_model(images, training=False)
                loss = self.KD_loss(tf.cast(labels, tf.float32), probabilities, probabilities_teacher)
            grads = tape.gradient(loss, self.student_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.student_model.trainable_variables))
            accuracy = self.accuracy_fn(tf.cast(labels, tf.float32), probabilities)
            self.train_accuracy.update_state(accuracy)
            self.train_loss.update_state(loss)

        for _ in tf.range(self.STEPS_PER_CALL):
            tf.distribute.get_strategy().run(train_step_fn, next(data_iter))

    @tf.function
    def valid_step(self, data_iter):
        def valid_step_fn(images, labels):
            probabilities = self.student_model(images, training=False)
            probabilities_teacher = self.teacher_model(images, training=False)
            loss = self.KD_loss(tf.cast(labels, tf.float32), probabilities, probabilities_teacher)
            accuracy = self.accuracy_fn(tf.cast(labels, tf.float32), probabilities)
            self.valid_accuracy.update_state(accuracy)
            self.valid_loss.update_state(loss)

        for _ in tf.range(self.VALIDATION_STEPS_PER_CALL):
            tf.distribute.get_strategy().run(valid_step_fn, next(data_iter))

    @tf.function
    def KD_loss(self, y_true, y_pred, y_teacher, lambd=0.5):
        y_pred, y_pred_KD = y_pred[:, :, :, :, 0], y_pred[:, :, :, :, 1]
        # Classic cross-entropy (without temperature)
        CE_loss = self.loss_fn(y_true, y_pred)
        # KL-Divergence loss for softened output (with temperature)
        KL_loss = tf.keras.losses.kl_divergence(y_teacher, y_pred_KD)

        return lambd * CE_loss + (1 - lambd) * KL_loss


def _get_teacher_soften(teacher_model, temperature):

    with tf.distribute.get_strategy().scope():
        if teacher_model.layers[-1].name == 'unet3d_layer':
            teacher_logits = teacher_model.layers[-1].decoder.last_conv.output
        else:
            teacher_logits = teacher_model.layers[-2].output

        temperature_layer = Lambda(lambda x: x / temperature)(teacher_logits)
        sigmoid_layer = Activation('sigmoid')(temperature_layer)
        teacher_soften = Model(teacher_model.input, sigmoid_layer)

    return teacher_soften


def _get_student_soften(student_model, temperature):

    with tf.distribute.get_strategy().scope():
        if student_model.layers[-1].name == 'unet3d_layer':
            student_logits = student_model.layers[-1].decoder.last_conv.output
        else:
            student_logits = student_model.layers[-2].output

        # Compute softmax
        probs = Activation("sigmoid")(student_logits)

        # Compute softmax with softened logits
        logits_T = Lambda(lambda x: x / temperature)(student_logits)
        probs_T = Activation("sigmoid")(logits_T)

        CombinedLayers = concatenate([probs, probs_T])

        student_soften = Model(student_model.input, CombinedLayers)

    return student_soften
