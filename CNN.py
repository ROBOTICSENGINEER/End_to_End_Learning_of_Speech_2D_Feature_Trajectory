""" CNN.py

UT Dallas HANND Senior Design: Akshay Chitale, Rodrigo Avila, Clarissa Curry, and Cameron Ford

Cite as:

IEEE
M. Jafarzadeh and Y. Tadesse, "End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands", in 2020 Second International Conference on Transdisciplinary AI (TransAI), 2020, pp. 25-33.

ACM
Jafarzadeh, M. and Tadesse, Y., 2020. End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands. In 2020 Second International Conference on Transdisciplinary AI (TransAI). IEEE, pp. 25-33.

BibTeX
@inbook{jafarzadeh_End_to_End_2020,
title={End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands},
booktitle={2020 Second International Conference on Transdisciplinary AI (TransAI)},
publisher={IEEE},
author={Jafarzadeh, Mohsen and Tadesse, Yonas},
ear={2020},
pages={25-33}
}


"""

import data_manager as dm
import json
import keras
import logging
import matplotlib.pyplot as plt
import tensorflow as tf


def global_config():
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    keras.backend.tensorflow_backend.set_session(tf.Session(config=config))


class _BaseCNN(object):
    """
    Base CNN for all CNNs, since these functions should all be the same
    """

    # Constants for running on Jetson
    VERBOSE = 1
    USE_MULTIPROCESSING = True
    WORKERS = 4
    MAX_QUEUE_SIZE = 4

    # Constant for all models
    METRICS = ['acc', 'mean_squared_error']

    def __init__(self, dataset=None, save_prefix=None):
        """
        Initializes the CNN

        :param dataset: The dm.DataManager data set to use
        :param save_prefix: The prefix for where to save result files
        """
        self._save_prefix = save_prefix
        self._dataset = dataset

        # Initially, model is not loaded
        self._model = None
        self._history = None

    def _build_model(self):
        raise NotImplementedError('This method should be overridden by a subclass with a specific Keras Model')

    def _build_sequence(self, _set, batch_size):
        raise NotImplementedError('This method should be overridden by a subclass with a specific Keras Sequence')

    def save_model(self):
        """
        Saves the model in <self._save_prefix>.hdf5

        :return: None
        """
        assert self._model is not None, 'Model not yet built or loaded'
        self._model.save(self._save_prefix + '.hdf5')

    def save_history(self):
        """
        Saves the training history in <self._save_prefix>.json
        :return:
        """
        assert self._history is not None, 'History not yet generated'
        with open(self._save_prefix + '.json', 'w') as f:
            json.dump(self._history.history, f)

    def save_history_plots(self):
        """
        Saves plots generated from the training history in <self._save_prefix>-acc.png and <self._save_prefix>-loss.png
        :return:
        """
        assert self._history is not None, 'History not yet generated'

        # Save each metric plot
        for metric in ['loss'] + self.METRICS:
            plt.clf()
            plt.plot(self._history.history[metric])
            plt.plot(self._history.history['val_' + metric])
            plt.title('model ' + metric)
            plt.ylabel(metric)
            plt.xlabel('epoch')
            plt.legend(['train', 'validation'], loc='upper left')
            plt.savefig(self._save_prefix + '-' + metric + '.png')

    def save_all(self):
        """
        Convenience method to save model, history, and plots

        :return: None
        """
        self.save_model()
        self.save_history()
        self.save_history_plots()

    def train(self, num_epochs=100, batch_size=512):
        """
        Trains the network based on dataset given at construction

        :param num_epochs: The number of training epochs to perform
        :param batch_size: The batch size for the data sequences
        :return: None
        """
        assert self._dataset is not None, 'No dataset given'

        # Build model
        logging.debug('Building Model...')
        self._build_model()

        # Set up sequences
        logging.debug('Setting up validation sequence...')
        val_gen = self._build_sequence(self._dataset.val_set, batch_size=batch_size)
        logging.debug('Setting up training sequence...')
        train_gen = self._build_sequence(self._dataset.train_set, batch_size=batch_size)

        # Run training
        logging.debug('Starting training...')
        self._history = self._model.fit_generator(train_gen,
                                                  validation_data=val_gen,
                                                  epochs=num_epochs,
                                                  verbose=self.VERBOSE,
                                                  use_multiprocessing=self.USE_MULTIPROCESSING,
                                                  workers=self.WORKERS,
                                                  max_queue_size=self.MAX_QUEUE_SIZE)
        logging.debug('Finished training.')

    def test(self, batch_size=512):
        """
        Returns the loss and accuracy over the test set

        :param batch_size: The batch size for the data sequences
        :return: A list of [loss, accuracy] over the training set
        """
        assert self._model is not None, 'Model not yet built or loaded'

        # Set up sequences
        logging.debug('Setting up test sequence...')
        test_gen = self._build_sequence(self._dataset.test_set, batch_size=batch_size)

        # Run test
        scores = self._model.evaluate_generator(test_gen,
                                                verbose=self.VERBOSE,
                                                use_multiprocessing=self.USE_MULTIPROCESSING,
                                                workers=self.WORKERS,
                                                max_queue_size=self.MAX_QUEUE_SIZE)

        return scores

    def load_model(self):
        self._model = keras.models.load_model(self._save_prefix + '.hdf5')

    def predict(self, input_data):
        """
        Calls predict on the model with already processed input data not in the data set. This is useful for live data

        :param input_data: Data in the format of the input of the network
        :return: A flattened array of the prediction output
        """
        return self._model.predict(input_data).flatten()


class SpectrogramToStringClassCNN(_BaseCNN):
    """
    Models and runs a CNN with spectrogram input and string class output using a SpectrogramToStringClassSequence

    The input for this CNN is expected to be spectrograms of shape (129, 71, 1), which are obtained using
    scipy.signal.spectrogram on 1s long audio at 16kHz sampling rate.

    The output for this CNN is a string classification
    """

    def __init__(self, targets, dataset=None, save_prefix=None):
        """
        Initializes the CNN

        :param dataset: The dm.DataManager data set to use
        :param targets: The list of target words
        :param save_prefix: The prefix for where to save result files
        """
        super(SpectrogramToStringClassCNN, self).__init__(dataset=dataset, save_prefix=save_prefix)

        self._targets = targets

        # Constant for this NN
        self._input_shape = (129, 71, 1)
        self._drop_out_rate = 0.5
        self._learning_rate = 0.001

    def _build_model(self):
        """
        Builds the Keras model for this CNN

        :return: None
        """
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = keras.layers.Conv2D(8, (10, 7), padding='valid', activation='relu', strides=1)(input_tensor)
        x = keras.layers.MaxPooling2D((7, 5))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(256, (7, 5), padding='valid', activation='relu', strides=1)(x)
        x = keras.layers.MaxPooling2D((5, 3))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(self._drop_out_rate)(x)
        output_tensor = keras.layers.Dense(len(self._targets) + 1, activation='softmax')(x)  # + 1 is for 'unknown'

        self._model = keras.Model(input_tensor, output_tensor)

        self._model.compile(loss=keras.losses.categorical_crossentropy,
                            optimizer=keras.optimizers.Adam(lr=self._learning_rate),
                            metrics=self.METRICS)

        self._model.summary()

    def _build_sequence(self, _set, batch_size):
        return dm.SpectrogramToStringClassSequence(_set, self._targets, batch_size)


class SpectrogramToVoltageCNN(_BaseCNN):
    """
    Models and runs a CNN with spectrogram input and string class output using a SpectrogramToStringClassSequence

    The input for this CNN is expected to be spectrograms of shape (129, 71, 1), which are obtained using
    scipy.signal.spectrogram on 1s long audio at 16kHz sampling rate.

    The output for this CNN is a string classification
    """

    def __init__(self, target_map, dataset=None, save_prefix=None):
        """
        Initializes the CNN

        :param dataset: The dm.DataManager data set to use
        :param targets: The list of target words
        :param save_prefix: The prefix for where to save result files
        """
        super(SpectrogramToVoltageCNN, self).__init__(dataset=dataset, save_prefix=save_prefix)

        self._map = target_map
        # Constant for this NN
        self._input_shape = (129, 71, 1)
        self._drop_out_rate = 0.5
        self._learning_rate = 0.001

    def _build_model(self):
        """
        Builds the Keras model for this CNN

        :return: None
        """
        input_tensor = keras.layers.Input(shape=self._input_shape)
        x = keras.layers.Conv2D(8, (10, 7), padding='valid', activation='relu', strides=1)(input_tensor)
        x = keras.layers.MaxPooling2D((7, 5))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Conv2D(128, (7, 5), padding='valid', activation='relu', strides=1)(x)
        x = keras.layers.MaxPooling2D((5, 3))(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(64, activation='relu')(x)
        x = keras.layers.Dropout(self._drop_out_rate)(x)
        x = keras.layers.Dense(5, activation='relu')(x)  #+ 1 is for 'unknown'
        x = keras.layers.Reshape((5, 1))(x)
        output_tensor = keras.layers.MaxPooling1D(1)(x)

        self._model = keras.Model(input_tensor, output_tensor)

        self._model.compile(loss=keras.losses.mean_squared_error,
                            optimizer=keras.optimizers.Adam(lr=self._learning_rate),
                            metrics=self.METRICS)

        self._model.summary()

    def _build_sequence(self, _set, batch_size):
        return dm.SpectrogramToVoltage(_set, self._map, batch_size)


if __name__ == '__main__':
    # Small test script to run select samples through a pre-trained SpectrogramToVoltageCNN
    m = keras.models.load_model('results/test7_spectro.hdf5')
    waves = ['zero/0b7ee1a0_nohash_1.wav', 'one/0a196374_nohash_0.wav', 'two/0a196374_nohash_1.wav',
             'three/0a196374_nohash_1.wav', 'four/9e92ef0c_nohash_1.wav', 'five/0ab3b47d_nohash_2.wav',
             'on/0a5636ca_nohash_2.wav', 'off/0b7ee1a0_nohash_1.wav']

    for x in waves:
        f = 'data_speech_commands/' + x
        val = dm.spectrogram_from_file(f)
        print('Output for ' + x + ':')
        print(m.predict(val))
    print('Test finished')
