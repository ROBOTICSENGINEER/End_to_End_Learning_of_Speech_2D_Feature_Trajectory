""" data_manager.py
    HANND with the HBS lab at UT Dallas

This module contains methods to manage the importing and shaping of NN data

The GSCD is found at https://storage.cloud.google.com/
    download.tensorflow.org/data/speech_commands_v0.02.tar.gz

More information can be found at: https://arxiv.org/abs/1804.03209

Data should be unzipped into a directory called data_speech_commands, with
    each word in its own directory
    
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

import errno
import keras
import logging
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import scipy.io.wavfile
import scipy.signal


class DataManager(object):
    """
    Manages the generation, loading, and processing of data sets

    Load in sets of (file_path, class) data by constructing an instance of the object

    Access the testing, validation, and training sets through the members test_set, val_set, and train_set, respectively
    """
    DEFAULT_MAP = {
        'zero': [1, 1, 1, 1, 1],
        'one': [1, 0, 1, 1, 1],
        'two': [1, 0, 0, 1, 1],
        'three': [1, 0, 0, 0, 1],
        'four': [1, 0, 0, 0, 0],
        'five': [0, 0, 0, 0, 0],
        'on': [1, 1, 1, 1, 1],
        'off': [0, 0, 0, 0, 0]
    }

    DEFAULT_LISTS_DIR = 'data_lists'
    DEFAULT_DATA_PATHS = ['data_speech_commands']
    DEFAULT_TARGET_LIST = [k for k in DEFAULT_MAP]  # ['one', 'two', 'three', 'four', 'five', 'zero', 'on', 'off']
    DEFAULT_BACKGROUND_NOISE = '_background_noise_'

    DEFAULT_TESTF = 'testing_list.txt'
    DEFAULT_VALF = 'validation_list.txt'
    DEFAULT_TRAINF = 'training_list.txt'

    def __init__(self, lists_dir=DEFAULT_LISTS_DIR, target_list=DEFAULT_TARGET_LIST[:],
                 data_paths=DEFAULT_DATA_PATHS[:], background_noise=DEFAULT_BACKGROUND_NOISE, testf=DEFAULT_TESTF,
                 valf=DEFAULT_VALF, trainf=DEFAULT_TRAINF, set_size_multiplier=1.0, unknown_ratio=0.5):
        """
        Loads in the data sets as specified

        :param lists_dir: The directory in which generated lists are stored
        :param target_list: The list of target words
        :param data_paths: A list of paths to the datasets to use. Paths should be folders in the current directory
            without any relative or absolute parts, such as ./ or /home/nvidia/...
        :param background_noise: The name of the background noise dir, which is ignored
        :param testf: The name of test list files
        :param valf: The name of validation list files
        :param trainf: The name of train list files
        :param set_size_multiplier: Reduces the data set size by this amount
        :param unknown_ratio: The fraction of total data that is unknown in class
        """

        # These variables should not change for the lifetime of the object
        self.LISTS_DIR = lists_dir
        self.DATA_PATHS = data_paths
        self.BACKGROUND_NOISE = background_noise
        self.TARGET_LIST = target_list
        self.TESTF = testf
        self.VALF = valf
        self.TRAINF = trainf

        # Generate sets as requested
        self.test_set, self.val_set, self.train_set = self._get_sets(set_size_multiplier=set_size_multiplier,
                                                                     unknown_ratio=unknown_ratio)

    def _generate_all_lists(self, data_path, test_fraction=0.09, val_fraction=0.09):
        """
        Returns the generated train, test, and validate lists from the data set

        :param data_path: The path to the data to generate from
        :return: (test list, validate list, train list)
        """
        assert test_fraction >= 0.0, 'test_fraction must be between 0 and 1'
        assert test_fraction <= 1.0, 'test_fraction must be between 0 and 1'
        assert val_fraction >= 0.0, 'val_fraction must be between 0 and 1'
        assert val_fraction <= 1.0, 'val_fraction must be between 0 and 1'
        assert test_fraction + val_fraction <= 1.0, 'test_fraction + val_fraction must be between 0 and 1'

        _all = [][:]

        logging.debug('Walking %s data to generate lists' % data_path)
        for root, dirs, files in os.walk(data_path):
            for d in dirs:
                # Ignore background noise
                if d == self.BACKGROUND_NOISE:
                    continue
                # Accept all wav files
                _all.extend(os.path.join(d, f) for f in os.listdir(os.path.join(root, d)) if f.endswith('.wav'))

        logging.debug('Shuffling lists for %s' % data_path)
        random.shuffle(_all)

        logging.debug('Generating test list for %s' % data_path)
        tmp_test = _all[:int(len(_all)*test_fraction)]

        logging.debug('Generating validation list for %s' % data_path)
        tmp_val = _all[int(len(_all) * test_fraction):int(len(_all) * (val_fraction + test_fraction))]

        logging.debug('Generating training list for %s' % data_path)
        tmp_train = _all[int(len(_all) * (val_fraction + test_fraction)):]

        return tmp_test, tmp_val, tmp_train

    def _generate_train_list(self, data_path, test_list, val_list):
        """
        Returns the generated train list based off of an existing test and validate list

        :param data_path: The path to the data to generate from
        :param test_list: The list of test files
        :param val_list: The list of validate files
        :return: The list of training files
        """
        tmp_test = [][:]

        logging.debug('Walking %s data to generate training list' % data_path)
        for root, dirs, files in os.walk(data_path):
            for d in dirs:
                # Ignore background noise
                if d == self.BACKGROUND_NOISE:
                    continue
                logging.debug('Processing %s' % d)
                # Accept all wav files not in existing lists
                tmp_test.extend(os.path.join(d, f) for f in os.listdir(os.path.join(root, d)) if f.endswith('.wav')
                                and os.path.join(d, f) not in test_list and os.path.join(d, f) not in val_list)

        return tmp_test

    def _get_test_and_validate(self, data_path):
        """
        Returns the test and validate lists from the data set if they are given, else returns (None, None)

        :param data_path: The path to the data to check
        :return: (test list, validate list) if they exist, else (None, None)
        """

        tmp_test = [][:]
        tmp_val = [][:]

        # get test and validate lists
        logging.debug('Getting test list for %s' % data_path)
        pth = os.path.join(data_path, self.TESTF)
        try:
            with open(pth, 'r') as f:
                tmp_test.extend(f.readlines())
        except IOError:
            logging.debug('Did not find test list at %s, will generate' % pth)
            return None, None

        logging.debug('Getting validation list for %s' % data_path)
        pth = os.path.join(data_path, self.VALF)
        try:
            with open(pth, 'r') as f:
                tmp_val.extend(f.readlines())
        except IOError:
            logging.debug('Did not find validation list at %s, will generate' % pth)
            return None, None

        return tmp_test, tmp_val

    def _save_file(self, lst, dataset, name):
        """
        Saves a data list in a file for later

        :param lst: The list to save
        :param dataset: The name of the dataset for this list
        :param name: The file name to save to
        :return: None
        """
        with open(os.path.join(self.LISTS_DIR, dataset, name), 'w') as f:
            for i in lst:
                f.write(i.strip())
                f.write('\n')

    def _get_sets(self, set_size_multiplier=1.0, unknown_ratio=0.5):
        """
        Returns lists of (file path, string class) data for test, validate, and train data

        :return: (test set, validate set, train set)
        """
        assert set_size_multiplier >= 0.0, 'set_size_multiplier must be between 0 and 1'
        assert set_size_multiplier <= 1.0, 'set_size_multiplier must be between 0 and 1'
        assert unknown_ratio >= 0.0, 'unknown_ratio must be between 0 and 1'
        assert set_size_multiplier <= 1.0, 'unknown_ratio must be between 0 and 1'

        test_set_target = [][:]
        val_set_target = [][:]
        train_set_target = [][:]
        test_set_unknown = [][:]
        val_set_unknown = [][:]
        train_set_unknown = [][:]

        # Ensure lists dir exists
        def safe_mkdir(pth):
            try:
                os.mkdir(pth)
            except OSError as err:
                if err.errno != errno.EEXIST:
                    raise
        safe_mkdir(self.LISTS_DIR)

        for dataset in self.DATA_PATHS:
            logging.debug('Processing %s data' % dataset)
            save = False

            # Ensure dataset dir exists
            safe_mkdir(os.path.join(self.LISTS_DIR, dataset))

            try:
                # For all datasets, first try to read pre-generated files for testing and validation
                logging.debug('Reading test list for %s' % dataset)
                with open(os.path.join(self.LISTS_DIR, dataset, self.TESTF), 'r') as testf:
                    test = [f.strip() for f in testf.readlines()]
                logging.debug('Reading validation list for %s' % dataset)
                with open(os.path.join(self.LISTS_DIR, dataset, self.VALF), 'r') as valf:
                    val = [f.strip() for f in valf.readlines()]
            except IOError:
                logging.debug('Test and validate lists for %s need to be fetched' % dataset)
                save = True
                test, val = self._get_test_and_validate(dataset)

            try:
                # For all datasets, first try to read the pre-generated file for train
                logging.debug('Reading train list for %s' % dataset)
                with open(os.path.join(self.LISTS_DIR, dataset, self.TRAINF), 'r') as trainf:
                    train = [f.strip() for f in trainf.readlines()]
            except IOError:
                logging.debug('Train list for %s needs to be generated' % dataset)
                save = True
                if test is not None and val is not None:
                    train = self._generate_train_list(dataset, test, val)
                else:
                    train = None

            # If nothing is set, need to generate all
            if test is None or val is None or train is None:
                logging.debug('Need to generate all lists for %s' % dataset)
                save = True
                test, val, train = self._generate_all_lists(dataset)

            # Save files if anything was fetched or generated
            if save:
                logging.debug('Saving test file for %s' % dataset)
                self._save_file(test, dataset, self.TESTF)
                logging.debug('Saving validation file for %s' % dataset)
                self._save_file(val, dataset, self.VALF)
                logging.debug('Saving training file for %s' % dataset)
                self._save_file(train, dataset, self.TRAINF)

            # Extend existing sets with the read sets
            def _f(f):
                """
                Modifies the file path to include the dataset name

                :param f: The file path without the dataset name
                :return: The file path with the dataset name
                """
                return os.path.join(dataset, f)

            def _class(f):
                """
                Gets the class from the file path
                :param f: The file path without the dataset name
                :return: The class name
                """
                return f.split('/')[0]

            logging.debug('Appending test set for %s' % dataset)
            test_set_target.extend((_f(f), _class(f)) for f in test if _class(f) in self.TARGET_LIST)
            test_set_unknown.extend((_f(f), _class(f)) for f in test if _class(f) not in self.TARGET_LIST)

            logging.debug('Appending validation set for %s' % dataset)
            val_set_target.extend((_f(f), _class(f)) for f in val if _class(f) in self.TARGET_LIST)
            val_set_unknown.extend((_f(f), _class(f)) for f in val if _class(f) not in self.TARGET_LIST)

            logging.debug('Appending train set for %s' % dataset)
            train_set_target.extend((_f(f), _class(f)) for f in train if _class(f) in self.TARGET_LIST)
            train_set_unknown.extend((_f(f), _class(f)) for f in train if _class(f) not in self.TARGET_LIST)

        # Do any requested post processing (set size and unknown ratio)
        # Set size
        def uk_size(_t_set):
            """
            Returns the needed size for the unknown part of the data set

            :param _t_set: The target for this set, already resized
            :return: The number of unknown itesm
            """
            return unknown_ratio * len(_t_set) / (1 - unknown_ratio)

        def resize(_set, amt):
            """
            Resizes a set to the desired size
            :param _set: The set to resize
            :param amt: The length to which to resize the set
            :return: The resized set
            """
            return random.sample(_set, int(amt))

        logging.debug('Resizing test set')
        test_set_target = resize(test_set_target, len(test_set_target) * set_size_multiplier)
        test_set_unknown = resize(test_set_unknown, uk_size(test_set_target))

        logging.debug('Resizing validation set')
        val_set_target = resize(val_set_target, len(val_set_target) * set_size_multiplier)
        val_set_unknown = resize(val_set_unknown, uk_size(val_set_target))

        logging.debug('Resizing training set')
        train_set_target = resize(train_set_target, len(train_set_target) * set_size_multiplier)
        train_set_unknown = resize(train_set_unknown, uk_size(train_set_target))

        # Combine and return sets
        logging.debug('Finalizing sets')
        test_set = test_set_target + test_set_unknown
        random.shuffle(test_set)

        val_set = val_set_target + val_set_unknown
        random.shuffle(val_set)

        train_set = train_set_target + train_set_unknown
        random.shuffle(train_set)

        return test_set, val_set, train_set


def wav_from_file(fpath):
    """
    Reads a wav file while enforcing 1s length at 16000 sampling rate

    :param fpath: The file to read
    :return: An np array of the wav file
    """
    wav = np.array(scipy.io.wavfile.read(fpath.strip())[1], dtype=float)/32768
    if len(wav) > 16000:
        wav = wav[:16000]
    else:
        wav = np.pad(wav, (0, 16000 - len(wav)), 'constant', constant_values=(0, 0))
    return wav


def spectrogram_plot_from_wav(wav):
    """
    Converts a wav np array into a log spectrogram

    :param wav: The np array of the wav audio
    :return: An tuple of (frequencies, times, spectrogram)
    """
    # Convert to spectrogram
    f, t, s = scipy.signal.spectrogram(wav, 16000)

    # Take log, add a very small amount to avoid log(0)
    spect = np.log(np.abs(s) + 1e-10)

    return f, t, spect


def spectrogram_from_wav(wav):
    """
    Converts a wav np array into a log spectrogram

    :param wav: The np array of the wav audio
    :return: An np array of the spectrogram
    """
    return spectrogram_plot_from_wav(wav)[2]


def spectrogram_from_file(fpath):
    """
    Reads a wav file while enforcing 1s length at 16000 sampling rate and returns the log spectrogram

    :param fpath: The file to read
    :return: An np array of the spectrogram
    """
    return spectrogram_from_wav(wav_from_file(fpath))


def plot_spectrogram(f, t, spect, output):
    """
    Plots a spectrogram

    :param f: The np array of frequencies
    :param t: The np array of times
    :param spect: The log spectrogram
    :return: None
    """
    plt.clf()

    figure(num=1, figsize=(12, 6), dpi=80, facecolor='w', edgecolor='k')
    sg = plt.subplot(1, 2, 1)
    plt.pcolormesh(t, f, spect)
    sg.set_title('Sprectrogram')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.show(block=False)

    to_plot = np.maximum(0, np.minimum(1, 1 - output))

    labels = ['thumb', 'index', 'middle', 'ring', 'pinky']
    index = np.arange(len(labels))
    bg = plt.subplot(1, 2, 2)
    bg.set_ylim([0, 1])
    plt.bar(index, to_plot)
    bg.set_title('Hand Position')
    plt.xticks(index, labels)
    plt.ylabel('Actuation')
    plt.xlabel('Finger')
    plt.show(block=False)
    plt.pause(0.01)



class SpectrogramToStringClassSequence(keras.utils.Sequence):
    """
    Keras Sequence to dynamically load in spectrogram and class data for a SpectrogramToStringClassCNN
    """

    def __init__(self, _set, targets, batch_size=512):
        """
        Creates the generator

        :param _set: The list of (path, class) data
        :param targets: The list of target words
        :param batch_size: The batch size, must be smaller than the dataset size
        """
        assert len(_set) >= batch_size, 'batch_size must be less than the set size'

        self._set = _set
        self._labels = targets + ['unknown']
        self._batch_size = batch_size

    def __len__(self):
        """
        Returns the length of the set in batched
        :return: The length of the set in batches
        """
        return int(np.ceil(len(self._set) / float(self._batch_size)))

    def __getitem__(self, idx):
        """
        Returns the idxth batch of data

        :param idx: The batch to get
        :return: (X data, Y data)
        """
        l = idx * self._batch_size      # low bound of batch
        h = (idx+1) * self._batch_size  # high bound of batch
        batch_x = [spectrogram_from_file(f) for f, _ in self._set[l:h]]
        batch_y = [self._labels.index(c) if c in self._labels else len(self._labels) - 1 for _, c in self._set[l:h]]

        return np.array(batch_x).reshape(-1, 129, 71, 1), keras.utils.to_categorical(batch_y, len(self._labels))


class SpectrogramToVoltage(keras.utils.Sequence):
    """
    Keras Sequence to dynamically load in spectrogram and class data for a SpectrogramToStringClassCNN
    """

    def __init__(self, _set, target_map, batch_size=512):
        """
        Creates the generator

        :param _set: The list of (path, class) data
        :param target_map: The map of target words to output vectors
        :param batch_size: The batch size, must be smaller than the dataset size
        """
        assert len(_set) >= batch_size, 'batch_size must be less than the set size'

        self._map = target_map
        self._unknown = np.zeros(5, dtype=float)
        self._set = _set
        self._batch_size = batch_size

    def __len__(self):
        """
        Returns the length of the set in batched
        :return: The length of the set in batches
        """
        return int(np.ceil(len(self._set) / float(self._batch_size)))

    def __getitem__(self, idx):
        """
        Returns the idxth batch of data

        :param idx: The batch to get
        :return: (X data, Y data)
        """
        l = idx * self._batch_size      # low bound of batch
        h = (idx+1) * self._batch_size  # high bound of batch
        batch_x = [spectrogram_from_file(f) for f, _ in self._set[l:h]]
        batch_y = [np.array(self._map[c], dtype=float) if c in self._map else self._unknown for _, c in self._set[l:h]]

        return np.array(batch_x).reshape(-1, 129, 71, 1), np.array(batch_y).reshape(-1, 5, 1)

