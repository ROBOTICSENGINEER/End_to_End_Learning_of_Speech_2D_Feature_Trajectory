""" microphone.py

This module contains the code to take 1s 16kHz input from the microphone

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

import argparse
import errno
import numpy as np
import os
import scipy.io.wavfile
import sounddevice as sd
import sys
import time

from data_manager import DataManager as DM


class Microphone(object):
    """
    Handles all input from the microphone

    Obtains a sample at a fixed sample rate and number of channels
    """

    def __init__(self, fs=16000, channels=1):
        """
        Sets up the microphone

        :param fs: The sampling rate in Hz
        :param channels: The number of audio channels. Should be one unless there is a good reason otherwise
        """
        self._fs = fs
        self._channels = channels

        # Set the given settings
        sd.default.samplerate = fs
        sd.default.channels = channels

    def get(self, samples=16000, duration=None):
        """
        Gets an audio sample of the specified length. This funciton is blocking

        :param samples: The number of samples to record
        :param duration: If provided, overrides [samples] and is the length of the recording in seconds
        :return: The recorded audio as a numpy array of size (samples,)
        """
        if duration is None:
            num_samples = samples
        else:
            num_samples = int(duration * self._fs)
        data = sd.rec(num_samples, blocking=True)

        # Ensure data shape and that values are from -1 to 1 (clip spikes)
        data = data.reshape((num_samples,))
        data = np.maximum(data, np.zeros((num_samples,)) - 1)
        data = np.minimum(data, np.zeros((num_samples,)) + 1)

        return data

    def get_fs(self):
        """
        Returns the sampling rate

        :return: The sampling rate
        """
        return self._fs


def main(_args):
    """
    Saves audio files for each word n times in custom_speech_commands

    :param _args: The parsed command line arguments
    :return: None
    """
    # Ensure dir exists
    def safe_mkdir(pth):
        try:
            os.mkdir(pth)
        except OSError as err:
            if err.errno != errno.EEXIST:
                raise
    safe_mkdir('custom_speech_commands')

    print('Preparing to sample')
    mic = Microphone()

    def save_f(w_, audio):
        """
        Saves the file with a unique file name
        :param w_: The word spoken
        :param audio: The audio to save
        :return: None
        """
        fname = os.path.join('custom_speech_commands', w_, time.strftime("%Y_%m_%d_at_%H_%M_%S_%f.wav"))
        scipy.io.wavfile.write(fname, mic.get_fs(), audio)

    for w in _args.words:
        safe_mkdir(os.path.join('custom_speech_commands', w))

        for t in range(_args.num_times):
            sys.stdout.flush()
            print('#%d: Press enter, then speak %s' % (t, w))
            input()
            print('Recording...')
            arr = mic.get()
            print('Done recording')
            if _args.play_back:
                print('Keep file? [Y/n]: ')
                sd.play(arr, 16000)
                sd.wait()
                res = input()
                if res.upper() in ['Y', 'YES']:
                    save_f(w, arr)
                    print('Saved file')
            else:
                save_f(w, arr)
                print('Saved file')

def get_args():
    """
    Returns the parsed command line arguments

    :return: The command line arguments
    """

    # To make sure params are positive
    def check_positive(_type):
        def check_positive_(val):
            ret = _type(val)  # value to return
            if ret <= 0:
                raise argparse.ArgumentTypeError('Value must be positive')
            return ret

        return check_positive_

    parser = argparse.ArgumentParser(description='HANND Automated Sample Recorder, HBS Lab, UT Dallas',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-w', '--words', help='The words to speak', nargs='*', default=DM.DEFAULT_TARGET_LIST)
    parser.add_argument('-n', '--num-times', help='Number of times to record each word', type=check_positive(int),
                        default=10)
    parser.add_argument('-p', '--play_back', help='Play back clips upon recording and ask whether to save',
                        action='store_true')

    _args = parser.parse_args()

    return _args


if __name__ == '__main__':
    args = get_args()
    main(args)
