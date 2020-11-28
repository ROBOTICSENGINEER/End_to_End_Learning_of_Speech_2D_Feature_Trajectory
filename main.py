""" main.py
This module is the driver for the whole program

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
import logging
import sys
import time

import data_manager as dm
import CNN
import i2c
import microphone


def main(_args):
    """
    The main function that runs the whole program

    :param _args: The parsed command line arguments
    :return: None
    """
    print('Setting up...')
    logging.debug('Building Data Manager')
    data_manager = dm.DataManager(set_size_multiplier=args.set_size)
    targets = dm.DataManager.DEFAULT_TARGET_LIST
    target_map = dm.DataManager.DEFAULT_MAP

    logging.debug('Building CNN')
    network = CNN.SpectrogramToVoltageCNN(target_map, dataset=data_manager, save_prefix='results/' + _args.name)
    if _args.load:
        logging.debug('Loading Model')
        network.load_model()
    else:
        logging.debug('Building Model')
        network.train(batch_size=int(args.set_size*512))
        logging.debug('Saving Model')
        network.save_all()

    if not _args.no_test:
        logging.debug('Testing Model')
        results = network.test(batch_size=int(args.set_size*512))

        logging.debug('TEST LOSS: %s' % results[0])
        logging.debug('TEST ACC: %s' % results[1])
        logging.debug('TEST MSE: %s' % results[2])

    # Ready for live audio, so set up microphone and I2C
    mic = microphone.Microphone()
    if not _args.i2c_suppress:
        dac = i2c.I2C()

    # Exit with keyboard interrupt, record on <Enter>
    while True:
        sys.stdout.flush()
        print('Press enter to speak or Ctrl-C to exit...')
        input()
        print('Recording...')

        wav = mic.get()

        print('Processing...')

        # Record time for full pipeline
        t = time.time()

        # Generate spectrogram
        _f, _t, val = dm.spectrogram_plot_from_wav(wav)

        # Process input
        result = network.predict(val.reshape(-1, 129, 71, 1))

        # Processing done
        t = time.time() - t

        print('Result %s in %.3f ms.' % (str(result), t * 1000))

        # Plot result
        dm.plot_spectrogram(_f, _t, val, result)

        # Send on I2C
        if not _args.i2c_suppress:
            print('Sending result on I2C...')
            dac.send_logical(i2c.DAC_A, result[0])  # thumb
            #dac.send_logical(i2c.DAC_E, result[1])  # index finger
            dac.send_logical(i2c.DAC_B, result[2])  # middle finger
            dac.send_logical(i2c.DAC_C, result[3])  # ring finger
            dac.send_logical(i2c.DAC_D, result[4])  # pinky finger

        print("Done.")


def get_args():
    """
    Returns the parsed command line arguments

    :return: The command line arguments
    """

    # To make sure params are 0 to 1
    def check_0_1(_type):
        def check_0_1_(val):
            ret = _type(val)  # value to return
            if ret < 0 or ret > 1:
                raise argparse.ArgumentTypeError('Value must be 0 to 1')
            return ret

        return check_0_1_

    parser = argparse.ArgumentParser(description='HANND Neural Network, HBS Lab, UT Dallas',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('name', help='Name of file to save/load')
    parser.add_argument('-l', '--load', help='Loads data instead of training', action='store_true')
    parser.add_argument('-n', '--no-test', help='Do not run testing', action='store_true')
    parser.add_argument('-i', '--i2c-suppress', help='Suppress I2C output', action='store_true')
    parser.add_argument('-s', '--set-size', help='Multiplier to reduce the set size', type=check_0_1(float),
                        default=1.0)

    _args = parser.parse_args()

    return _args

if __name__ == '__main__':
    args = get_args()
    CNN.global_config()
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)  # Print statements only for debug
    try:
        print('==========     HANND End-to-End Speech Control Neural Network Started      ==========')
        main(args)
    except KeyboardInterrupt:
        print('Goodbye.')
        print('==========     HANND End-to-End Speech Control Neural Network Finished     ==========')
