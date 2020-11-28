""" i2c.py
This module contains the i2c communication code for the LTC2605 DAC

The datasheet can for the DAC can be found at:
    https://www.analog.com/media/en/technical-documentation/data-sheets/2605fa.pdf


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

import i2cdev

# Definitions from the LTC2605 DAC datasheet

# Page 12 of datasheet, DAC I2C Address on bus
# NOTE: Many more configurations of address exist. Add them here if they are relevant
ADDR_DAC_ALL_FLOAT = 0x41
ADDR_GLOBAL = 0x73

# Page 11 of datasheet, commands
CMD_WRITE = 0x0
CMD_UPDATE = 0x1
CMD_WRITE_UPDATE_ALL = 0x2
CMD_WRITE_UPDATE = 0x3
CMD_POWER_DOWN = 0x4
CMD_NO_OP = 0xF

# Page 11 of datasheet, addresses
DAC_A = 0x0
DAC_B = 0x1
DAC_C = 0x2
DAC_D = 0x3
DAC_E = 0x4
DAC_F = 0x5
DAC_G = 0x6
DAC_H = 0x7
DAC_ALL = 0xF


class I2C(object):
    """
    Wrapper for i2cdev.I2C that has HANND specific sending and addresses. Note that python has to be run as sudo for
    I2C to work
    """

    def __init__(self, bus=1, dac_addr=ADDR_DAC_ALL_FLOAT):
        """
        Sets up I2C to the DAC

        :param bus: The I2C bus to use on the Jetson. Should be 1 unless there is a good reason otherwise. For I2C bus
            1, SDA is the immediately to the right of the labeled J21, and SCL is immediately to the right of SDA. Any
            ground pin on the Jetson should work
        :param dac_addr: The address of the DAC. If all address control pins are left floating on the DAC, this should
            stay at ADDR_DAC_ALL_FLOAT
        """
        self._dev = i2cdev.I2C(dac_addr, bus)

    def send(self, addr, value, cmd=CMD_WRITE_UPDATE):
        """
        Sends a 16-bit value to the given address on the DAC

        :param addr: The DAC output pin to send the value to. These are defined in this file as DAC_X (eg. DAC_A,
            DAC_B, ...)
        :param value: The 16-bit value to send to the DAC. 0x0000 is the minimum value and 0xFFFF is the maximum value
        :param cmd: The command to give the DAC. Should be CMD_WRITE_UPDATE unless there is a good reason otherwise
        :return: None
        """
        first = int(((cmd & 0xFF) << 4) | (addr & 0xFF))
        second = int((value >> 8) & 0xFF)
        third = int(value & 0xFF)
        self._dev.write(bytes([first, second, third]))

    def send_logical(self, addr, value, cmd=CMD_WRITE_UPDATE):
        """
        Sends a logical value from 0-1 to the given address on the DAC. Values less than 0 are clipped to 0 and values
        greater than 1 are clipped to 1.

        :param addr: The DAC output pin to send the value to. These are defined in this file as DAC_X (eg. DAC_A,
            DAC_B, ...)
        :param value: The logical floating point value to send to the DAC. 0 maps to 0x0000 and 1 maps to 0xFFFF. All
            values outside the allowed range are clipped so that they can be safely sent
        :param cmd: The command to give the DAC. Should be CMD_WRITE_UPDATE unless there is a good reason otherwise
        :return: None
        """
        value = max(0, value)
        value = min(1, value)
        to_send = int(value * 0xFFFF)
        self.send(addr, to_send, cmd)

    def __del__(self):
        """
        Closes I2C file upon destruction of object

        :return: None
        """
        self._dev.close()


if __name__ == '__main__':
    # Small I2C demo for testing if this file is run as main
    import time

    dac = I2C()

    # Send every few data points
    for i in range(0xFF):
        dac.send(DAC_A, (i << 8) | i)
        time.sleep(0.1)
