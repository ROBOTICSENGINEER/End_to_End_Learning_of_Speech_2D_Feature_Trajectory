# End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands

### End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands


Speech is one of the most common forms of communication in humans. Speech commands are essential parts of multimodal controlling of prosthetic hands. In the past decades, researchers used automatic speech recognition systems for controlling prosthetic hands by using speech commands. Automatic speech recognition systems learn how to map human speech to text. Then, they used natural language processing or a look-up table to map the estimated text to a trajectory. However, the performance of conventional speech-controlled prosthetic hands is still unsatisfactory. Recent advancements in general-purpose graphics processing units (GPGPUs) enable intelligent devices to run deep neural networks in real-time. Thus, architectures of intelligent systems have rapidly transformed from the paradigm of composite subsystems optimization to the paradigm of end-to-end optimization. In this paper, we propose an end-to-end convolutional neural network (CNN) that maps speech 2D features directly to trajectories for prosthetic hands. The proposed convolutional neural network is lightweight, and thus it runs in real-time in an embedded GPGPU. The proposed method can use any type of speech 2D feature that has local correlations in each dimension such as spectrogram, MFCC, or PNCC. We omit the speech to text step in controlling the prosthetic hand in this paper. The network is written in Python with Keras library that has a TensorFlow backend. We optimized the CNN for NVIDIA Jetson TX2 developer kit. Our experiment on this CNN demonstrates a root-mean-square error of 0.119 and 20ms running time to produce trajectory outputs corresponding to the voice input data. To achieve a lower error in real-time, we can optimize a similar CNN for a more powerful embedded GPGPU such as NVIDIA AGX Xavier. 



# Cite as:


### IEEE

M. Jafarzadeh and Y. Tadesse, "End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands", in 2020 Second International Conference on Transdisciplinary AI (TransAI), 2020, pp. 25-33.



### ACM

Jafarzadeh, M. and Tadesse, Y., 2020. End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands. In 2020 Second International Conference on Transdisciplinary AI (TransAI). IEEE, pp. 25-33.



### BibTeX

@inbook{jafarzadeh_End_to_End_2020,
title={End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands},
booktitle={2020 Second International Conference on Transdisciplinary AI (TransAI)},
publisher={IEEE},
author={Jafarzadeh, Mohsen and Tadesse, Yonas},
year={2020},
pages={25-33}
}





# HANND Senior Design
Akshay Chitale, Rodrigo Avila, Clarissa Curry, and Cameron Ford

## Dataset
The GSCD is found at [here](https://storage.cloud.google.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz)

More information can be found at [here](https://arxiv.org/abs/1804.03209)

Data should be unzipped into a directory called `data_speech_commands`, with each word in its own directory

Custom speech commands should be placed in a directory called `custom_speech_commands`, with folder names of the word being said in the wav files in that folder. Using custom speech commands would require adding `custom_speech_commands` to `data_manager.DataManager.DEFAULT_DATA_PATHS`

## Description of Files
- `CNN.py` - The most important aspect of our software, is the Neural Network. This CNN as stated before is an end-to-end network, where the network is fed a spectrogram made from a wav file, and outputs an array of values that correspond to voltages. The network has two convolutional layers, which are the most adequate to extract useful speech features from the spectrogram. Maxpooling groups n items and outputs the maximum of those items. The last two layers, which are fully connected, narrow down the number of outputs we desired, in this case, the second fully connected layer had a dimension of five to correspond with the number of fingers we are actuating.  
- `data_manager.py` - This file manipulees the data, so that it can easily be fed into the network or a pre-trained model. It starts by setting our training set to our expected output values, these labels are not seen by network, since we are using regression and not classification. This part also sets our unknown set which is being recognized, but has no output mapped to any item in the set. By setting out sets, we generate our lists to be used based on the data given.  In this file, we are also able to receive a wav file and convert it into a spectrogram by performing a Fourier transform and plotting the signal in the frequency domain instead of the time domain. This allows us to extract better features from the signal, and makes recognition far easier and accurate. 
- `i2c.py` - This file controls I2C from the Jetson to the hand. The file has a set of values that correspond to the right DAC addresses. Because DAC reads instructions and bytes in a specific order, where commands go in firsts as hex values, followed by the DAC address, then ending with the 16-bit taken from network output. The network output is scaled to 16-bit, so an output of ‘1’ in the network would correspond to 0xFFFF for the DAC 
- `main.py` - This file is responsible for running all the software at once, from here the user is able to interact with all aspects, from the network, microphone and I2C. From here the user can train the neural network, load a pre-trained model, enable/disable I2C, and even shorten the size of the data set if needed for training purposes. 
- `microphone.py` - The microphone file has controls for microphone input in the overall program. It also allows for creating of a custom dataset by saving files into a custom directory for future use.  


# License 

Copyright (c) 2020, Mohsen Jafarzadeh

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
3. All advertising materials mentioning features or use of this software must display the following acknowledgement: This product includes software developed by the <organization>.
4. Neither the name of the <organization> nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY MOHSEN JAFARZADEH ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


