# End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands

End-to-End Learning of Speech 2D Feature-Trajectory for Prosthetic Hands


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


Speech is one of the most common forms of communication in humans. Speech commands are essential parts of multimodal controlling of prosthetic hands. In the past decades, researchers used automatic speech recognition systems for controlling prosthetic hands by using speech commands. Automatic speech recognition systems learn how to map human speech to text. Then, they used natural language processing or a look-up table to map the estimated text to a trajectory. However, the performance of conventional speech-controlled prosthetic hands is still unsatisfactory. Recent advancements in general-purpose graphics processing units (GPGPUs) enable intelligent devices to run deep neural networks in real-time. Thus, architectures of intelligent systems have rapidly transformed from the paradigm of composite subsystems optimization to the paradigm of end-to-end optimization. In this paper, we propose an end-to-end convolutional neural network (CNN) that maps speech 2D features directly to trajectories for prosthetic hands. The proposed convolutional neural network is lightweight, and thus it runs in real-time in an embedded GPGPU. The proposed method can use any type of speech 2D feature that has local correlations in each dimension such as spectrogram, MFCC, or PNCC. We omit the speech to text step in controlling the prosthetic hand in this paper. The network is written in Python with Keras library that has a TensorFlow backend. We optimized the CNN for NVIDIA Jetson TX2 developer kit. Our experiment on this CNN demonstrates a root-mean-square error of 0.119 and 20ms running time to produce trajectory outputs corresponding to the voice input data. To achieve a lower error in real-time, we can optimize a similar CNN for a more powerful embedded GPGPU such as NVIDIA AGX Xavier. 

