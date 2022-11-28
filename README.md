# 3D Unet Framework

This is a Tensorflow 2 based implementation of a generic 3D-Unet.

This repo allows the creation of 3D-Unet of any structure, letting you to choose the depth of the network, the number of consecutive convolutions, the number of filters on each level, the kernel size on each level, the pooling method (AvgPooling or MaxPooling), the inclusion of dropout after pooling and concatenation, inclusion of batch normalization after convolutions or not, the padding method, the upsampling method (UpSampling or ConvTranspose) and the activation function after convolutions.

To create this network:
![alt text] (https://github.com/luiserrador/ml/blob/master/images/unet.png?raw=true)
