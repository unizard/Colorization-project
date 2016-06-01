========================================================================
INTRODUCTION

Fully Automatic Image Colorization Code
Created by Zezhou Cheng (chengzezhou@sjtu.edu.cn) 

This MATLAB code implements the colorization algorithm presented in the following paper:

Zezhou Cheng, Qingxiong Yang, and Bin Sheng.
Image Colorization Using Neural Networks.
Submitted to IEEE Transactions on Image Processing 2016.

The preliminary version of this work is described in:

Zezhou Cheng, Qingxiong Yang, and Bin Sheng. 
Deep Colorization.
Proceedings of the IEEE International Conference on Computer Vision. 2015: 415-423.

========================================================================
INSTALLATION

This code was tested on MATLAB 2015b under UBUNTU 14.04.4 LTS

If you want to colorize your own images, you need to install the following package:

Caffe :  https://github.com/longjon/caffe/tree/future   OR   https://github.com/BVLC/caffe/
Please refer to http://caffe.berkeleyvision.org/installation.html


========================================================================
USAGE

(1)  For the experiment on your own test images :

	Single image colorization(default name of image 'test.jpg') :

	--- run demo_single.m 

        Batch colorization (default path of images 'testset'):
    
        --- run demo_batch.m

(2)  For a quick experiment on the bundled examples without the requirement of installing CAFFE:

	--- run run_examples.m

========================================================================
LIBRARY

The following libraries are included in this code :

(1) DAISY Descriptor
http://cvlab.epfl.ch/software/daisy

(2) GIST Descriptor
http://people.csail.mit.edu/torralba/code/spatialenvelope/

(3) Fully Convolutional Networks For semantic Segmantation 
https://github.com/shelhamer/fcn.berkeleyvision.org
Note that we retrained this model on grayscale version of images from SIFT Flow Database. 
Our trained model is placed in 'scene_parsing_model' folder, named as 'semantic_segmentation.caffemodel'

(4) Domain Transform RF filter
http://inf.ufrgs.br/~eslgastal/DomainTransform/

