caffe_root = '../../caffe-future/' 
import sys
import os
import scipy.io as sio
sys.path.insert(0, caffe_root + 'python')

import numpy as np
from PIL import Image

import caffe


def rgb2gray(rgb):
	return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])	
gray_flag = True

testdata_path = 'testset'
testresult_path = 'test_result'

if os.path.exists(testresult_path) == False:
	os.makedirs(testresult_path)

# load net
net = caffe.Net('deploy.prototxt', semantic_segmentation.caffemodel', caffe.TEST)
# load image, switch to BGR, subtract mean, and make dims C x H x W for Caffe
filelist = os.listdir(testdata_path)
count = 0
for filename in filelist:
	count = count + 1;
	print('#{0} : {1}'.format(count, filename))
	impath = testdata_path + '/' + filename
	im = Image.open(impath)
	in_ = np.array(im, dtype=np.float32)
	if gray_flag == True:
		in_ = rgb2gray(in_)
		(row, col) = in_.shape
		im3 = np.zeros([row, col, 3], dtype = np.float32)
		for i in range(3):
			im3[:,:,i] = in_
		in_ = im3
	in_ = in_[:,:,::-1]
	in_ -= np.array((104.00698793,116.66876762,122.67891434))
	in_ = in_.transpose((2,0,1))
	# shape for input (data blob is N x C x H x W), set data
	net.blobs['data'].reshape(1, *in_.shape)
	net.blobs['data'].data[...] = in_
	# run net and take argmax for prediction
	net.forward()
	# out = net.blobs['score'].data[0].argmax(axis=0)
	out = net.blobs['score'].data[0]
	resultname = filename[:-4] + '.mat'
	resultpath = testresult_path + '/' + resultname 
	sio.savemat(resultpath, {'result':out})






