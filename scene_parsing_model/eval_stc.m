caffe_root = '../../caffe-future';
addpath(genpath(caffe_root));

model = 'deploy.prototxt';
weights = 'semantic_segmentation.caffemodel';

caffe.set_mode_gpu();
net = caffe.Net(model, 'test');
net.copy_from(weights);

im_data = imread('test.jpg');
if size(im_data, 3) == 3
    im_data = rgb2gray(im_data);
end
im_data = double(repmat(im_data, [1,1,3]));

[width, height, ~] = size(im_data);
im_data = imresize(im_data, [width, height]);
mean_data = zeros(width, height, 3);
mean_data(:, :, 1) = 104.00698793 * ones(width, height);
mean_data(:, :, 2) = 116.66876762 * ones(width, height);
mean_data(:, :, 3) = 116.66876762 * ones(width, height);
im_data = im_data - mean_data;
im_data = permute(im_data, [2,1,3]);
im_data = single(im_data);

net.blobs('data').reshape([width height 3 1]);
net.reshape();

res = net.forward({im_data});
prob = res{1};
result = permute(prob, [2, 1, 3]);

save('stc', 'result');


