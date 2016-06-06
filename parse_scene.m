function [result] = parse_scene(test_gray)
% load model
if ~exist('stc_net', 'var')
    model = 'scene_parsing_model/deploy.prototxt';
    weights = 'scene_parsing_model/semantic_segmentation.caffemodel';
    %caffe.set_mode_gpu();
    caffe.set_mode_cpu();
    stc_net = caffe.Net(model, 'test');
    stc_net.copy_from(weights);
end

%  scene parsing
im_data = double(repmat(test_gray, [1,1,3]));
% width = 256;
% height = 256; 
[width, height, ~] = size(test_gray);
im_data = imresize(im_data, [width, height]);
mean_data = zeros(width, height, 3);
mean_data(:, :, 1) = 104.00698793 * ones(width, height);
mean_data(:, :, 2) = 116.66876762 * ones(width, height);
mean_data(:, :, 3) = 116.66876762 * ones(width, height);
im_data = im_data - mean_data;
im_data = permute(im_data, [2,1,3]); % Transpose
im_data = single(im_data);
stc_net.blobs('data').reshape([height width 3 1]);
stc_net.reshape();
res = stc_net.forward({im_data});
prob = res{1};
result = permute(prob, [3, 2, 1]);
end

