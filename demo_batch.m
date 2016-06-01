%% set parameters && load model
disp('Load model (colorization model && scene parsing model) ... ')
addpath(genpath('.'));
caffe_root = '../caffe-future';
addpath(genpath(caffe_root));
% load colorization model
if ~exist('net', 'var')
    load('colorization_model/NETWORK_TIP_V1.mat')
end
% load scene parsing model
if ~exist('stc_net', 'var')
    model = 'scene_parsing_model/deploy.prototxt';
    weights = 'scene_parsing_model/semantic_segmentation.caffemodel';
    caffe.set_mode_gpu();
    stc_net = caffe.Net(model, 'test');
    stc_net.copy_from(weights);
end
ngr = 7; % patch size
edge = (ngr-1)/2;
stcNum = 33;
daisyNum = 32;

%% colorization
test_root = 'testset';
images = dir(fullfile(test_root, '*.jpg'));
names = {images.name};

for im_id = 1 : numel(names)
    imgname = char(fullfile(test_root, names(im_id)));
    fprintf('\n'); disp(imgname);
    %% read a target image
    test_image = imread(imgname);
    if size(test_image, 3) == 3
        test_gray = rgb2gray(test_image);
    else
        test_gray = test_image;
    end
    
    %% generate semantic feature
    disp('Generate semantic feature ... ')
    %  scene parsing
    im_data = double(repmat(test_gray, [1,1,3]));
    [width, height, ~] = size(test_gray);
    im_data = imresize(im_data, [width, height]);
    mean_data = zeros(width, height, 3);
    mean_data(:, :, 1) = 104.00698793 * ones(width, height);
    mean_data(:, :, 2) = 116.66876762 * ones(width, height);
    mean_data(:, :, 3) = 116.66876762 * ones(width, height);
    im_data = im_data - mean_data;
    im_data = permute(im_data, [2,1,3]);
    im_data = single(im_data);
    stc_net.blobs('data').reshape([height width 3 1]);
    stc_net.reshape();
    res = stc_net.forward({im_data});
    prob = res{1};
    test_stclabel.result = permute(prob, [3, 2, 1]); % stc_channels first
    
    %% generate daisy feature
    disp('Generate Daisy feature ... ')
    dzy = compute_daisy(test_gray,7,1,3,8);
    %% compute gist and find nearest cluster center
    disp('Find nearest image cluster ... ')
    % set params of gist
    clear param
    param.orientationsPerScale = [8 8 8 8]; % number of orientations per scale (from HF to LF)
    param.numberBlocks = 4;
    param.fc_prefilt = 4;
    [gist, param] = LMgist(test_gray, '', param); % 512 dim, gist is computed from gray image
    
    % stc histogram
    nbins = 33;
    [~, stc_label_tmp] = max(test_stclabel.result, [], 1);
    stc_label_tmp = stc_label_tmp - 1;
    stc_hist_test = hist(reshape(stc_label_tmp, [numel(stc_label_tmp), 1]), [1:nbins]);
    stc_hist_test = stc_hist_test ./ sum(stc_hist_test); 
    
    % gist distance and stc histogram similarity
    top_k = 3;
    dist_ind = 0;
    for layer_id = 1 : numel(net) % layer
        for kc_id = 1 : numel(net(layer_id).net_data) % cluster
            kcenter = net(layer_id).net_data(kc_id).kcenter;
            gist_eudist = sqrt((gist - kcenter) * (gist - kcenter)');
            stchist_ref = net(layer_id).net_data(kc_id).stchist;
            stchist_cos = sum(stc_hist_test .* stchist_ref) / (sqrt(stc_hist_test * stc_hist_test') * sqrt(stchist_ref * stchist_ref'));
            dist_ind = dist_ind + 1;
            dist_record(dist_ind).gist_dist = gist_eudist;
            dist_record(dist_ind).stc_dist = stchist_cos;
            dist_record(dist_ind).layer = layer_id;
            dist_record(dist_ind).kc = kc_id;
            dist_record(dist_ind).stchist_ref = stchist_ref;
        end
    end
    [~, gist_order] = sort([dist_record.gist_dist]);
    topk_ind1 = gist_order(1:top_k);
    [~, stc_order] = sort([dist_record(topk_ind1).stc_dist], 'descend');
    final_order = topk_ind1(stc_order);
    min_layer_id = dist_record(final_order(1)).layer;
    min_kc_id = dist_record(final_order(1)).kc;
    subnet = net(min_layer_id).net_data(min_kc_id).nn;
    
   %% colorize 
    fprintf('Using NN of ----- Layer #%d -- Cluster #%d \n', min_layer_id, min_kc_id);
    fprintf('Start colorization (size: %d-by-%d )  ...  ', width, height);
    
	if ~exist('colorization_results', 'dir')
		mkdir('colorization_results');
		mkdir('colorization_results/before_refined')
		mkdir('colorization_results/after_refined')
    end
    
    png_name = char(names(im_id));
    png_name = strcat(png_name(1:end-4),'.png');
    result_name =  char(fullfile('colorization_results/before_refined', png_name));
    result_refined_name = char(fullfile('colorization_results/after_refined', png_name));
    
    execute_color;
    fprintf(' completely\n');
    
    toc;
    
end