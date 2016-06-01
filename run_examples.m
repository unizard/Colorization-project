clear;
addpath(genpath('Daisy'))
disp('Load colorization model ... ')
load('colorization_model/NETWORK_TIP_V1.mat') 
ngr = 7; % neighbor
edge = (ngr-1)/2;
% parameters of NN
stcNum = 33;
daisyNum = 32;
% read gray-scale image
fileFolder_gray = 'examples/image';
in_gray = dir(fullfile(fileFolder_gray,'*.jpg'));
imgnames_gray = {in_gray.name};
imgnum_gray = numel(imgnames_gray);
% read semantic labels data
fileFolder_stc = 'examples/stc';
in_stc = dir(fullfile(fileFolder_stc,'*.mat'));
matnames_stc = {in_stc.name};
% read Daisy feature
fileFolder_daisy = 'examples/daisy';
in_daisy = dir(fullfile(fileFolder_daisy,'*.mat'));
matnames_daisy = {in_daisy.name};

imgnum = imgnum_gray;
for ii = 1 : imgnum
    names_gray = fullfile(fileFolder_gray, imgnames_gray(ii));
    names_stc = fullfile(fileFolder_stc, matnames_stc(ii));
    names_daisy = fullfile(fileFolder_daisy, matnames_daisy(ii));
    
    disp(char(names_gray))
    test_image = imread(char(names_gray));
    if size(test_image, 3) == 3
        test_gray = rgb2gray(test_image);
    else
        test_gray = test_image;
    end
    
    test_stclabel = load(char(names_stc));
    
    load(char(names_daisy));
    
    png_name = char(imgnames_gray(ii));
    png_name = strcat(png_name(1:end-4),'.png');
    
	if ~exist('colorization_results', 'dir')
		mkdir('colorization_results');
		mkdir('colorization_results/before_refined')
		mkdir('colorization_results/after_refined')
	end
    
    result_name =  char(fullfile('colorization_results/before_refined', png_name));
    result_refined_name = char(fullfile('colorization_results/after_refined', png_name));

    [row,column] = size(test_gray);
    
    tic;
    % compute gist and find nearest cluster center
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
    
    fprintf('------ Layer #%d -- Cluster #%d \n', min_layer_id, min_kc_id);

    fprintf('for %d image (%d * %d)... \n', ii, row, column);
    execute_color;
    toc;
    fprintf(' completely\n');

end
