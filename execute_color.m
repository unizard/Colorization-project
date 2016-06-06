%% test:

% Make input feature matrix.
test_stc = double(test_stclabel.result);
[r,c] = size(test_gray);
test_gray = uint8(test_gray);
istep = edge+1:r-edge;
jstep = edge+1:c-edge;
[~, ilen] = size(istep);
[~, jlen] = size(jstep);
test_input = zeros(ngr*ngr+daisyNum+stcNum,ilen*jlen);
test_index = 1;

stc_label = zeros(1, ilen*jlen); 

for i=1:ilen
    for j=1:jlen
        is = istep(i);
        js = jstep(j);
        % Patch feature.
        test_tmp1 = test_gray(is-edge:is+edge,js-edge:js+edge);
        test_gy = reshape(test_tmp1,ngr*ngr,1);
        test_gy = double(test_gy)/255;
        
        %test_pixel = test_gray(is,js); 
        
        % Daisy feature
        test_tmp1 = display_descriptor(dzy,is-1,js-1);
        test_daisy = reshape(test_tmp1,daisyNum,1);
        
        % Semantic feature
        test_s = test_stc(:, is,js);

        % Update
        test_input(:,test_index) = [test_gy; test_daisy; test_s];        
        test_index = test_index + 1;
    end
end


% Compute result. test_input: m x n, test_output: 2 x n.
test_output = sim(subnet, test_input); 
test_index = 1;
test_final = zeros(r,c,2);
for i=1:ilen
    for j=1:jlen
        is = istep(i);
        js = jstep(j);
        test_rgb = test_output(:,test_index);
        test_final(is,js,:) = test_rgb;
        test_index = test_index + 1;
    end
end

%% process edge
i_ed = [1:edge, r-edge+1:r];
j_ed = [1:edge, c-edge+1:c];
[~,len_ed] = size(i_ed);
test_index_ed = 1;
test_input_ed = zeros(ngr*ngr+daisyNum+stcNum,len_ed*c+len_ed*(r-2*edge));
% Left edge.
for m=1:len_ed
    for ns = 1:c
        ms = i_ed(m);
        test_tmp1 = test_gray(ms,ns)*uint8(ones(ngr,ngr));
        for a = -edge:edge
            for b = -edge:edge
                if(ms+a>0 && ms+a<=r && ns+b>0 && ns+b<=c)
                    test_tmp1(edge+1+a, edge+1+b) = test_gray(ms+a,ns+b);
                end
            end
        end
        % Patch Feature.
        test_gy_ed = reshape(test_tmp1,ngr*ngr,1);
        test_gy_ed = double(test_gy_ed)/255;
        
        % DAISY Feature.
        test_tmp1 =  display_descriptor(dzy,ms-1,ns-1);
        test_daisy_ed = reshape(test_tmp1,daisyNum,1);
        
        %test_pixel_ed = test_gray(ms,ns);
        %test_pixel_ed = double(test_pixel_ed)/255;
        
        % Semantic Feature.
        test_stc_ed = test_stc(:, ms, ns);
        
        % Update
        test_input_ed(:,test_index_ed) = [test_gy_ed;  test_daisy_ed; test_stc_ed];
        test_index_ed = test_index_ed+1;
    end
end
% Right edge.
for ms=edge+1:r-edge
    for n=1:len_ed
        ns = j_ed(n);
        test_tmp1 = test_gray(ms,ns)*uint8(ones(ngr,ngr));
        for a=-edge:edge
            for b=-edge:edge
                if(ms+a>0 && ms+a<=r && ns+b>0 && ns+b<=c)
                    test_tmp1(edge+1+a, edge+1+b) = test_gray(ms+a,ns+b);
                end
            end
        end
        % Patch Feature.
        test_gy_ed = reshape(test_tmp1,ngr*ngr,1);
        test_gy_ed = double(test_gy_ed)/255;
        
        % DAISY Feature.
        test_tmp1 =  display_descriptor(dzy,ms-1,ns-1);
        test_daisy_ed = reshape(test_tmp1,daisyNum,1);
        
        % Semantic Feature.
        test_stc_ed = test_stc(:, ms, ns);
        
        % Update.
        test_input_ed(:,test_index_ed) = [test_gy_ed;  test_daisy_ed; test_stc_ed];
        test_index_ed = test_index_ed+1;
    end
end

% Compute result for edges. test_input_ed: m x k, test_output_ed: 2 x k.
test_output_ed  = sim(subnet, test_input_ed);
test_index_ed = 1;
% Right edge result.
for m=1:len_ed
    for ns = 1:c
        ms = i_ed(m);
        test_final(ms,ns,:) = test_output_ed(:, test_index_ed);
        test_index_ed = test_index_ed+1;
    end
end
% Left edge result.
for ms=edge+1:r-edge
    for n = 1:len_ed
        ns = j_ed(n);
        test_final(ms,ns,:) = test_output_ed(:, test_index_ed);
        test_index_ed = test_index_ed+1;
    end
end

y = double(test_gray);
u = test_final(:,:,1) * 255;
v = test_final(:,:,2) * 255;
test_final_img = yuv2rgb(y,u,v);
% save colorization result before refinement
imwrite(test_final_img,result_name); 

%% refine using edge-preserving filter
% the total times of executing refinement 
refine_time = 5; 
I =  y / 255;

% r = 10; 
% eps = 0.1^2; % try eps=0.1^2, 0.2^2, 0.4^2
% 
% u = guidedfilter(I, u, r, eps);
% v = guidedfilter(I, v, r, eps);

sigma_s = 10;
sigma_r = 0.4;
for i = 1:refine_time
	u= RF(u, sigma_s, sigma_r, 3, I);
	v= RF(v, sigma_s, sigma_r, 3, I);
end
test_final_img_refined= yuv2rgb(y,u,v);
% save colorization result after refinement
imwrite(test_final_img_refined,result_refined_name);