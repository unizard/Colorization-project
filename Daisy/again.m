im = imread('21.jpg');
dzy = compute_daisy(im);
 save test_daisy.mat dzy -v7.3
b = rgb2gray(im);
imwrite(b,'test_gray.jpg');