% tranform color images into gray scale image in batch.

fileFolder = fullfile('E:\Experiments\Materials\Final\test_img');
in = dir(fullfile(fileFolder,'*.jpg'));
imgnames = {in.name};
[r,c] = size(imgnames);
for i=1:c
    names(i) = strcat('E:\Experiments\Materials\Final\test_img\',imgnames(i));
    rgb = imread(char(names(i)));
    gray = rgb2gray(rgb);
    savePath = strcat('E:\Experiments\Materials\Final\test_gray\',imgnames(i));
    imwrite(gray,char(savePath));
end

