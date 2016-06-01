function [y,u,v] = rgb2yuv( rgb )
%RGB2YUV Summary of this function goes here
%   Detailed explanation goes here

wry = 0.299; wgy = 0.587; wby = 0.114;
wru=-0.147; wgu=-0.28886; wbu=0.436;
wrv=0.615; wgv=-0.515; wbv=-0.1;

rgb = double(rgb);

r = rgb(:,:,1);
g = rgb(:,:,2);
b = rgb(:,:,3);

% y = wry*r + wgy*g + wby*b;
% u = wru*r + wgu*g + wbu*b;
% v = wrv*r + wgv*g + wbv*b;


y = 0.299*r + 0.587*g + 0.114*b;
u = -0.147*r - 0.289*g + 0.436*b;
v = 0.615*r - 0.515*g - 0.100*b;

end

