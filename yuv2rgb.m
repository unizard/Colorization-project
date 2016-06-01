function rgb = yuv2rgb( y, u, v )
%YUV2RGB Summary of this function goes here
%   Detailed explanation goes here

[r,c] = size(y);
y = double(y);
u = double(u);
v = double(v);
rgb = zeros(r, c, 3);

% wwb=1.14;
% wwgu=-0.395; wwbu=-0.581;
% wwgv=2.03;
% 
% rr=y+wwb*v;
% gg=y+wwgu*u+wwbu*v;
% bb=y+wwgv*u;

rr = y + 1.14*v;
gg = y - 0.39*u - 0.58*v;
bb = y + 2.03*u;

rgb(:,:,1) = uint8(max(0, min(255,rr+ 0.5)));
rgb(:,:,2) = uint8(max(0, min(255,gg+ 0.5)));
rgb(:,:,3) = uint8(max(0, min(255,bb+ 0.5)));

rgb = uint8(rgb);

end

