function ssn=qx_psnr(a,b)
[h,w,d]=size(a);
ssn=0;
for y=1:h
    for x=1:w
        for c=1:d
            ab=(double(a(y,x,c))-double(b(y,x,c)))/255;
            ssn=ssn+ab*ab;
        end
    end
end
if(ssn<0.000001)
    ssn=48000.0;
else
    ssn=log(double(h*w*d)/ssn)*10.0/log(10.0);
end