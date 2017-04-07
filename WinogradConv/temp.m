clear;
clc;

batch_size = single([ 64 32 16 8 4 2 1]);


    
a = single(gpuArray(rand(32,32,64,256)));
b = single(gpuArray(rand(3,3,64,128)));
conv.padding = 1;

tic;
out1 =  wino_conv(a,b,conv);
toc;

tic;
out2 = im2col_conv(a,b,conv);
toc;

%{
error = abs(out1 - out2);
error = sum(error(:))
%}