
function [out] = col2im_4D_sliding_v1(patches)

%// Get sizes for later usages
[x,y,block_n,channel_n,image_n] = size(patches);

%// Get the row index
row_idx_temp = reshape(repmat([1:x]',[sqrt(block_n) 1]),[],1);
scaliing_vector_temp = [0 : sqrt(block_n)-1] .* (x * y);
scaling_vector = reshape(repmat(scaliing_vector_temp,[x 1]),[],1);
row_idx = row_idx_temp + scaling_vector;

%// Get the col index
col_idx_temp = reshape(repmat([1: x: x*y-(x-1)],[1 sqrt(block_n)]),1,[]);
scaliing_vector_temp = [0 : sqrt(block_n)-1] .* (x*y*sqrt(block_n));
scaling_vector = reshape(repmat(scaliing_vector_temp,[x 1]),1,[]);
col_idx = col_idx_temp + scaling_vector -1;

%// Get the 2D index
lidx_2D = bsxfun(@plus,row_idx,col_idx);
lidx_2D = gpuArray(lidx_2D);
%// Get the 3D index
lidx_3D = pagefun(@plus,lidx_2D,x*y*block_n*permute(0:channel_n-1,[1 3 2]));

%// Get the 4D index
lidx_4D = pagefun(@plus,lidx_3D,x*y*block_n* channel_n*permute((0:image_n-1),[1 3 4 2]));

out = patches(lidx_4D);


return;





