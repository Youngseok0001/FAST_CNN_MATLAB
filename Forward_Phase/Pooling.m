function [out] = Pooling( A, blocksize)

global Pooling;

if blocksize == 1
    out = A;    
else
    
%// Store blocksizes
nrows = blocksize;
ncols = blocksize;
ncols = gpuArray(ncols);

%// Store stepsizes along rows and cols
d_row = nrows;
d_col = ncols;
d_col = gpuArray(d_col);
%// Get sizes for later usages
[m,n,r,w] = size(A);

%// Start indices for each block
start_ind = reshape(bsxfun(@plus,[1:d_row:m-nrows+1]',[0:d_col:n-ncols]*m),[],1); %//'

%// Row indices
lin_row = permute(bsxfun(@plus,start_ind,[0:nrows-1])',[1 3 2]);  %//'

%// 2D linear indices
lidx_2D = reshape(bsxfun(@plus,lin_row,[0:ncols-1]*m),nrows*ncols,[]);
%// 3D linear indices
lidx_3D = pagefun(@plus,lidx_2D,m*n*permute((0:r-1),[1 3 2]));
%// 4D linear indices
lidx_4D = pagefun(@plus,lidx_3D,m*n*r*permute((0:w-1),[1 3 4 2]));

%// Get linear indices based on row and col indices and get desired output
im2col_out = A(lidx_4D);

temp = reshape(im2col_out,blocksize,blocksize,[],r,w);

if strcmp(Pooling,'MEAN')

temp = mean(mean(temp));

elseif strcmp(Pooling,'MAX')
    
temp = max(max(temp));

elseif strcmp(Pooling,'MIN')
    
temp = min(min(temp));

end

out = reshape(temp,m/2,n/2,r,w);


end


