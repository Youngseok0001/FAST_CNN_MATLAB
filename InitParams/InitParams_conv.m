function [ w, b, output_size ] = InitParams_conv(input_size, spec)

%Parameters:
%spec.filterDim  -    Filter size for conv layer
%spec.numFilters -    Number of filters for conv layer
%spec.poolDim    -    Pooling dimension, (should divide imageDim-filterDim+1)
%spec.strides    -    Strides for convolution
%spec.padding    -    Calibrated to preserve size
%spec.filterDepth -   Depth of filter
% 
%
% Returns:
% w - weights of filters
% b - bias for each filters
% output_size - The size of activation functions which will be passed onto the next layer 

%% %% Initialize parameters randomly ba sed on layer sizes.


 w = sqrt(2/(spec.filterDim^2* spec.filterDepth)) * randn(spec.filterDim, spec.filterDim,...
                  spec.filterDepth, spec.numFilters); 
              
 b = zeros(spec.numFilters, 1);

output_dim  = (input_size(1) - spec.filterDim +2*(spec.padding))/ spec.strides +1;
output_dim  = output_dim/ spec.poolDim;
output_size = [output_dim, output_dim, spec.numFilters];










 
 
 
 




end

