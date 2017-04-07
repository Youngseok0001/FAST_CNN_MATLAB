function [convolvedFeatures_relu, convolvedFeatures_pooled] = Forward_conv(conv, images, stack)

% Parameters:
%  conv:
%  conv.filterDim  -    Filter size for conv layer
%  conv.filterDepth -   Depth of filter
%  conv.numFilters -    Number of filters for conv layer
%  conv.poolDim    -    Pooling dimension, (should divide imageDim-filterDim+1)
%  conv.strides    -    Strides for convolution
%  conv.padding    -    Calibrated to preserve size
%  conv.convDim    -    Dimension after convolution
%  conv.outputDim  -    Dimension after pooling

%  images - large images to convolve with, matrix in the form
%           images(row, col, depth, image number)

%  stack - stack.W is of shape (filterDim, filterDim, fiterDepth, numFilters)
%          stack.b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form of
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
global Convolution_type; 
global Activation_function;


if strcmp(Convolution_type,'IM2COL')

convolvedFeatures = im2col_conv(images, stack.W, conv);

elseif strcmp(Convolution_type,'WINOGRAD')
    
convolvedFeatures = wino_conv(images, stack.W, conv);

end

b = permute(stack.b,[2 4 1 3]);
convolvedFeatures = bsxfun(@plus, convolvedFeatures,b);

if strcmp(Activation_function,'RELU')

convolvedFeatures_relu = relu(convolvedFeatures);

elseif strcmp(Activation_function,'SIGMOID')
    
convolvedFeatures_relu = sigmoid(convolvedFeatures);

end

convolvedFeatures_pooled = Pooling(convolvedFeatures_relu, conv.poolDim);

end


