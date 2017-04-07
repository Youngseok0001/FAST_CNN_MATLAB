function [ conv, full ] = Cnn_spec_Cfar10_3( )

global imageDim;
global numClasses;
global input_size;


i = 1;
conv{i}.filterDim = 3;           % Filter size for conv layer
conv{i}.filterDepth =input_size(3);          % Depth of a filter
conv{i}.numFilters = 32;         % Number of filters for conv layer
conv{i}.poolDim = 1;             % Pooling dimension, (should divide imageDim-filterDim+1)
conv{i}.strides = 1;             % strides for convolution
conv{i}.padding = (conv{i}.filterDim - 1)/2; % calibrated to preserve size
conv{i}.convDim = (imageDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; % dimension after convolution
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim; % dimension after pooling

i = 2;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 32;   
conv{i}.poolDim = 2;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;

i = 3;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 64;   
conv{i}.poolDim = 1;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;

i = 4;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 64;   
conv{i}.poolDim = 2;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;

i = 5;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 128;   
conv{i}.poolDim = 2;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;

i = 6;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 128;   
conv{i}.poolDim = 2;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;

i = 7;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 128;   
conv{i}.poolDim = 1;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;

i = 8;
conv{i}.filterDim = 3;    
conv{i}.filterDepth = conv{i-1}.numFilters;
conv{i}.numFilters = 128;   
conv{i}.poolDim = 2;       
conv{i}.strides = 1;       
conv{i}.padding = (conv{i}.filterDim - 1)/2; 
conv{i}.convDim = (conv{i-1}.outputDim - conv{i}.filterDim + 2*conv{i}.padding)/conv{i}.strides +1; 
conv{i}.outputDim = conv{i}.convDim / conv{i}.poolDim;


full{9}.outputSize = numClasses;



end

