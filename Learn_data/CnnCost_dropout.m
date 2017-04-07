function [cost, grad, preds] = CnnCost_dropout(theta,images,labels,numClasses,...
                                conv,full,pred)
% Calcualte cost and gradient for a single layer convolutional
% neural network followed by a softmax layer with cross entropy
% objective.
%                            
% Parameters:
%  theta      -  unrolled parameter vector
%  images     -  stores images in imageDim x imageDim x imageDepth x numImges
%  labels     -  labels of imageset
%  numClasses -  number of classes to predict
%  conv:
%  conv.filterDim  -    Filter size for conv layer
%  conv.numFilters -    Number of filters for conv layer
%  conv.poolDim    -    Pooling dimension, (should divide imageDim-filterDim+1)
%  conv.strides    -    Strides for convolution
%  conv.padding    -  Calibrated to preserve size
%  conv.convDim    -    Dimension after convolution
%  conv.outputDim  -    Dimension after pooling

%  full:
%  full.outputSize   -  number of activations after passing through the layer
%  pred - if true, skips computing gradient. only return cost.

%
%
% Returns:
%  cost       -  cross entropy cost
%  grad       -  gradient with respect to theta (if pred==False)
%  preds      -  list of predictions for each example (if pred==True)

po = false;
preds = 0;
if exist('pred','var')
    po = pred;
end;
%% Reshape parameters and setup gradient matrices

image_size = size(images);
numImages = size(images,4);


stack = params2stack(theta, image_size, conv, full);

%% Convolutional Layer
activations = cell(numel(full),1); 
activationsPooled = cell(numel(full),1);

for  i  = 1 : numel(conv)
    
    if i == 1
    
    
    [activations{i}, activationsPooled{i}] = Forward_conv(conv{i},images, stack{i});
    else
        
     [activations{i}, activationsPooled{i}] = Forward_conv(conv{i}, activationsPooled{i-1}, stack{i});
 
     
     if i == numel(conv)
         
       activationsPooled{i} = reshape(activationsPooled{i},[],numImages);
       
     end
    end
end

%% Fc and Softmax Layer

for i = numel(conv)+1 : numel(full)
    
    if i == numel(full)
        
        [activations{i}, activationsPooled{i} ] = Forward_softmax(stack{i}, activationsPooled{i-1});
    else
        
        [activations{i}, activationsPooled{i}] = Forward_fc(stack{i}, activationsPooled{i-1});
    end
    
end


%% Calculate Cost

cost = Forward_cost(activations{i}, labels);

if po
    [~,preds] = max(activations{numel(activations)},[],1);
    preds = preds';
    grad = 0;
    return;
end;
%% Backpropagation


for i = numel(full)+1 : -1 : numel(conv)+2
    
 
    if i == numel(full)+1
         
        [stack_grad{i-1}.W, stack_grad{i-1}.b , delta{i}] = Backward_softmax(activationsPooled{i-1},...
                                                activationsPooled{i-2}, labels);
       
    else
        [stack_grad{i-1}.W, stack_grad{i-1}.b , delta{i}] = Backward_fc(activationsPooled{i-1}, activationsPooled{i-2},...        
                                                         stack{i}, delta{i+1});
        
    end
end

for i = numel(conv) + 1 : -1 : 2
    
    if i == numel(conv) + 1
        
       [stack_grad{i-1}.W, stack_grad{i-1}.b , delta{i}] = Backward_conv(activations{i-1}, activationsPooled{i-2},...
                                                         stack{i},delta{i+1},conv{i-1});

       
    else
        
               if  i == 2
                   
                   [stack_grad{i-1}.W, stack_grad{i-1}.b , delta{i}] = Backward_conv(activations{i-1}, images,...
                                                         stack{i}, delta{i+1}, conv{i-1}, conv{i});  
               else

                   [stack_grad{i-1}.W, stack_grad{i-1}.b , delta{i}] = Backward_conv(activations{i-1}, activationsPooled{i-2},...
                                                         stack{i}, delta{i+1}, conv{i-1}, conv{i});  
                                
               end
    end
    
end

                                                                                                          
%% Unroll gradient into grad vector for minFunc

grad = stack2params(stack_grad);

end

